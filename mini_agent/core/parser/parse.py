"""Parse the response"""
import json
import re
from typing import Optional, Tuple

_INVOKE = re.compile(r'invoke\s+name="([^"]+)"')
_PARAM = re.compile(r'parameter\s+name="([^"]+)"[^>]*>(.*?)</', re.S)
_ACTION_START = re.compile(r'\{\s*"(?:thought|action)"\s*:')   # how a reply object begins
_CODE_ARG = re.compile(r'"(?:cmd|code)"\s*:')
LONG_REPLY = 800          # characters: a reply this long with a cmd or code argument carries code
CODE_HINT = ("Put code in write_file's content (its own argument), then run the file; "
             "don't put multi-line programs inside bash quotes. Inside a JSON string, "
             'write \\" for a double quote and \\n for a newline')


def _syntax_error(response: str, start: int, err: json.JSONDecodeError) -> str:
    """Why a reply object that begins like an action does not parse, and where."""
    near = response[max(start, err.pos - 20):err.pos + 8]
    text = f"the reply's JSON does not parse: {err.msg} at character {err.pos}, near `{near}`"
    if len(response) >= LONG_REPLY and _CODE_ARG.search(response):
        text += f". {CODE_HINT}"
    return text


def _from_markup(response: str):
    """Some models reply in their provider's tool-call markup instead of JSON text.

    The markup names a tool and its parameters; this reads the first invocation.
    """
    m = _INVOKE.search(response)
    if not m:
        return None
    args = {}
    for name, value in _PARAM.findall(response[m.end():]):
        value = value.strip()
        if name == "args":
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    args.update(parsed)
                    continue
            except json.JSONDecodeError:
                pass
        args[name] = value
    return response[:m.start()].strip(), m.group(1), args


def parse_response(response: str) -> Tuple[str, Optional[str], dict]:
    """Parse the model's reply into (thought, action, args).

    Only the first JSON object with a string "action" field is used. Anything
    after it (a second action, or an observation the model wrote itself) is
    discarded: observations must come from tools that actually ran.
    A reply in tool-call markup is read the same way (first invocation only).
    If neither form is present, action is None and args["error"] says why;
    the loop returns that error to the model instead of executing anything.
    When the reply has an object that begins like an action ({"thought": or
    {"action":) but does not parse, the error gives the JSON error and where,
    and args["json_syntax"] is True.
    """
    decoder = json.JSONDecoder()
    start = response.find("{")
    syntax = None             # the first object that begins like an action but does not parse
    while start >= 0:
        try:
            data, _ = decoder.raw_decode(response, start)
        except json.JSONDecodeError as err:
            data = None
            if syntax is None and _ACTION_START.match(response, start):
                syntax = _syntax_error(response, start, err)
        if isinstance(data, dict) and isinstance(data.get("action"), str):
            args = data.get("args")
            thought = data.get("thought")
            thought = "" if thought is None else str(thought)
            return thought, data["action"], args if args is not None else {}
        start = response.find("{", start + 1)
    markup = _from_markup(response)
    if markup is not None:
        return markup
    if syntax is not None:
        return response, None, {"error": syntax, "json_syntax": True}
    return response, None, {"error": 'the reply contains no JSON object with a string "action" field'}
