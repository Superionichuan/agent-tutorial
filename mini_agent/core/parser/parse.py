"""Parse the response"""
import json
import re
from typing import Optional, Tuple

_INVOKE = re.compile(r'invoke\s+name="([^"]+)"')
_PARAM = re.compile(r'parameter\s+name="([^"]+)"[^>]*>(.*?)</', re.S)


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
    """
    decoder = json.JSONDecoder()
    start = response.find("{")
    while start >= 0:
        try:
            data, _ = decoder.raw_decode(response, start)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("action"), str):
            args = data.get("args")
            thought = data.get("thought")
            thought = "" if thought is None else str(thought)
            return thought, data["action"], args if args is not None else {}
        start = response.find("{", start + 1)
    markup = _from_markup(response)
    if markup is not None:
        return markup
    return response, None, {"error": 'the reply contains no JSON object with a string "action" field'}
