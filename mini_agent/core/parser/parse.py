"""Parse the response"""
import json
from typing import Optional, Tuple


def parse_response(response: str) -> Tuple[str, Optional[str], dict]:
    """Parse the model's reply into (thought, action, args).

    Only the first JSON object with a string "action" field is used. Anything
    after it (a second action, or an observation the model wrote itself) is
    discarded: observations must come from tools that actually ran.
    If no such object exists, action is None and args["error"] says why;
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
    return response, None, {"error": 'the reply contains no JSON object with a string "action" field'}
