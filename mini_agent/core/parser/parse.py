"""Parse the response"""
import json
from typing import Tuple


def parse_response(response: str) -> Tuple[str, str, dict]:
    """Parse the LLM response into (thought, action, args)"""
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(response[start:end])
            return (
                data.get("thought", ""),
                data.get("action", "done"),
                data.get("args", {})
            )
    except:
        pass
    return response, "done", {}
