"""Execute tools"""
from typing import Callable, Dict, Optional


def execute(action: str, args: dict, tools: Dict[str, Callable],
            approve: Optional[Callable[[str, dict], bool]] = None) -> str:
    """Execute a tool call.

    approve(action, args) -> bool is an optional policy check. A denied call,
    or a policy that raises an exception, does not run; the reason is
    returned to the model as the observation.
    """
    if action not in tools:
        return f"Error: unknown tool '{action}'"
    if not isinstance(args, dict):
        return "Error: arguments must be a JSON object of named values"
    if approve is not None:
        try:
            allowed = approve(action, args)
        except Exception as e:
            return f"Error: the approval check failed ({e}); '{action}' did not run"
        if not allowed:
            return f"Error: '{action}' was not approved and did not run"
    try:
        return str(tools[action](**args))
    except Exception as e:
        return f"Error: {e}"
