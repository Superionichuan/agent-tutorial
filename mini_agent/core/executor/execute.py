"""Execute tools"""
from typing import Callable, Dict, Optional


def execute(action: str, args: dict, tools: Dict[str, Callable],
            approve: Optional[Callable[[str, dict], bool]] = None) -> str:
    """Execute a tool call.

    approve(action, args) -> bool is an optional policy check. A denied call
    is not executed; the denial is returned to the model as the observation.
    """
    if action not in tools:
        return f"Error: unknown tool '{action}'"
    if approve is not None and not approve(action, args):
        return f"Error: '{action}' was not approved and did not run"
    try:
        return str(tools[action](**args))
    except Exception as e:
        return f"Error: {e}"
