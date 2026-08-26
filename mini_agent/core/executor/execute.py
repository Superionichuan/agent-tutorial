"""Execute tools"""
from typing import Callable, Dict


def execute(action: str, args: dict, tools: Dict[str, Callable]) -> str:
    """Execute a tool call"""
    if action not in tools:
        return f"Unknown tool: {action}"
    try:
        return str(tools[action](**args))
    except Exception as e:
        return f"Error: {e}"
