"""Append a step"""
from .state import State


def add_step(state: State, thought: str, action: str, args: dict, result: str):
    """Append an executed step"""
    state.steps.append({
        "thought": thought,
        "action": action,
        "args": args,
        "result": result
    })
