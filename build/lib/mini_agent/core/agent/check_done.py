"""Check for completion"""

DONE_ACTIONS = ("done", "answer")


def is_done(action: str) -> bool:
    """Check whether the action means done"""
    return action in DONE_ACTIONS
