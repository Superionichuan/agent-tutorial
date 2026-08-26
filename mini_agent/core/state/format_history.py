"""Format history"""
from .state import State


def format_history(state: State, max_tokens: int = 80000) -> str:
    """Format step history, keeping most recent steps within the token budget"""
    if not state.steps:
        return "(none)"

    # Walk from the most recent step, accumulate until the token budget
    lines = []
    total_chars = 0
    for s in reversed(state.steps):
        line = f"{s['action']}({s['args']}) → {s['result']}"
        # Rough estimate: ~2 chars = 1 token
        if total_chars + len(line) > max_tokens * 2:
            break
        lines.append(line)
        total_chars += len(line)

    # Restore chronological order, add indices
    lines = list(reversed(lines))
    start = len(state.steps) - len(lines) + 1
    return "\n".join(f"{start + i}. {line}" for i, line in enumerate(lines))
