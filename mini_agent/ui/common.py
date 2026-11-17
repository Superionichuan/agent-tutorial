"""What the plain terminal mode and the full-screen interface share (no textual needed)."""
from pathlib import Path
from typing import Optional

# One JSONL event log per interactive session; `mini_agent -c` continues the latest.
SESS_DIR = Path.home() / ".mini_agent_sessions"

PLAN_GUIDANCE = ("[Plan mode] You are in plan mode: explore and plan only; do not run tools "
                 "with side effects (file writes / mutating bash). Produce a complete step-by-step "
                 "plan first (markdown, starting with a # heading) and wait for user confirmation.\n")


def latest_session(directory: Optional[Path] = None) -> Optional[Path]:
    """The most recent session log, or None."""
    files = sorted((directory or SESS_DIR).glob("*.jsonl"))
    return files[-1] if files else None


def resume_state(unfinished: dict):
    """The State to continue an unfinished turn with (SessionLog.unfinished_turn()):
    the steps rebuilt from its events, in-turn compactions applied; a tool call that
    has no result is not run again (the model is told it did not finish)."""
    from ..core.state import State
    from ..lifecycle import restore_steps
    restored = restore_steps(unfinished["events"])
    if isinstance(restored, State):
        return restored
    return State(goal=unfinished["goal"], steps=list(restored))


def new_session_path(directory: Optional[Path] = None) -> Path:
    """A fresh session log path, named by the time so that the latest sorts last;
    a second session in the same second gets _02, _03, ... (which sort after it)."""
    import time
    directory = directory or SESS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = str(int(time.time()))
    path, n = directory / f"{stamp}.jsonl", 1
    while path.exists():
        n += 1
        path = directory / f"{stamp}_{n:02d}.jsonl"
    return path
