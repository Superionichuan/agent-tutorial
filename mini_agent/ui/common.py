"""What the plain terminal mode and the full-screen interface share (no textual needed)."""
import os
import re
from pathlib import Path
from typing import Optional

# One JSONL event log per interactive session; `mini_agent -c` continues the latest.
SESS_DIR = Path.home() / ".mini_agent_sessions"

PLAN_GUIDANCE = ("[Plan mode] You are in plan mode: explore and plan only; do not run tools "
                 "with side effects (file writes / mutating bash). Produce a complete step-by-step "
                 "plan first (markdown, starting with a # heading) and wait for user confirmation.\n")


def _session_order(path: Path):
    """Sort key: <seconds>[_<n>] names (new_session_path) by seconds, then n as a number
    (so _100 comes after _99); any other name by its modification time."""
    m = re.fullmatch(r"(\d+)(?:_(\d+))?", path.stem)
    if m:
        return int(m.group(1)), int(m.group(2) or 1), path.name
    return int(path.stat().st_mtime), 0, path.name


def latest_session(directory: Optional[Path] = None) -> Optional[Path]:
    """The most recent session log that has any events, or None. An empty file is a
    session that was opened and left without a message: there is nothing to continue."""
    files = [p for p in (directory or SESS_DIR).glob("*.jsonl")
             if p.is_file() and p.stat().st_size > 0]
    return max(files, key=_session_order, default=None)


def resume_state(unfinished: dict):
    """The State to continue an unfinished turn with (SessionLog.unfinished_turn()):
    the steps rebuilt from its events, in-turn compactions applied; a tool call that
    has no result is not run again (the model is told it did not finish). None for a
    pending task (its turn never began): run it as a new turn."""
    if unfinished.get("pending"):
        return None
    from ..core.state import State
    from ..lifecycle import restore_steps
    restored = restore_steps(unfinished["events"])
    if isinstance(restored, State):
        return restored
    return State(goal=unfinished["goal"], steps=list(restored))


def new_session_path(directory: Optional[Path] = None) -> Path:
    """A fresh session log, named by the time so that the latest sorts last; a second
    session in the same second gets _02, _03, ... The empty file is created here, at once,
    so two sessions (in one program or two) never get the same name."""
    import time
    directory = directory or SESS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp, n = str(int(time.time())), 1
    while True:
        path = directory / (f"{stamp}.jsonl" if n == 1 else f"{stamp}_{n:02d}.jsonl")
        try:
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL))
            return path
        except FileExistsError:
            n += 1
