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


def compact_notice(d: dict) -> str:
    """One line for a context/compact event (the lifecycle slimmed the earlier steps)."""
    return (f"✻ context compacted before step {d.get('step')}: {d.get('before_tokens', 0):,} → "
            f"{d.get('after_tokens', 0):,} tokens ({d.get('method', '?')})")


def stop_notice(d: dict) -> str:
    """The line after a task that stopped on its token budget, its step limit, a full
    context window, or an approval nobody could give; "" otherwise."""
    if d.get("reason") == "token_limit":
        return (f"This task used up its token budget ({d.get('detail', '')} tokens). "
                "`mini_agent -c` continues it with a new budget.")
    if d.get("reason") == "max_steps":
        return "This task reached its step limit. `mini_agent -c` continues it."
    if d.get("reason") == "needs_approval":
        return (f"This task needs {d.get('tool') or 'a tool that must be approved'}, and there is "
                "no terminal to ask for approval. `mini_agent -c --yes` continues it and allows "
                "it; or run `mini_agent -c` in a terminal to be asked.")
    if d.get("reason") == "context_full":
        return ("The next step of this task does not fit in the model's context window, even "
                "after compaction. Choose a model with a larger window and run `mini_agent -c`, "
                "or start a smaller task.")
    return ""


def session_conversation(agent, log, plain_text, until_seq=None) -> str:
    """The earlier conversation of the session for the next task: with a lifecycle, within
    its token budget (summarised by the model only when pruning is not enough); without
    one, the old rule (compact the log when it passes 2,500 characters).
    until_seq: the conversation as it was up to that event, for a task continued with -c
    (it gets the text it started with; read only, nothing is compacted)."""
    if getattr(agent, "lifecycle", None) is not None:
        return agent.lifecycle.conversation(log, until_seq=until_seq)
    if until_seq is not None:
        return plain_text(until_seq=until_seq)
    if len(plain_text(limit=999999)) > 2500:
        log.compact(agent.llm)
    return plain_text()


def mark_message(agent, log, seq) -> None:
    """Tell the lifecycle which logged user message the next run answers (session file and
    seq), so user.md records it once, even when -c runs a task again that was sent but
    never started."""
    if getattr(agent, "lifecycle", None) is not None and seq is not None:
        agent.lifecycle.message_id = f"{Path(log.path).name}#{seq}"


def _session_order(path: Path):
    """Sort key: <seconds>[_<n>] names (new_session_path) by seconds, then n as a number
    (so _100 comes after _99); any other name by its modification time."""
    m = re.fullmatch(r"(\d+)(?:_(\d+))?", path.stem)
    if m:
        return int(m.group(1)), int(m.group(2) or 1), path.name
    return int(path.stat().st_mtime), 0, path.name


def _has_message(path: Path) -> bool:
    """True when the log holds a user message or a started turn. A session that was
    opened and left without one (an empty file, or only a setting such as plan mode)
    has nothing to continue."""
    import json
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict) and event.get("type") in ("user/message", "turn/start"):
                    return True
    except OSError:
        return False
    return False


def latest_session(directory: Optional[Path] = None) -> Optional[Path]:
    """The most recent session log that holds a user message, or None."""
    files = [p for p in (directory or SESS_DIR).glob("*.jsonl") if p.is_file() and p.stat().st_size > 0]
    for path in sorted(files, key=_session_order, reverse=True):
        if _has_message(path):
            return path
    return None


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


def note_approval_now(state, unfinished: dict, approve, can_ask: bool) -> None:
    """A task that stopped because nobody could approve a risky call (needs_approval), now
    continued where that call can run: say so after the refused call's result, so that the
    model calls the tool again instead of working round it. approve: the policy of this run
    ("auto", "ask", "deny" or an Approval); can_ask: someone is there to answer "ask"."""
    if state is None or not state.steps or unfinished.get("reason") != "needs_approval":
        return
    policy = getattr(approve, "policy", approve)
    if policy == "auto":
        note = "Now it may run: the task was continued with risky tools allowed."
    elif policy == "ask" and can_ask:
        note = "Now it can be approved: the task was continued at a terminal, where the user is asked."
    else:
        return
    last = state.steps[-1]
    if "did not run: it needs approval" in str(last.get("result", "")):
        last["result"] = f"{last['result']}. {note} Call it again if it is still needed."


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
