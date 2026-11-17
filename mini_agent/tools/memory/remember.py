"""Memory tools — let the agent remember on its own"""
from ...mem import append_memory, append_log
from datetime import datetime


def remember(content: str, evidence: str = "") -> str:
    """Save an important lesson to long-term memory

    Use for:
    - constraints: special requirements of a query
    - solutions: approaches that worked
    - pitfalls: lessons from failures

    Args:
        content: the note
        evidence: how you know it (a command and its output, a file and line, a source).
            A note with evidence is a checked fact; with mini_agent.lifecycle it is shown
            again, in full, after every compaction.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    if evidence is None or not str(evidence).strip():
        append_memory(f"- [{timestamp}] {content}")
        return f"Remembered: {content}"
    # stored next to the note; further lines indented, so the entry has no empty line
    note = "\n  ".join(str(content).strip().splitlines())
    proof = "\n    ".join(str(evidence).strip().splitlines())
    append_memory(f"- [{timestamp}] {note}\n  evidence: {proof}")
    return f"Remembered: {content} (with evidence)"


def log(content: str) -> str:
    """Append a note to today's log

    Use for:
    - tasks performed
    - intermediate results
    - transient info
    """
    timestamp = datetime.now().strftime("%H:%M")
    append_log(f"- [{timestamp}] {content}")
    return f"Logged: {content}"
