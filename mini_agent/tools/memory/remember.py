"""Memory tools — let the agent remember on its own"""
from ...mem import append_memory, append_log
from datetime import datetime


def remember(content: str) -> str:
    """Save an important lesson to long-term memory

    Use for:
    - constraints: special requirements of a query
    - solutions: approaches that worked
    - pitfalls: lessons from failures
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    append_memory(f"- [{timestamp}] {content}")
    return f"Remembered: {content}"


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
