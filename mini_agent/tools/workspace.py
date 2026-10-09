"""The directory that relative tool paths refer to.

Agent.run() sets it to the agent's workspace for the duration of the run.
Outside a run it is unset, and paths are relative to the current directory.
"""
from contextvars import ContextVar
from pathlib import Path

current: ContextVar = ContextVar("workspace", default=None)


def resolve(path: str) -> Path:
    """Absolute paths stay as they are; relative paths start at the workspace."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else Path(current.get() or Path.cwd()) / p


def cwd() -> str:
    """The directory in which shell commands run"""
    return str(current.get() or Path.cwd())
