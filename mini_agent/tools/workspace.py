"""The directory that relative tool paths refer to, and the limit on file tools.

Agent.run() sets the workspace for the duration of the run, and with
Agent(confine=True) (the default) also confines the file tools to it: a path
that resolves outside the workspace, also through a symlink, is refused.
Outside a run nothing is set: paths are relative to the current directory and
not limited.
"""
from contextvars import ContextVar
from pathlib import Path

current: ContextVar = ContextVar("workspace", default=None)
confined: ContextVar = ContextVar("workspace_confined", default=False)


class OutsideWorkspace(PermissionError):
    """A file tool was asked for a path outside the workspace."""


def resolve(path: str) -> Path:
    """Absolute paths stay as they are; relative paths start at the workspace.

    While the file tools are confined, a path whose real location (after
    following symlinks) is outside the workspace raises OutsideWorkspace.
    """
    p = Path(path).expanduser()
    base = Path(current.get() or Path.cwd())
    target = p if p.is_absolute() else base / p
    if confined.get():
        try:
            real, root = target.resolve(), base.resolve()
        except (OSError, RuntimeError) as e:          # e.g. a symlink loop
            raise OutsideWorkspace(f"cannot resolve {path}: {e}") from None
        if real != root and not real.is_relative_to(root):
            raise OutsideWorkspace(f"{path} is outside the workspace ({root}); "
                                   "file tools only work inside it")
    return target


def cwd() -> str:
    """The directory in which shell commands run"""
    return str(current.get() or Path.cwd())
