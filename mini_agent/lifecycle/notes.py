"""The files of the written memory.

<workspace>/.mini_agent/user.md     every user message, verbatim, appended by code
<workspace>/.mini_agent/handoff.md  where a turn stands: goal, steps done, next step, files
                                    touched; "Status: open ..." while there is work to pick up
notes with evidence                 entries of the agent's long-term memory file written by
                                    remember(content, evidence)

Nothing is read or written through a symbolic link, and .mini_agent must resolve inside the
workspace: a link planted there cannot send these writes (or reads) elsewhere.
"""
import os
import re
from datetime import datetime
from pathlib import Path

USER_TITLE = "# User messages\n\nEvery message to the agent, verbatim, appended by mini_agent.\n"
# "## 3 · 2026-11-17 16:40:12 · 152 chars": the length bounds the message, so a line in a
# message that looks like a header does not end it
_USER_ENTRY = re.compile(r"^## (\d+) · (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?: · (\d+) chars)?$", re.M)
_EVIDENCE = re.compile(r"^  evidence: ", re.M)
_STATUS = re.compile(r"^Status: (\w+)", re.M)
_STATUS_LINE = re.compile(r"^Status: .*$", re.M)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def inside(workspace) -> Path:
    """<workspace>/.mini_agent, if it resolves inside the workspace; else OSError."""
    root = Path(workspace)
    path = root / ".mini_agent"
    if not path.resolve().is_relative_to(root.resolve()):
        raise OSError(f"{path} leads outside the workspace; lifecycle files are not used")
    return path


def folder(workspace) -> Path:
    """<workspace>/.mini_agent, created with a .gitignore so that git leaves it alone."""
    path = inside(workspace)
    path.mkdir(parents=True, exist_ok=True)
    path = inside(workspace)                     # made by someone else meanwhile: check again
    ignore = path / ".gitignore"
    if not ignore.exists() and not ignore.is_symlink():
        _write(ignore, "*\n")
    return path


def _plain(path: Path) -> Path:
    if path.is_symlink():
        raise OSError(f"{path} is a symbolic link; lifecycle files are not used through links")
    return path


def _read(path: Path) -> str:
    """The file's text exactly as written ("" if missing); no newline translation."""
    if not _plain(path).exists():
        return ""
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def _write(path: Path, text: str, mode: str = "w") -> None:
    with open(_plain(path), mode, encoding="utf-8", newline="") as f:
        f.write(text)


def append_user(path: Path, text: str) -> None:
    """Append one user message, verbatim, as the next numbered entry."""
    old = _read(path)
    n = len(user_messages(path)) + 1
    _write(path, ("" if old else USER_TITLE) + f"\n## {n} · {now()} · {len(text)} chars\n{text}\n",
           "a")


def user_messages(path: Path) -> list:
    """The entries of user.md as (number, time, text), oldest first."""
    text = _read(path)
    entries, pos = [], 0
    while True:
        head = _USER_ENTRY.search(text, pos)
        if head is None:
            return entries
        start = head.end() + 1
        if head.group(3) is not None:                       # length given: exactly that text
            end = start + int(head.group(3))
            body, pos = text[start:end], end
        else:                                               # edited by hand: up to the next header
            following = _USER_ENTRY.search(text, start)
            end = following.start() if following else len(text)
            body, pos = text[start:end].rstrip("\n"), end
        entries.append((int(head.group(1)), head.group(2), body))


def write_atomic(path: Path, text: str) -> None:
    """Replace a file in one step, so that a crash never leaves half of it."""
    tmp = path.with_name(path.name + ".tmp")
    _write(tmp, text)
    os.replace(tmp, _plain(path))


def open_handoff(path: Path) -> str:
    """The handoff without its title line when its status is open, else ""."""
    text = _read(path)
    status = _STATUS.search(text)
    if not status or status.group(1) != "open":
        return ""
    return text.split("\n", 1)[1].strip() if text.startswith("# ") else text.strip()


def close_handoff(path: Path, status: str) -> None:
    """Replace the status line of an open handoff (e.g. "closed · ..."): it is not shown again."""
    if not open_handoff(path):
        return
    write_atomic(path, _STATUS_LINE.sub(lambda m: "Status: " + status, _read(path), count=1))


def evidence_notes(text: str) -> list:
    """The entries of a memory file that carry evidence, oldest first.

    Entries are separated by an empty line (append_memory writes "\\n<entry>\\n");
    remember(content, evidence) indents every further line of an entry, so an
    entry with evidence never contains an empty line.
    """
    return [entry.strip("\n") for entry in text.split("\n\n") if _EVIDENCE.search(entry)]
