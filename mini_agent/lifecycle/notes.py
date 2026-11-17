"""The files of the written memory.

<workspace>/.mini_agent/user.md     every user message, verbatim, appended by code
<workspace>/.mini_agent/handoff.md  where a turn stands: goal, steps done, next step, files
                                    touched; "Status: open ..." while there is work to pick up
notes with evidence                 entries of the agent's long-term memory file written by
                                    remember(content, evidence)

Nothing is read or written through a symbolic link: .mini_agent must be a real directory
of the workspace. Where the system allows (POSIX), every file is opened through a handle
on that directory, so a link swapped in while a file is opened cannot send it elsewhere.
"""
import os
import re
from datetime import datetime
from pathlib import Path

USER_TITLE = "# User messages\n\nEvery message to the agent, verbatim, appended by mini_agent.\n"
# "## 3 · 2026-11-17 16:40:12 · 152 chars · id <message>": the length bounds the message, so a
# line in it that looks like a header does not end it; the id (optional) records a message
# sent again (a task continued with -c) once
_USER_ENTRY = re.compile(r"^## (\d+) · (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
                         r"(?: · (\d+) chars)?(?: · id (\S+))?\r?$", re.M)
_EVIDENCE = re.compile(r"^  evidence: ", re.M)
_STATUS = re.compile(r"^Status: (\w+)", re.M)
_STATUS_LINE = re.compile(r"^Status: .*$", re.M)
_SESSION = re.compile(r"^Session: (\S+)$", re.M)
# files opened relative to a handle on .mini_agent, never through a link (POSIX)
_AT = (os.open in os.supports_dir_fd and os.rename in os.supports_dir_fd
       and hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY"))
_FLAGS = {"r": os.O_RDONLY, "w": os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
          "a": os.O_WRONLY | os.O_CREAT | os.O_APPEND}


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _no_link(path: Path) -> Path:
    if path.is_symlink():
        raise OSError(f"{path} is a symbolic link; lifecycle files are not used through links")
    return path


def inside(workspace) -> Path:
    """<workspace>/.mini_agent if it is not a link and resolves inside the workspace; else OSError."""
    root = Path(workspace)
    path = _no_link(root / ".mini_agent")
    if not path.resolve().is_relative_to(root.resolve()):      # e.g. a junction
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


def _handle(folder: Path) -> int:
    """An open handle on .mini_agent itself; fails if it is a link (POSIX)."""
    root = os.open(folder.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        return os.open(folder.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root)
    finally:
        os.close(root)


def _open(path: Path, mode: str):
    """A text file of .mini_agent, opened without following a link; no newline translation."""
    _no_link(path)
    if not _AT:
        return open(path, mode, encoding="utf-8", newline="")
    handle = _handle(path.parent)
    try:
        fd = os.open(path.name, _FLAGS[mode] | os.O_NOFOLLOW, 0o666, dir_fd=handle)
    finally:
        os.close(handle)
    return open(fd, mode, encoding="utf-8", newline="")


def read(path: Path) -> str:
    """The file's text exactly as written ("" if missing)."""
    try:
        f = _open(path, "r")
    except FileNotFoundError:
        return ""
    with f:
        return f.read()


def _write(path: Path, text: str, mode: str = "w") -> None:
    with _open(path, mode) as f:
        f.write(text)


def write_atomic(path: Path, text: str) -> None:
    """Replace a file in one step, so that a crash never leaves half of it."""
    tmp = path.with_name(path.name + ".tmp")
    _write(tmp, text)
    if not _AT:
        os.replace(tmp, _no_link(path))
        return
    handle = _handle(path.parent)
    try:
        os.replace(tmp.name, path.name, src_dir_fd=handle, dst_dir_fd=handle)
    finally:
        os.close(handle)


def message_key(message_id) -> str:
    """A message id as written in a header: one word."""
    return re.sub(r"\s+", "_", str(message_id).strip()) if message_id is not None else ""


def append_user(path: Path, text: str, message_id=None) -> bool:
    """Append one user message, verbatim, as the next numbered entry. A message whose id is
    already recorded (sent again: a task continued with -c) is not appended. Returns whether
    it was appended."""
    old = read(path)
    entries = _entries(old)
    key = message_key(message_id)
    if key and any(entry[3] == key for entry in entries):
        return False
    head = f"## {len(entries) + 1} · {now()} · {len(text)} chars" + (f" · id {key}" if key else "")
    _write(path, ("" if old else USER_TITLE) + f"\n{head}\n{text}\n", "a")
    return True


def user_messages(path: Path) -> list:
    """The entries of user.md as (number, time, text), oldest first."""
    return [entry[:3] for entry in _entries(read(path))]


def entries(path: Path) -> list:
    """The entries of user.md as (number, time, text, id), oldest first."""
    return _entries(read(path))


def session_of(message_id):
    """The session a message id names: the part before its last "#" (None without an id)."""
    key = message_key(message_id)
    return key.rsplit("#", 1)[0] if key else None


def handoff_session(text: str):
    """The session that wrote a handoff (None for a handoff that does not say)."""
    found = _SESSION.search(text)
    return found.group(1) if found else None


def _entries(text: str) -> list:
    """(number, time, text, id) of each entry, oldest first."""
    entries, pos = [], 0
    while True:
        head = _USER_ENTRY.search(text, pos)
        if head is None:
            return entries
        start = head.end() + 1
        end = start + int(head.group(3)) if head.group(3) is not None else -1
        if end >= start and _whole(text, end):          # the length holds: exactly that text
            body, pos = text[start:end], end
        else:                    # no length (edited by hand) or a damaged entry: to the next header
            following = _USER_ENTRY.search(text, start)
            end = following.start() if following else len(text)
            body, pos = text[start:end].rstrip("\r\n"), end
        entries.append((int(head.group(1)), head.group(2), body, head.group(4)))


def _whole(text: str, end: int) -> bool:
    """An entry as append_user writes it ends here: a newline, then the end or the next entry."""
    return text[end:end + 1] == "\n" and (end + 1 == len(text) or text.startswith("\n## ", end + 1))


def open_handoff(path: Path) -> str:
    """The handoff without its title line when its status is open, else ""."""
    text = read(path)
    status = _STATUS.search(text)
    if not status or status.group(1) != "open":
        return ""
    return text.split("\n", 1)[1].strip() if text.startswith("# ") else text.strip()


def close_handoff(path: Path, status: str) -> None:
    """Replace the status line of an open handoff (e.g. "closed · ..."): it is not shown again."""
    if not open_handoff(path):
        return
    write_atomic(path, _STATUS_LINE.sub(lambda m: "Status: " + status, read(path), count=1))


def evidence_notes(text: str) -> list:
    """The entries of a memory file that carry evidence, oldest first.

    Entries are separated by an empty line (append_memory writes "\\n<entry>\\n");
    remember(content, evidence) indents every further line of an entry, so an
    entry with evidence never contains an empty line.
    """
    return [entry.strip("\n") for entry in text.split("\n\n") if _EVIDENCE.search(entry)]
