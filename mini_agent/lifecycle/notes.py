"""The files of the written memory.

<workspace>/.mini_agent/user.md     every user message, verbatim, appended by code
<workspace>/.mini_agent/handoff.md  where a turn stands: goal, steps done, next step, files
                                    touched; "Status: open ..." while there is work to pick up
notes with evidence                 entries of the agent's long-term memory file written by
                                    remember(content, evidence)
"""
import os
import re
from datetime import datetime
from pathlib import Path

USER_TITLE = "# User messages\n\nEvery message to the agent, verbatim, appended by mini_agent.\n"
_USER_ENTRY = re.compile(r"^## (\d+) · (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})$", re.M)
_EVIDENCE = re.compile(r"^  evidence: ", re.M)
_STATUS = re.compile(r"^Status: (\w+)", re.M)
_STATUS_LINE = re.compile(r"^Status: .*$", re.M)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def folder(workspace) -> Path:
    """<workspace>/.mini_agent, created with a .gitignore so that git leaves it alone."""
    path = Path(workspace) / ".mini_agent"
    path.mkdir(parents=True, exist_ok=True)
    ignore = path / ".gitignore"
    if not ignore.exists():
        ignore.write_text("*\n", encoding="utf-8")
    return path


def append_user(path: Path, text: str) -> None:
    """Append one user message, verbatim, as the next numbered entry."""
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    n = len(_USER_ENTRY.findall(old)) + 1
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        if not old:
            f.write(USER_TITLE)
        f.write(f"\n## {n} · {now()}\n{text}\n")


def user_messages(path: Path) -> list:
    """The entries of user.md as (number, time, text), oldest first."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    heads = list(_USER_ENTRY.finditer(text))
    entries = []
    for i, head in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        entries.append((int(head.group(1)), head.group(2), text[head.end() + 1:end].rstrip("\n")))
    return entries


def write_atomic(path: Path, text: str) -> None:
    """Replace a file in one step, so that a crash never leaves half of it."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def open_handoff(path: Path) -> str:
    """The handoff without its title line when its status is open, else ""."""
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    status = _STATUS.search(text)
    if not status or status.group(1) != "open":
        return ""
    return text.split("\n", 1)[1].strip() if text.startswith("# ") else text.strip()


def close_handoff(path: Path, status: str) -> None:
    """Replace the status line of an open handoff (e.g. "closed · ..."): it is not shown again."""
    if not open_handoff(path):
        return
    text = path.read_text(encoding="utf-8")
    write_atomic(path, _STATUS_LINE.sub(lambda m: "Status: " + status, text, count=1))


def evidence_notes(text: str) -> list:
    """The entries of a memory file that carry evidence, oldest first.

    Entries are separated by an empty line (append_memory writes "\\n<entry>\\n");
    remember(content, evidence) indents every further line of an entry, so an
    entry with evidence never contains an empty line.
    """
    return [entry.strip("\n") for entry in text.split("\n\n") if _EVIDENCE.search(entry)]
