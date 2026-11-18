"""One process per session log.

A session log (~/.mini_agent/sessions/<name>.jsonl) is written by the one process that
runs it. While it runs, that process holds an exclusive lock on <name>.jsonl.lock, so a
second mini_agent on the same session (a second terminal, or a login item continuing the
folder's task while it still runs) is refused instead of writing into the same log.

The operating system drops the lock when its process ends, however it ends (kill -9
included), so a lock is never left behind; the lock file itself is removed when the lock
is released normally, and an old one left by a killed process is simply reused.
"""
import atexit
import os
from pathlib import Path
from typing import Optional

LOCK_OFFSET = 1 << 30   # Windows locks a byte range: one byte far past the pid written at 0


class SessionBusy(RuntimeError):
    """Another process holds this session."""


def lock_path(session) -> Path:
    session = Path(session)
    return session.with_name(session.name + ".lock")


def _lock(fd: int) -> bool:
    try:
        if os.name == "nt":
            import msvcrt
            os.lseek(fd, LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(fd: int) -> None:
    try:
        if os.name == "nt":
            import msvcrt
            os.lseek(fd, LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass


def _holder(fd: int) -> Optional[int]:
    """The pid the holder wrote at the start of the file, if it can be read."""
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        text = os.read(fd, 32).decode("ascii", "replace").strip()
        return int(text) if text.isdigit() else None
    except (OSError, ValueError):
        return None


def busy_message(session, pid: Optional[int] = None) -> str:
    by = f" (process {pid})" if pid else ""
    return (f"mini_agent: this session is in use by another mini_agent process{by}: {session}\n"
            "Wait for it to finish, or stop it, then try again.")


class SessionLock:
    """lock = SessionLock(path).acquire() holds the session until lock.release() or the end
    of the process; raises SessionBusy when another process holds it."""

    def __init__(self, session):
        self.session = Path(session)
        self.path = lock_path(session)
        self.fd: Optional[int] = None

    def acquire(self) -> "SessionLock":
        for _ in range(10):
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            if not _lock(fd):
                pid = _holder(fd)
                os.close(fd)
                raise SessionBusy(busy_message(self.session, pid))
            try:      # its last holder may have removed the file between our open and lock
                same = os.path.samestat(os.fstat(fd), os.stat(self.path))
            except OSError:
                same = False
            if same:
                break
            _unlock(fd)
            os.close(fd)
        else:
            raise SessionBusy(busy_message(self.session))
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"{os.getpid():<20}\n".encode("ascii"))    # fixed width: no stale tail
        self.fd = fd
        atexit.register(self.release)
        return self

    def release(self) -> None:
        if self.fd is None:
            return
        fd, self.fd = self.fd, None
        if os.name != "nt":
            try:          # removed while still held: nobody else can hold this file
                os.unlink(self.path)
            except OSError:
                pass
        _unlock(fd)
        os.close(fd)
        if os.name == "nt":
            try:          # fails harmlessly if another process opened it meanwhile
                os.unlink(self.path)
            except OSError:
                pass
