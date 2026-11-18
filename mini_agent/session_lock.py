"""One process per session log.

A session log (~/.mini_agent/sessions/<name>.jsonl) is written by the one process that
runs it. While it runs, that process holds an exclusive lock on <name>.jsonl.lock, so a
second mini_agent on the same session (a second terminal, or a login item continuing the
folder's task while it still runs) is refused instead of writing into the same log.

The operating system drops the lock when its process ends, however it ends (kill -9
included), so a lock is never left behind; the lock file itself is removed when the lock
is released normally, and an old one left by a killed process is simply reused.

The holder writes its pid and a random token in the lock file. The runs of `--keep-alive`
work in the session that their supervisor holds; they get the token in TOKEN_ENV, and a
process without it cannot use their way in.

On a file system that cannot lock files at all (some network file systems), the session is
used without the check, after one warning.
"""
import atexit
import errno
import os
import secrets
import sys
from pathlib import Path
from typing import Optional

LOCK_OFFSET = 1 << 30   # Windows locks a byte range: one byte far past the pid written at 0
TOKEN_ENV = "MINI_AGENT_KEEPALIVE_TOKEN"
# the errors that mean "another process holds it" (msvcrt reports a held range as EACCES)
BUSY_ERRNOS = {errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES,
               getattr(errno, "EDEADLK", errno.EAGAIN)}
_warned = False


class SessionBusy(RuntimeError):
    """Another process holds this session."""


def lock_path(session) -> Path:
    session = Path(session)
    return session.with_name(session.name + ".lock")


def _lock(fd: int) -> Optional[OSError]:
    """None when locked; else the error (BUSY_ERRNOS: another process holds it)."""
    try:
        if os.name == "nt":
            import msvcrt
            os.lseek(fd, LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return None
    except OSError as e:
        return e


def _cannot_lock(e: OSError) -> None:
    global _warned
    if not _warned:
        _warned = True
        print(f"mini_agent: this file system cannot lock files ({e.strerror or e}); going on "
              "without the check that only one mini_agent process uses a session.",
              file=sys.stderr)


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
        text = os.read(fd, 64).decode("ascii", "replace").split("\n", 1)[0].strip()
        return int(text) if text.isdigit() else None
    except (OSError, ValueError):
        return None


def holder_token(session) -> Optional[str]:
    """The token in the session's lock file (the second line), if any."""
    try:
        with open(lock_path(session), "rb") as f:
            lines = f.read(256).decode("ascii", "replace").split("\n")
    except OSError:
        return None
    token = lines[1].strip() if len(lines) > 1 else ""
    return token or None


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
        self.token = secrets.token_hex(16)

    def acquire(self) -> "SessionLock":
        for _ in range(10):
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            error = _lock(fd)
            if error is not None and error.errno in BUSY_ERRNOS:
                pid = _holder(fd)
                os.close(fd)
                raise SessionBusy(busy_message(self.session, pid))
            if error is not None:          # this file system cannot lock: no check, say so once
                _cannot_lock(error)
                break
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
        os.lseek(fd, 0, os.SEEK_SET)      # fixed width: no stale tail
        os.write(fd, f"{os.getpid():<20}\n{self.token}\n".encode("ascii"))
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
