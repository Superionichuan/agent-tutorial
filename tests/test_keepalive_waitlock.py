"""keep-alive: stopping a run ends even when Popen's wait lock was left held.

CI (macOS, Python 3.11) hung in the supervisor's last proc.wait(): a stop signal handled right
after Popen._wait took its internal _waitpid_lock (before its try:) leaves that lock held. From
then on wait(timeout) always times out, even once the child has exited, and wait() never
returns. Here the test holds the lock itself and checks that _end still ends the child and
comes back within its bounded waits (END_WAITS, shortened).
Run: python -m pytest tests/test_keepalive_waitlock.py
"""
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from mini_agent.cli import keepalive  # noqa: E402

pytestmark = pytest.mark.skipif(os.name == "nt", reason="Popen's _waitpid_lock exists on POSIX only")

WAITS = (0.2, 0.2, 0.5)        # _end's three waits, shortened: before the signal, after it, after SIGKILL
CHILDREN = {
    # the CI case: the run has exited and is a zombie nobody can reap
    "exited": "pass",
    # a run that ignores the stop signal: only SIGKILL ends it
    "ignores the signal": "import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(60)",
}


@pytest.mark.parametrize("kind", list(CHILDREN))
def test_end_comes_back_when_the_wait_lock_is_held(kind, monkeypatch):
    monkeypatch.setattr(keepalive, "END_WAITS", WAITS, raising=False)
    child = subprocess.Popen([sys.executable, "-c", CHILDREN[kind]])
    time.sleep(0.5)                                  # started (and, for "exited", ended)
    lock = child._waitpid_lock
    assert lock.acquire(timeout=5)                   # what the interrupted wait left behind
    ender = threading.Thread(target=keepalive._end, args=(child, signal.SIGINT), daemon=True)
    started = time.time()
    try:
        ender.start()                                # (signal.signal only works in the main thread:
        ender.join(sum(WAITS) + 4)                   #  _end leaves this test's handlers alone)
        assert not ender.is_alive(), f"_end still waiting after {time.time() - started:.1f} s"
    finally:
        lock.release()                               # whatever happened, let every wait finish
    ender.join(15)
    assert child.wait(timeout=10) == (0 if kind == "exited" else -signal.SIGKILL)
