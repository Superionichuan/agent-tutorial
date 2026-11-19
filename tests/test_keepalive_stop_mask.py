"""The keep-alive supervisor holds the stop signals back while it polls its run.

Popen.poll() and Popen.wait() take an internal lock and release it in a finally block. A stop
(Ctrl+C, SIGTERM) raised between taking the lock and entering that block leaves the lock held;
every later wait on the process then blocks or times out. On macOS CI this hung the supervisor
in _end's last wait, with its run already a zombie. The supervisor now checks the run with
poll() while the stop signals are held back, takes a stop when the mask is restored (in its own
code, with the lock released), and bounds every wait in _end (the held lock itself is tested in
test_keepalive_waitlock.py).

POSIX only: Windows has no pthread_sigmask, and its Popen.wait takes no such lock.
"""
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import pytest

from mini_agent.cli import keepalive
from test_keepalive_blackbox import children, done, in_order, make, wait_until  # noqa: F401

pytestmark = pytest.mark.skipif(not keepalive.SHIELD, reason="POSIX signal masks")

STOPS = set(keepalive.ASKED_TO_STOP)


def blocked_now():
    return set(signal.pthread_sigmask(signal.SIG_BLOCK, []))


class Polled:
    """A stand-in for the run: poll() records the mask, and can send this process a signal."""

    def __init__(self, send=None, status=None):
        self.send, self.status = send, status
        self.mask = None
        self.finished = False

    def poll(self):
        self.mask = blocked_now()
        if self.send is not None:
            # Arrives while poll runs. Sent to this thread: a mask covers only the thread that
            # set it, and the test process may have other threads (the supervisor has none).
            signal.pthread_kill(threading.get_ident(), self.send)
            time.sleep(0.05)
        self.finished = True                     # reached only if the signal did not interrupt
        return self.status


def test_the_stop_signals_are_held_back_while_poll_runs():
    before = blocked_now()
    proc = Polled(status=7)
    assert keepalive._poll_shielded(proc) == 7
    assert STOPS <= proc.mask
    assert blocked_now() == before and not (STOPS & before)


@pytest.mark.parametrize("sig", sorted(STOPS & {signal.SIGINT, signal.SIGTERM}),
                         ids=lambda s: signal.Signals(s).name)
def test_a_stop_during_poll_is_raised_after_it_in_the_supervisor(sig):
    saved = signal.signal(sig, keepalive._stopped)
    try:
        proc = Polled(send=sig)
        expected = KeyboardInterrupt if sig == signal.SIGINT else keepalive.Stopped
        with pytest.raises(expected) as info:
            keepalive._poll_shielded(proc)
    finally:
        signal.signal(sig, saved)
    assert proc.finished                                  # poll ran to its end
    frames = traceback.extract_tb(info.tb)
    names = [f.name for f in frames]
    assert "poll" not in names                            # nothing raised inside poll
    # The handler ran while the mask was restored: _poll_shielded -> signal.pthread_sigmask
    # (the standard library's wrapper around the C call that runs the handler) -> _stopped.
    at = names.index("_poll_shielded")
    assert "pthread_sigmask(signal.SIG_SETMASK" in frames[at].line
    assert names[at + 1:] == ["pthread_sigmask", "_stopped"]
    assert Path(frames[at + 1].filename).name == "signal.py"


@pytest.mark.skipif(not Path("/proc/self/status").exists(), reason="reads /proc (Linux)")
def test_the_run_does_not_inherit_blocked_stop_signals(make):
    hold = threading.Event()
    s = make(in_order(hold, then=done("too late")))
    proc = s.start("--keep-alive", "Compute 6*7.")
    try:
        wait_until(lambda: s.server.count() == 1, what="the first request")
        child = wait_until(lambda: children(proc.pid), what="the run's process")[0]
        status = Path(f"/proc/{child}/status").read_text()
        blocked = int(next(line.split()[1] for line in status.splitlines()
                           if line.startswith("SigBlk:")), 16)
        assert not [s_ for s_ in STOPS if blocked & (1 << (s_ - 1))], status
    finally:
        hold.set()
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(30)


def test_end_says_when_it_stops_waiting_for_a_killed_run(monkeypatch, capsys):
    """With the lock left held, _end returns after its bounded waits and says why."""
    monkeypatch.setattr(keepalive, "END_WAITS", (0.2, 0.2, 0.5))
    saved = {s: signal.getsignal(s) for s in STOPS}       # _end ignores further stops
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    proc._waitpid_lock.acquire()
    try:
        t0 = time.monotonic()
        keepalive._end(proc, signal.SIGINT)
        took = time.monotonic() - t0
    finally:
        proc._waitpid_lock.release()
        proc.kill()
        proc.wait(10)
        for s, handler in saved.items():
            signal.signal(s, handler)
    assert took < 5, took
    assert "was killed but has not been reaped" in capsys.readouterr().err
