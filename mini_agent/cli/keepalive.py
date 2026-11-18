"""`mini_agent --keep-alive [N] "task"` and `mini_agent -c --keep-alive [N]`: keep a long task going.

This process only supervises. The task runs in a child process (`python -m mini_agent` with
the same options), and every later run continues the same session with -c. When a run
ends, its events in the session log say what happened:

  done                               -> exit 0
  a stop that will not fix itself    -> that stop's exit status, no continue:
      a risky call needs approval and there is no terminal to ask (3); the token budget, the
      step limit, a full context window, a task offline mode has no recording of, invalid
      replies, or a model error that waiting will not change: HTTP 4xx other than 408 and 429
      (wrong key, no credit, unknown model, a request too large) or a missing setting (1);
      a usage or set-up error (2); the session in use by another process (4); a stop someone
      asked for: Ctrl+C, SIGTERM or SIGHUP (130, 143, 129)
  a passing failure                  -> wait, then continue with -c:
      the provider failed or could not be reached (no HTTP status, 408, 429, 5xx, an error
      inside a reply), or the run ended without a stop (killed, out of memory, crashed)

The waits grow: 30 s, 1, 2, 4, 8 min, then 10 min ($MINI_AGENT_KEEPALIVE_WAITS, seconds
separated by commas, replaces them). After N continues in a row that complete no step it
gives up with exit 75 ("try again later"); a run that completes a step starts the count
again. All runs share the task's token budget. The supervisor holds the session's lock all
the time (one process per session); on POSIX the running child shares it, so the lock
stays held if the supervisor is killed while its child still runs.
"""
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from ..llm.retry import retryable
from ..session import SessionLog
from ..session_lock import TOKEN_ENV, SessionBusy, SessionLock
from ..ui.common import EXIT_BUSY, SESS_DIR, latest_session, new_session_path

DEFAULT_TRIES = 5
WAITS = (30.0, 60.0, 120.0, 240.0, 480.0, 600.0)
EXIT_GAVE_UP = 75
WINDOWS_CTRL_C = 0xC000013A          # STATUS_CONTROL_C_EXIT: a console Ctrl+C ended the process
ASKED_TO_STOP = tuple(getattr(signal, n) for n in ("SIGINT", "SIGTERM", "SIGHUP") if hasattr(signal, n))
STOPPED_STATUS = {128 + s: s for s in ASKED_TO_STOP}    # a run that stopped itself on the signal
SETTING_ERRORS = ("Error: no server address", "Error: no model")
# Each run imports the package this supervisor runs from, never a mini_agent.py or mini_agent/
# in the task's folder (python -m would put that folder first on sys.path).
PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent.parent)
LAUNCH = ("import os, sys; sys.path[:1] = [p for p in sys.path[:1] if p]; root = {root!r}; "
          "same = lambda p: os.path.normcase(os.path.realpath(p)); "
          "any(same(p) == same(root) for p in sys.path) or sys.path.insert(0, root); "
          "from mini_agent.cli.main import main; sys.exit(main())")


class Stopped(BaseException):
    """SIGTERM or SIGHUP reached the supervisor."""

    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


_spawning = False         # a run's process is being started: hold a stop until it is known
_held: List[int] = []


def _stopped(signum, frame):
    """The supervisor's handler for Ctrl+C, SIGTERM and SIGHUP."""
    if _spawning:
        _held.append(signum)
        return
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    raise Stopped(signum)


def waits() -> List[float]:
    """The waits before continue 1, 2, ... (the last one repeats)."""
    raw = os.environ.get("MINI_AGENT_KEEPALIVE_WAITS", "").strip()
    if not raw:
        return list(WAITS)
    try:
        values = [float(x) for x in raw.split(",")]
    except ValueError:
        values = []
    if not values or any(v < 0 for v in values):
        raise ValueError("MINI_AGENT_KEEPALIVE_WAITS must be seconds separated by commas, "
                         f"such as 30,60,120; not {raw!r}")
    return values


def say(text: str) -> None:
    print(f"keep-alive: {text}", file=sys.stderr, flush=True)


def span(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s} s"
    m, r = divmod(s, 60)
    return f"{m} min" + (f" {r} s" if r else "")


def signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


def passing(error: str) -> bool:
    """A model error that waiting may fix: no HTTP status (the network, a timeout, a reply cut
    short), 408, 429, 5xx, or an error sent inside a reply; not a missing setting or another 4xx."""
    m = re.search(r"\bHTTP (\d{3})\b", error)
    if m is None:
        return not error.startswith(SETTING_ERRORS)
    status = int(m.group(1))
    return status == 408 or retryable(status) or status < 400


def outcome(rc: int, ended: Optional[dict]) -> Tuple[str, int, str]:
    """("done" | "stop" | "again", exit status, why) for a run that ended with exit status rc;
    ended: the data of the turn/end it logged, or None (it logged none)."""
    reason = (ended or {}).get("reason")
    if reason == "done":
        return "done", 0, "done"
    if rc == WINDOWS_CTRL_C or rc in STOPPED_STATUS or (rc < 0 and -rc in ASKED_TO_STOP):
        sig = -rc if rc < 0 else STOPPED_STATUS.get(rc, signal.SIGINT)
        return "stop", 128 + sig, f"the task was stopped ({signal_name(sig)})"
    if rc == 2:
        return "stop", 2, "the run could not start (a usage or set-up error)"
    if rc == EXIT_BUSY:
        return "stop", EXIT_BUSY, "the session is in use by another process"
    if ended is not None:
        if reason == "needs_approval":
            return "stop", 3, "a risky call needs approval"
        if reason == "llm_error":
            error = str(ended.get("answer", ""))
            why = "the model call failed: " + error.removeprefix("Error: ").splitlines()[0][:160]
            return ("again" if passing(error) else "stop"), 1, why
        return "stop", 1, f"the task stopped ({reason})"
    if rc < 0:
        return "again", 1, f"the run was ended by {signal_name(-rc)}"
    return "again", 1, f"the run ended before the task did (exit status {rc})"


def _events(path: Path, after: int = 0) -> list:
    return [e for e in SessionLog(path).events() if e["seq"] > after]


def _last_seq(path: Path) -> int:
    return max((e["seq"] for e in SessionLog(path).events()), default=0)


def _has_task(path: Path) -> bool:
    return any(e["type"] in ("user/message", "turn/start") for e in SessionLog(path).events())


def spent(events) -> int:
    """Tokens one run used: the counters in its step/think events are running totals for the
    run (lifecycle summaries included); a stop on the budget after a summary has them too."""
    used = 0
    for e in events:
        d = e.get("data") or {}
        try:
            if e["type"] == "step/think":
                used = max(used, int(d.get("tokens_in") or 0) + int(d.get("tokens_out") or 0))
            elif e["type"] == "turn/end" and d.get("reason") == "token_limit":
                m = re.match(r"\s*([\d,]+)\s*/", str(d.get("detail", "")))
                if m:
                    used = max(used, int(m.group(1).replace(",", "")))
        except (TypeError, ValueError):
            continue
    return used


def child_argv(args, session: Path, budget_left: int, first: bool) -> List[str]:
    """The command of one run: the same options, this session, the budget left; the task the
    first time, -c afterwards."""
    argv = [sys.executable, "-c", LAUNCH.format(root=PACKAGE_ROOT)]
    if args.yes or args.approve == "auto":
        argv.append("--yes")
    elif args.approve != "ask":
        argv += ["--approve", args.approve]
    if args.workspace:
        argv += ["--workspace", str(Path(args.workspace).resolve())]
    if not args.confine:
        argv.append("--no-confine")
    if not args.lifecycle:
        argv.append("--no-lifecycle")
    if args.coding:
        argv.append("--coding")
    if args.backend:
        argv += ["--backend", args.backend]
    if args.model:
        argv += ["--model", args.model]
    argv += ["--budget", str(budget_left), "--keepalive-session", str(session)]
    return argv + (["--", *args.task] if first else ["-c"])


def _end(proc: subprocess.Popen, signum: int) -> None:
    """Stop the running child. It usually got the same signal already (a terminal's Ctrl+C or a
    service stop reaches the whole process group): give it a moment, then pass the signal on,
    then end it."""
    for s in ASKED_TO_STOP:
        try:
            signal.signal(s, signal.SIG_IGN)
        except (OSError, ValueError):
            pass
    try:
        proc.wait(timeout=3)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "nt":
            proc.terminate()
        else:
            proc.send_signal(signum)
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _run(argv: List[str], lock: SessionLock) -> int:
    """Run one child and return its exit status (negative: the signal that ended it). A stop
    that comes while the child is being started is held until its process is known, then the
    child is stopped too."""
    global _spawning
    keep = {"pass_fds": (lock.fd,)} if os.name != "nt" and lock.fd is not None else {}
    env = dict(os.environ, **{TOKEN_ENV: lock.token})    # its way into the session we hold
    proc = None
    try:
        _spawning = True
        try:
            proc = subprocess.Popen(argv, env=env, **keep)
        finally:
            _spawning = False
        if _held:
            signum = _held[0]
            _held.clear()
            raise KeyboardInterrupt() if signum == signal.SIGINT else Stopped(signum)
        while True:
            try:
                return proc.wait(timeout=1)       # short waits: Ctrl+C is seen on Windows too
            except subprocess.TimeoutExpired:
                continue
    except KeyboardInterrupt:
        if proc is not None:
            _end(proc, signal.SIGINT)
        raise
    except Stopped as e:
        if proc is not None:
            _end(proc, e.signum)
        raise


def supervise(args, budget: int) -> int:
    """Run the task (args.task) or continue this folder's unfinished one (-c), keeping it going
    as the module docstring says. Returns the exit status."""
    tries = args.keep_alive
    try:
        table = waits()
    except ValueError as e:
        print(f"mini_agent: {e}", file=sys.stderr)
        return 2
    folder = Path(args.workspace or os.getcwd()).resolve()
    if args.task:
        session = new_session_path(SESS_DIR)
    else:
        session = latest_session(SESS_DIR, workspace=folder, unfinished=True)
        if session is None:              # nothing to do: not an error (a login item may run this)
            print(f"mini_agent: nothing unfinished to continue in this folder ({folder}).")
            return 0
    try:
        lock = SessionLock(session).acquire()
    except SessionBusy as e:
        print(e, file=sys.stderr)
        return EXIT_BUSY

    def stop_by(e) -> int:
        for s in ASKED_TO_STOP:          # one stop is enough: a second one must not cut this short
            try:
                signal.signal(s, signal.SIG_IGN)
            except (OSError, ValueError):
                pass
        sig = signal.SIGINT if isinstance(e, KeyboardInterrupt) else e.signum
        SessionLog(session).append("keepalive/stop", why=f"stopped by {signal_name(sig)}",
                                   exit_status=128 + sig)
        say(f"stopped ({signal_name(sig)}). `mini_agent -c` continues the task later.")
        return 128 + sig

    for s in ASKED_TO_STOP:
        try:
            signal.signal(s, _stopped)
        except (OSError, ValueError):
            pass
    try:
        return _keep_going(args, session, lock, budget, tries, table)
    except (KeyboardInterrupt, Stopped) as e:     # wherever it comes: a run, a wait, between
        return stop_by(e)


def _keep_going(args, session: Path, lock: SessionLock, budget: int, tries: int,
                table: List[float]) -> int:
    used = misses = runs = 0
    while True:
        first = bool(args.task) and not _has_task(session)
        before = _last_seq(session)
        rc = _run(child_argv(args, session, budget - used, first), lock)
        runs += 1
        events = _events(session, after=before)
        used += spent(events)
        if _has_task(session) and SessionLog(session).unfinished_turn() is None:
            return 0                                     # the task finished
        ended = next((e["data"] for e in reversed(events) if e["type"] == "turn/end"), None)
        kind, status, why = outcome(rc, ended)
        if kind == "done":
            return 0
        if kind == "stop":                               # the run already said why
            return status
        if any(e["type"] == "tool/result" for e in events):
            misses = 0                                   # it got somewhere: count again from 0
        elif runs > 1:
            misses += 1
        if misses >= tries:
            SessionLog(session).append("keepalive/stop", why=why, exit_status=EXIT_GAVE_UP)
            say(f"{tries} continue{'' if tries == 1 else 's'} in a row made no progress (the last: "
                f"{why}). Stopping for now: `mini_agent -c --keep-alive` continues the task later.")
            return EXIT_GAVE_UP
        left = budget - used
        if left <= 0:
            SessionLog(session).append("keepalive/stop", why="token budget used up", exit_status=1)
            say(f"the task used up its token budget ({used:,} of {budget:,} tokens). "
                "`mini_agent -c` continues it with a new budget.")
            return 1
        wait = table[min(misses, len(table) - 1)]
        SessionLog(session).append("keepalive/continue", attempt=misses + 1, of=tries, why=why,
                                   wait_s=wait, budget_left=left, exit_status=rc)
        say(f"{why}; continuing in {span(wait)} (try {misses + 1} of {tries}). Ctrl+C stops it.")
        time.sleep(wait)
