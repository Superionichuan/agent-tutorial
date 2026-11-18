"""Keep-alive review: cases where the supervisor does the wrong thing.

Each test describes one case and the behaviour expected instead. The black-box tests start the
real command line against a fake model server (helpers from test_keepalive_blackbox.py); two
tests call the supervisor or the session lock directly, to put a signal or a filesystem error
exactly where it can happen.
"""
import errno
import http.server
import json
import os
import signal
import subprocess
import sys
import threading

import pytest

from test_keepalive_blackbox import POSIX, done, in_order, make  # noqa: F401 (make is a fixture)


# ---------- the runs import another mini_agent than the supervisor ----------

def test_a_mini_agent_py_in_the_folder_does_not_replace_the_agent(make, tmp_path):
    """`mini_agent` (the installed command) imports the installed package; each run is
    `python -m mini_agent`, which puts the current folder first on sys.path. A file or folder
    named mini_agent in the working folder (a script of one's own, a clone of the repository)
    is then what every run executes. Expected: the runs use the supervisor's package."""
    s = make(in_order(done("finished")))
    (s.work / "mini_agent.py").write_text(
        "from pathlib import Path\nPath('impostor_ran.txt').write_text('yes')\n", encoding="utf-8")
    launcher = tmp_path / "bin" / "launch.py"          # like the installed `mini_agent` script
    launcher.parent.mkdir()
    launcher.write_text("import sys\nfrom mini_agent.cli import main\nsys.exit(main())\n",
                        encoding="utf-8")
    out, err = open(s.logs / "launch.out", "w+"), open(s.logs / "launch.err", "w+")
    proc = subprocess.run([sys.executable, str(launcher), "--yes", "--keep-alive", "1",
                           "Say finished."], cwd=s.work, env=s.env, stdin=subprocess.DEVNULL,
                          stdout=out, stderr=err, timeout=120)
    out.seek(0)
    err.seek(0)
    shown = out.read() + err.read()
    assert not (s.work / "impostor_ran.txt").exists(), "a run executed ./mini_agent.py\n" + shown
    assert proc.returncode == 0 and "finished" in shown, shown
    assert s.server.count() == 1


# ---------- a permanent model error the Claude backend reports without its status ----------

def anthropic_server(status, error_type, message):
    """A fake Anthropic API that answers every request with this error; .hits counts them."""
    body = json.dumps({"type": "error", "error": {"type": error_type, "message": message}})

    class Handler(http.server.BaseHTTPRequestHandler):
        hits = 0

        def do_POST(self):
            type(self).hits += 1
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, Handler, f"http://127.0.0.1:{httpd.server_address[1]}/v1"


@pytest.mark.parametrize("stream", ["false", "true"])
@pytest.mark.parametrize("status, error_type, message", [
    (401, "authentication_error", "invalid x-api-key"),
    (400, "invalid_request_error", "Your credit balance is too low to access the Anthropic API."),
    (404, "not_found_error", "model: fake"),
])
def test_a_wrong_key_with_the_claude_backend_stops_at_once(make, stream, status, error_type,
                                                           message):
    """Keep-alive tells a permanent model error from a passing one by the "HTTP <status>" in
    the error. The Claude backend without streaming returns the provider's message alone
    ("Error: invalid x-api-key"), so a wrong key, no credit or an unknown model look like a
    network failure: the task is continued N times (with the default waits, about 15 minutes)
    and ends with 75. Expected, as with streaming and the other backends: stop at once with 1."""
    s = make(in_order(done("unused")))
    httpd, handler, url = anthropic_server(status, error_type, message)
    try:
        proc = s.run("--yes", "--keep-alive", "2", "Say hi.", LLM_BACKEND="claude",
                     LLM_BASE_URL="", LLM_MODEL="fake", ANTHROPIC_BASE_URL=url,
                     ANTHROPIC_API_KEY="sk-ant-test-0000", LLM_STREAM=stream)
    finally:
        httpd.shutdown()
    shown = s.stdout(proc) + s.stderr(proc)
    assert proc.returncode == 1, shown
    assert handler.hits == 1, f"{handler.hits} requests\n{shown}"
    assert not s.of_type("keepalive/continue"), shown


# ---------- the session lock on a filesystem without file locks ----------

@pytest.mark.skipif(not POSIX, reason="flock")
@pytest.mark.parametrize("code", ["ENOLCK", "EOPNOTSUPP", "ENOSYS"])
def test_a_filesystem_without_file_locks_is_not_called_a_busy_session(tmp_path, monkeypatch, code):
    """flock fails with ENOLCK, EOPNOTSUPP or ENOSYS where the filesystem cannot lock (for
    example NFS when its lock service cannot be reached). The lock treats every error as
    "held by another process", so every one-shot task, terminal session and -c in that home
    is refused with exit 4 and "in use by another mini_agent process". Expected: only "would
    block" means busy; otherwise go on without the lock (with a note), or say that the folder
    has no file locks."""
    import fcntl
    from mini_agent.session_lock import SessionLock
    err = getattr(errno, code)

    def flock(fd, operation):
        if operation & fcntl.LOCK_UN:
            return
        raise OSError(err, os.strerror(err))

    monkeypatch.setattr(fcntl, "flock", flock)
    path = tmp_path / "session.jsonl"
    path.touch()
    try:
        lock = SessionLock(path).acquire()
    except Exception as e:                                  # noqa: BLE001
        assert "in use by another" not in str(e), f"no file locks reported as a busy session: {e}"
    else:
        lock.release()


# ---------- a signal while the supervisor reads the log between two runs ----------

@pytest.mark.skipif(not POSIX, reason="SIGTERM and SIGHUP to oneself")
@pytest.mark.parametrize("sig", ["SIGTERM", "SIGHUP"])
def test_a_signal_while_reading_the_log_between_runs_stops_cleanly(tmp_path, monkeypatch, sig):
    """The supervisor handles SIGTERM and SIGHUP during a run and during a wait. Between the
    two, it reads the session log (which takes longer the longer the task) and writes the
    continue event; a signal there escapes as a traceback and exit 1, without the stop event.
    Expected, as during a wait: exit 128+signal and keepalive/stop in the log."""
    from mini_agent.cli import keepalive
    from mini_agent.cli.main import build_parser, keep_alive_count
    from mini_agent.session import SessionLog
    signum = getattr(signal, sig)
    sessions = tmp_path / "sessions"
    monkeypatch.setattr(keepalive, "SESS_DIR", sessions)
    monkeypatch.setattr(keepalive, "_run", lambda argv, lock: -signal.SIGKILL)  # a run was killed
    real_events = keepalive._events

    def events(path, after=0):
        os.kill(os.getpid(), signum)            # arrives while the log is read
        return real_events(path, after)

    monkeypatch.setattr(keepalive, "_events", events)
    argv = ["--keep-alive", "2", "a task"]
    args = build_parser().parse_args(argv)
    assert keep_alive_count(args, argv) == ""
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        status = keepalive.supervise(args, 1000)
    except BaseException as e:                              # noqa: BLE001
        pytest.fail(f"{sig} escaped the supervisor as {type(e).__name__}")
    finally:
        for s, handler in saved.items():
            signal.signal(s, handler)
    assert status == 128 + signum
    [log] = sessions.glob("*.jsonl")
    assert list(SessionLog(log).events())[-1]["type"] == "keepalive/stop"


# ---------- the count after --keep-alive ----------

@pytest.mark.parametrize("count", ["-1", "\u00b2"])
def test_a_count_that_is_not_a_whole_number_of_1_or_more_is_a_usage_error(make, count):
    """`--keep-alive -1 "task"` runs the task "-1 task" (only all-digit words are taken as the
    count), and `--keep-alive \u00b2 "task"` (a superscript two: str.isdigit() is true, int()
    fails) ends in a ValueError traceback. Expected, as for 0: exit 2, "--keep-alive N must be
    1 or more", and no model call."""
    s = make(in_order(done("unused")))
    proc = s.run("--yes", "--keep-alive", count, "Say hi.")
    shown = s.stdout(proc) + s.stderr(proc)
    assert proc.returncode == 2 and "Traceback" not in shown, shown
    assert s.server.count() == 0, shown
