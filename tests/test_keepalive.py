"""--keep-alive: a task survives passing failures (the provider down, the process killed) and
stops at once on anything waiting will not change. Each test runs the real command line
against a fake provider; the waits between runs are set to 0 (MINI_AGENT_KEEPALIVE_WAITS)."""
import http.server
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from mini_agent.cli import keepalive
from mini_agent.cli.main import build_parser, keep_alive_count
from mini_agent.session import SessionLog
from mini_agent.session_lock import SessionBusy, SessionLock
from test_cli import cli, env_for

POSIX = pytest.mark.skipif(os.name == "nt", reason="signals and $PPID: POSIX only")
DONE = json.dumps({"thought": "done", "action": "done", "args": {"answer": "finished"}})


def bash(cmd):
    return json.dumps({"thought": "run", "action": "bash", "args": {"cmd": cmd}})


class Flaky:
    """A fake OpenAI-compatible provider: each request takes the next item of plan, an int
    (that HTTP status, with an error body) or a reply text; when plan is empty, DONE."""

    def __init__(self, plan):
        self.plan, self.requests = list(plan), 0
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, status, data, kind):
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests += 1
                item = server.plan.pop(0) if server.plan else DONE
                if isinstance(item, int):
                    data = json.dumps({"error": {"message": f"status {item}"}}).encode()
                    return self.send(item, data, "application/json")
                usage = {"prompt_tokens": 10, "completion_tokens": 5}
                if body.get("stream"):
                    chunks = [{"choices": [{"index": 0, "delta": {"content": item}}]},
                              {"choices": [], "usage": usage}]
                    data = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
                    return self.send(200, data.encode(), "text/event-stream")
                data = json.dumps({"choices": [{"message": {"role": "assistant", "content": item},
                                                "finish_reason": "stop"}], "usage": usage})
                self.send(200, data.encode(), "application/json")

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()


def env_of(server, waits="0", **extra):
    env = env_for(server, MINI_AGENT_KEEPALIVE_WAITS=waits, **extra)
    env["LLM_BASE_URL"] = server.url
    return env


def sessions(env):
    return sorted((Path(env["MINI_AGENT_HOME"]) / "sessions").glob("*.jsonl"))


def events(path, type=None):
    return [e for e in SessionLog(path).events() if type is None or e["type"] == type]


def start(args, env, cwd, **kw):
    """The supervisor as a background process (SIGINT back to its default: a test run started
    as a background job would make every child ignore Ctrl+C)."""
    extra = {} if os.name == "nt" else {"preexec_fn": lambda: signal.signal(signal.SIGINT, signal.SIG_DFL)}
    return subprocess.Popen([sys.executable, "-m", "mini_agent", *args], env=env, cwd=cwd,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", **extra, **kw)


def wait_for(check, seconds=60.0):
    end = time.time() + seconds
    while time.time() < end:
        if check():
            return True
        time.sleep(0.05)
    return False


# ---------- the option ----------

@pytest.mark.parametrize("argv,count,task", [
    (["--keep-alive", "Compute", "3**7"], 5, ["Compute", "3**7"]),
    (["--keep-alive", "Compute 3**7"], 5, ["Compute 3**7"]),
    (["--keep-alive", "3", "Compute"], 3, ["Compute"]),
    (["--keep-alive=2", "Compute"], 2, ["Compute"]),
    (["--keep-alive", "--yes", "Compute"], 5, ["Compute"]),
    (["-c", "--keep-alive"], 5, []),
])
def test_the_count_is_optional(argv, count, task):
    args = build_parser().parse_args(argv)
    if args.task and args.task[0] == "--":
        args.task = args.task[1:]
    assert keep_alive_count(args, argv) == ""
    assert args.keep_alive == count and args.task == task


@pytest.mark.parametrize("argv,message", [
    (["--keep-alive", "0", "task"], "1 or more"),
    (["--keep-alive=x", "task"], "whole number"),
    (["--keep-alive"], "needs a task"),
    (["--tui", "-c", "--keep-alive"], "--tui"),
])
def test_usage_errors(argv, message):
    r = cli(*argv, env=env_for(LLM_BACKEND="offline"))
    assert r.returncode == 2 and message in r.stderr, r.stderr


def test_bad_waits_are_a_usage_error():
    r = cli("--keep-alive", "task", env=env_for(LLM_BACKEND="offline", MINI_AGENT_KEEPALIVE_WAITS="soon"))
    assert r.returncode == 2 and "MINI_AGENT_KEEPALIVE_WAITS" in r.stderr


# ---------- what is worth another run ----------

@pytest.mark.parametrize("error,again", [
    ("Error: HTTP 503: overloaded", True), ("Error: HTTP 500: x", True), ("Error: HTTP 429: slow down", True),
    ("Error: HTTP 408: timeout", True), ("Error: HTTP 200: provider returned error", True),
    ("Error: ConnectError: [Errno 111] Connection refused", True), ("Error: ReadTimeout: timed out", True),
    ("Error: unexpected reply: {}", True),
    ("Error: HTTP 400: context too long", False), ("Error: HTTP 401: bad key", False),
    ("Error: HTTP 402: no credit", False), ("Error: HTTP 403: forbidden", False),
    ("Error: HTTP 404: no such model", False), ("Error: HTTP 413: too large", False),
    ("Error: HTTP 422: bad args", False), ("Error: no model: set LLM_MODEL (or model=...)", False),
    ("Error: no server address: set LLM_BASE_URL", False),
])
def test_which_model_errors_are_passing(error, again):
    assert keepalive.passing(error) is again
    kind, status, _ = keepalive.outcome(1, {"reason": "llm_error", "answer": error})
    assert (kind, status) == (("again", 1) if again else ("stop", 1))


@pytest.mark.parametrize("rc,ended,expected", [
    (0, {"reason": "done"}, ("done", 0)),
    (-9, None, ("again", 1)),                        # SIGKILL: out of memory, kill -9
    (-11, None, ("again", 1)),                       # a crash
    (1, None, ("again", 1)),                         # no turn/end (Windows taskkill /F is exit 1)
    (130, None, ("stop", 130)), (0xC000013A, None, ("stop", 130)),
    (-15, None, ("stop", 143)), (-1, None, ("stop", 129)), (-2, None, ("stop", 130)),
    (2, None, ("stop", 2)), (4, None, ("stop", 4)),
    (3, {"reason": "needs_approval"}, ("stop", 3)),
    (1, {"reason": "token_limit"}, ("stop", 1)), (1, {"reason": "max_steps"}, ("stop", 1)),
    (1, {"reason": "context_full"}, ("stop", 1)), (1, {"reason": "no_recording"}, ("stop", 1)),
    (1, {"reason": "invalid_reply"}, ("stop", 1)),
])
def test_how_a_run_ended(rc, ended, expected):
    if os.name == "nt" and rc in (-15, -1, -2):
        pytest.skip("POSIX signals")
    assert keepalive.outcome(rc, ended)[:2] == expected


def test_tokens_a_run_used():
    evs = [{"type": "step/think", "data": {"tokens_in": 10, "tokens_out": 5}},
           {"type": "step/think", "data": {"tokens_in": 30, "tokens_out": 12}},
           {"type": "turn/end", "data": {"reason": "token_limit", "detail": "1,050/1,000"}}]
    assert keepalive.spent(evs[:2]) == 42 and keepalive.spent(evs) == 1050
    assert keepalive.spent([{"type": "step/think", "data": {"tokens_in": "x"}}]) == 0


def test_keepalive_events_keep_a_pending_task_pending(tmp_path):
    log = SessionLog(tmp_path / "s.jsonl")
    log.append("user/message", content="g")
    log.append("keepalive/continue", attempt=1, of=5, why="w", wait_s=0, budget_left=9, exit_status=-9)
    info = log.unfinished_turn()
    assert info["pending"] and info["goal"] == "g"


# ---------- one process per session ----------

def test_a_session_has_one_holder(tmp_path):
    session = tmp_path / "s.jsonl"
    session.touch()
    lock = SessionLock(session).acquire()
    code = ("import sys; sys.path.insert(0, %r)\n"
            "from mini_agent.session_lock import SessionLock, SessionBusy\n"
            "try:\n    SessionLock(%r).acquire(); print('got it')\n"
            "except SessionBusy as e:\n    print('busy', e)\n") % (str(Path(__file__).parents[1]), str(session))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout
    assert out.startswith("busy") and f"process {os.getpid()}" in out
    lock.release()
    assert not (tmp_path / "s.jsonl.lock").exists()
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout
    assert out.startswith("got it")


@POSIX
def test_a_killed_holder_leaves_no_lock(tmp_path):
    session = tmp_path / "s.jsonl"
    session.touch()
    code = ("import sys, time; sys.path.insert(0, %r)\n"
            "from mini_agent.session_lock import SessionLock\n"
            "SessionLock(%r).acquire(); print('held', flush=True); time.sleep(60)\n"
            ) % (str(Path(__file__).parents[1]), str(session))
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "held"
    with pytest.raises(SessionBusy):
        SessionLock(session).acquire()
    proc.kill()
    proc.wait()
    assert (tmp_path / "s.jsonl.lock").exists()            # the file stays, the lock does not
    SessionLock(session).acquire().release()


# ---------- the command line ----------

def test_503_more_than_the_retries_then_it_works():
    with Flaky([503] * 4 + [503] * 4 + [DONE]) as server:  # two runs fail (1 + 3 retries each)
        env = env_of(server)
        r = cli("--keep-alive", "the task", env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stderr.count("keep-alive: the model call failed: HTTP 503") == 2
    assert "continuing in 0 s (try 1 of 5)" in r.stderr and "(try 2 of 5)" in r.stderr
    [path] = sessions(env)
    assert [e["data"]["attempt"] for e in events(path, "keepalive/continue")] == [1, 2]
    assert events(path, "turn/end")[-1]["data"]["reason"] == "done"


@pytest.mark.parametrize("status", [401, 404])
def test_a_wrong_key_or_unknown_model_stops_at_once(status):
    with Flaky([status]) as server:
        env = env_of(server)
        r = cli("--keep-alive", "the task", env=env)
        assert server.requests == 1                          # no retry, no second run
    assert r.returncode == 1 and "keep-alive:" not in r.stderr, r.stdout + r.stderr
    assert not events(sessions(env)[0], "keepalive/continue")


def test_needs_approval_without_a_terminal_stops_at_once():
    with Flaky([bash("echo hi")]) as server:
        env = env_of(server)
        r = cli("--keep-alive", "the task", env=env)
        assert server.requests == 1
    assert r.returncode == 3 and "keep-alive:" not in r.stderr and "mini_agent -c --yes" in r.stdout


def test_gives_up_after_n_runs_without_progress():
    with Flaky([408] * 10) as server:                       # 408: not retried within a run
        env = env_of(server)
        r = cli("--keep-alive", "2", "the task", env=env)
        assert server.requests == 3                          # the first run and 2 continues
    assert r.returncode == 75, r.stdout + r.stderr
    assert "2 continues in a row made no progress" in r.stderr
    path = sessions(env)[0]
    assert [e["data"]["exit_status"] for e in events(path, "keepalive/stop")] == [75]


def test_a_run_that_completes_a_step_starts_the_count_again():
    plan = [408, bash("echo 1"), 408, bash("echo 2"), 408, DONE]
    with Flaky(plan) as server:
        env = env_of(server)
        r = cli("--yes", "--keep-alive", "1", "the task", env=env)
    assert r.returncode == 0, r.stdout + r.stderr          # with N = 1 it survives three failures
    assert [e["data"]["attempt"] for e in events(sessions(env)[0], "keepalive/continue")] == [1, 1, 1]


def test_continue_with_nothing_unfinished_exits_0():
    r = cli("-c", "--keep-alive", env=env_for(LLM_BACKEND="offline"))
    assert r.returncode == 0 and "nothing unfinished to continue in this folder" in r.stdout
    assert r.stderr == ""


@POSIX
def test_killed_mid_step_then_continued_and_finished():
    work = tempfile.mkdtemp()
    plan = [bash("echo one >> count.txt"), bash("echo $PPID > pid.txt; echo two >> count.txt; sleep 30"), DONE]
    with Flaky(plan) as server:
        env = env_of(server)
        proc = start(["--yes", "--keep-alive", "count"], env, work)
        pid = Path(work) / "pid.txt"
        assert wait_for(lambda: pid.exists() and pid.read_text().strip())
        os.kill(int(pid.read_text()), signal.SIGKILL)          # the child, as an out-of-memory kill
        out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0, out + err
    assert "the run was ended by SIGKILL; continuing" in err
    assert (Path(work) / "count.txt").read_text().split() == ["one", "two"]   # not run twice
    path = sessions(env)[0]
    results = [e["data"]["content"] for e in events(path, "tool/result")]
    assert len(results) == 1 and events(path, "turn/end")[-1]["data"]["reason"] == "done"


@POSIX
def test_the_runs_share_the_budget():
    work = tempfile.mkdtemp()
    plan = [bash("echo $PPID > pid.txt; sleep 30")] + [bash("echo x")] * 20
    with Flaky(plan) as server:
        env = env_of(server)
        proc = start(["--yes", "--budget", "100", "--keep-alive", "count"], env, work)
        pid = Path(work) / "pid.txt"
        assert wait_for(lambda: pid.exists() and pid.read_text().strip())
        os.kill(int(pid.read_text()), signal.SIGKILL)
        out, err = proc.communicate(timeout=60)
    assert proc.returncode == 1, out + err                     # the shared budget ran out
    path = sessions(env)[0]
    [cont] = events(path, "keepalive/continue")
    assert cont["data"]["budget_left"] == 85                   # 100 minus the first run's 15
    end = events(path, "turn/end")[-1]["data"]
    assert end["reason"] == "token_limit" and end["detail"].endswith("/85")
    used = 15 + int(end["detail"].split("/")[0].replace(",", ""))
    assert used <= 100 + 15                                    # at most one call past the budget


@POSIX
def test_ctrl_c_during_a_wait_stops_at_once():
    with Flaky([408] * 5) as server:
        env = env_of(server, waits="60")
        proc = start(["--keep-alive", "the task"], env, tempfile.mkdtemp())
        assert wait_for(lambda: sessions(env) and events(sessions(env)[0], "keepalive/continue"))
        t0 = time.time()
        proc.send_signal(signal.SIGINT)
        out, err = proc.communicate(timeout=20)
    assert proc.returncode == 130 and time.time() - t0 < 10, out + err
    assert "Traceback" not in err
    assert events(sessions(env)[0], "keepalive/stop")[-1]["data"]["exit_status"] == 130


@POSIX
def test_sigterm_to_the_supervisor_stops_the_running_child():
    work = tempfile.mkdtemp()
    with Flaky([bash("echo $PPID > pid.txt; sleep 30")]) as server:
        env = env_of(server)
        proc = start(["--yes", "--keep-alive", "the task"], env, work)
        pid = Path(work) / "pid.txt"
        assert wait_for(lambda: pid.exists() and pid.read_text().strip())
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=30)
        child = int(pid.read_text())
    assert proc.returncode == 143, out + err
    assert not Path(f"/proc/{child}").exists() or "zombie" in Path(f"/proc/{child}/status").read_text().lower()


@POSIX
def test_a_second_process_on_the_session_is_refused_and_a_new_task_does_not_steal_it():
    work = tempfile.mkdtemp()
    with Flaky([408, 408]) as server:
        env = env_of(server, waits="6,1")
        proc = start(["--keep-alive", "first task"], env, work)
        assert wait_for(lambda: sessions(env) and events(sessions(env)[0], "keepalive/continue"))
        first = sessions(env)[0]
        r = cli("-c", env=env, cwd=work)                    # the same folder, while it waits
        assert r.returncode == 4 and "in use by another mini_agent process" in r.stderr
        server.plan[:0] = [DONE]                            # the next request answers the new task
        r = cli("second task", env=env, cwd=work)           # a new task in the folder meanwhile
        assert r.returncode == 0
        out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0, out + err
    goals = {p.name: [e["data"]["goal"] for e in events(p, "turn/start")] for p in sessions(env)}
    assert goals[first.name] == ["first task"] * len(goals[first.name])   # it kept its own session
    assert any(g == ["second task"] for g in goals.values())
