"""Keep-alive, as a user meets it: `mini_agent --keep-alive` against a fake model server.

Every test starts the real command line in a subprocess, in a fresh home and folder, and looks
only at what a user can see: the exit status, the screen, the session log, the requests the
server received, and the files the tools wrote. The waits between runs are set to 0 seconds,
or to a few seconds where a test acts during a wait.

A failed model call is made by closing the connection without an answer: keep-alive treats it as
temporary, and a run does not retry it, so it costs no time. The 503 test uses real 503s, which a
run retries three times (1 + 2 + 4 s) before it gives up.
"""
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

POSIX = os.name == "posix"
WINDOWS = os.name == "nt"
PASS_THROUGH = ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATHEXT", "COMSPEC")
INTERRUPTED = "interrupted before this call finished"


def _has_keep_alive() -> bool:
    home = tempfile.mkdtemp()
    env = {k: os.environ[k] for k in PASS_THROUGH if k in os.environ}
    env.update(HOME=home, USERPROFILE=home, MINI_AGENT_HOME=os.path.join(home, ".mini_agent"))
    try:
        out = subprocess.run([sys.executable, "-m", "mini_agent", "--help"], env=env, cwd=home,
                             capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return False
    return "--keep-alive" in out.stdout


pytestmark = pytest.mark.skipif(not _has_keep_alive(), reason="this build has no --keep-alive")


# ---------- the model's side ----------

def act(action, **args) -> str:
    return json.dumps({"thought": "next", "action": action, "args": args})


def done(answer) -> str:
    return act("done", answer=answer)


CALC = act("calc", expr="6*7")
DROP = "drop"            # close the connection without an answer


def status(code):
    return ("status", code)


def costly(text, tokens):
    return ("usage", text, tokens)


class Request:
    def __init__(self, body):
        self.body = body
        content = body.get("messages", [{}])[-1].get("content", "")
        self.prompt = content if isinstance(content, str) else json.dumps(content)
        rest = self.prompt.split("[Goal]\n", 1)[1] if "[Goal]\n" in self.prompt else ""
        self.goal = rest.split("\n\n[Tools]", 1)[0].strip()
        self.time = time.time()
        self.tokens = 0          # what the server charged for it (0 when it failed)


class Server:
    """A fake OpenAI-compatible server; answer(request, n) decides the reply to the n-th request.

    A reply is the model's text (HTTP 200), DROP, status(code), costly(text, tokens), or a
    threading.Event: the request is held until the event is set, then dropped.
    """

    def __init__(self, answer):
        # re-entrant: an answer function may call count() while its request is being recorded
        self.answer, self.requests, self.lock = answer, [], threading.RLock()
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                request = Request(body)
                with server.lock:
                    server.requests.append(request)
                    reply = server.answer(request, len(server.requests))
                if isinstance(reply, threading.Event):
                    reply.wait(60)
                    reply = DROP
                if reply == DROP:
                    self.close_connection = True
                    return
                if isinstance(reply, tuple) and reply[0] == "status":
                    self.send(reply[1], "application/json",
                              json.dumps({"error": {"message": f"fake error {reply[1]}"}}).encode())
                    return
                tokens = 15
                if isinstance(reply, tuple) and reply[0] == "usage":
                    reply, tokens = reply[1], reply[2]
                usage = {"prompt_tokens": tokens - 5, "completion_tokens": 5}
                request.tokens = tokens
                if body.get("stream"):
                    chunks = [{"choices": [{"index": 0, "delta": {"content": reply}}]},
                              {"choices": [], "usage": usage}]
                    data = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
                    self.send(200, "text/event-stream", data.encode())
                else:
                    self.send(200, "application/json", json.dumps(
                        {"choices": [{"message": {"role": "assistant", "content": reply},
                                      "finish_reason": "stop"}], "usage": usage}).encode())

            def send(self, code, kind, data):
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def count(self, goal=None) -> int:
        with self.lock:
            return sum(1 for r in self.requests if goal is None or r.goal == goal)

    def close(self):
        self.httpd.shutdown()


def in_order(*replies, then=None):
    """Replies by request number; after the list, `then` (default: done("finished"))."""
    last = done("finished") if then is None else then
    return lambda request, n: replies[n - 1] if n <= len(replies) else last


# ---------- the user's side ----------

class Session:
    """One test's home, folder and fake server, and the mini_agent processes started in it."""

    def __init__(self, tmp_path, answer, waits="0"):
        self.server = Server(answer)
        self.home, self.work = tmp_path / "home", tmp_path / "work"
        self.home.mkdir()
        self.work.mkdir()
        self.logs = tmp_path / "logs"
        self.logs.mkdir()
        self.env = {k: os.environ[k] for k in PASS_THROUGH if k in os.environ}
        self.env.update(HOME=str(self.home), USERPROFILE=str(self.home), PYTHONIOENCODING="utf-8",
                        MINI_AGENT_HOME=str(self.home / ".mini_agent"),
                        LLM_BACKEND="openai_compat", LLM_BASE_URL=self.server.url,
                        LLM_MODEL="fake", MINI_AGENT_KEEPALIVE_WAITS=waits)
        self.procs = []

    def start(self, *args, **env):
        """Start `mini_agent ARGS` (no terminal: stdin is empty) in its own process group."""
        n = len(self.procs)
        out = open(self.logs / f"{n}.out", "w+", encoding="utf-8")
        err = open(self.logs / f"{n}.err", "w+", encoding="utf-8")
        extra = {"start_new_session": True} if POSIX else {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        proc = subprocess.Popen([sys.executable, "-m", "mini_agent", *args], cwd=self.work,
                                env={**self.env, **env}, stdin=subprocess.DEVNULL,
                                stdout=out, stderr=err, **extra)
        proc.out, proc.err = out, err
        self.procs.append(proc)
        return proc

    def run(self, *args, timeout=120, **env):
        proc = self.start(*args, **env)
        proc.wait(timeout)
        return proc

    @staticmethod
    def stdout(proc) -> str:
        proc.out.flush()
        return Path(proc.out.name).read_text(encoding="utf-8")

    @staticmethod
    def stderr(proc) -> str:
        proc.err.flush()
        return Path(proc.err.name).read_text(encoding="utf-8")

    def session_files(self):
        return sorted((self.home / ".mini_agent" / "sessions").glob("*.jsonl"))

    def events(self, path=None):
        found = []
        for p in ([path] if path else self.session_files()):
            for line in Path(p).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        found.append(json.loads(line))
                    except ValueError:            # a line being written
                        pass
        return found

    def of_type(self, kind, path=None):
        return [e["data"] for e in self.events(path) if e["type"] == kind]

    def pinned(self) -> Path:
        """The session file the supervisor continues (the one with keep-alive events)."""
        for p in self.session_files():
            if any(e["type"].startswith("keepalive/") for e in self.events(p)):
                return p
        raise AssertionError("no session file has keep-alive events")

    def close(self):
        for proc in self.procs:
            if proc.poll() is None:
                if POSIX:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except OSError:
                        pass
                else:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                   capture_output=True)
                proc.wait(30)
            proc.out.close()
            proc.err.close()
        self.server.close()


@pytest.fixture
def make(tmp_path):
    made = []

    def factory(answer, waits="0"):
        made.append(Session(tmp_path, answer, waits))
        return made[-1]

    yield factory
    for s in made:
        s.close()


def wait_until(condition, timeout=60, what="the condition"):
    end = time.time() + timeout
    while time.time() < end:
        value = condition()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def alive(pid) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:                                          # a zombie is not alive
        return "Z" not in subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                                         capture_output=True, text=True).stdout
    except OSError:
        return True


def children(pid):
    out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def hard_kill(pid):
    """kill -9 (taskkill /F on Windows): no handler runs, nothing is cleaned up."""
    if WINDOWS:
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
    else:
        os.kill(pid, signal.SIGKILL)


posix_only = pytest.mark.skipif(not POSIX, reason="signals and process groups as on POSIX")


# ---------- 1. temporary model errors ----------

def test_503_more_than_three_times_then_200_continues_and_finishes(make):
    s = make(in_order(*[status(503)] * 4, CALC, done("forty-two")))
    proc = s.run("--keep-alive", "Compute 6*7 with calc.")
    assert proc.returncode == 0, s.stderr(proc)
    assert "forty-two" in s.stdout(proc)
    assert s.server.count() == 6                              # 4 x 503 (one call: 1 + 3 retries), then 2
    continues = s.of_type("keepalive/continue")
    assert len(continues) == 1 and "503" in str(continues[0].get("why"))
    assert "keep-alive: " in s.stderr(proc)


@pytest.mark.parametrize("code", [401, 404])
def test_401_and_404_stop_at_once_without_waiting(make, code):
    s = make(lambda request, n: status(code), waits="30")
    started = time.time()
    proc = s.run("--keep-alive", "Compute 6*7.")
    assert proc.returncode == 1, s.stderr(proc)
    assert time.time() - started < 20                         # no 30-second keep-alive wait
    assert s.server.count() == 1                              # not retried, not continued
    assert s.of_type("keepalive/continue") == []


# ---------- 3. a child killed in the middle of a step (taskkill /F on Windows) ----------

def test_child_killed_mid_step_continues_without_running_the_call_twice(make, tmp_path):
    marker = tmp_path / "runs.txt"
    if WINDOWS:
        kill = ("import subprocess; subprocess.run(['taskkill', '/F', '/PID', str(os.getppid())], "
                "capture_output=True)")
    else:
        kill = "import signal; os.kill(os.getppid(), signal.SIGKILL)"
    code = f"import os\nopen({str(marker)!r}, 'a').write('ran\\n')\n{kill}\n"
    s = make(in_order(act("run_python", code=code), then=done("recovered")))
    proc = s.run("--yes", "--keep-alive", "Run the code once.")
    assert proc.returncode == 0, s.stderr(proc)
    assert "recovered" in s.stdout(proc)
    assert marker.read_text(encoding="utf-8") == "ran\n"      # the cut-off call did not run again
    assert INTERRUPTED in s.server.requests[1].prompt         # the model is told it was cut off
    assert len(s.of_type("keepalive/continue")) == 1


# ---------- 4. Ctrl+C and SIGTERM ----------

@posix_only
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_a_signal_during_a_wait_stops_at_once_and_never_continues(make, sig):
    s = make(lambda request, n: DROP, waits="30")
    proc = s.start("--keep-alive", "Compute 6*7.")
    wait_until(lambda: s.of_type("keepalive/continue"), what="the first wait")
    asked = s.server.count()
    os.kill(proc.pid, sig)
    sent = time.time()
    proc.wait(15)
    assert proc.returncode == 128 + sig, s.stderr(proc)
    assert time.time() - sent < 10                             # not after the 30-second wait
    time.sleep(1)
    assert s.server.count() == asked                          # never continued
    assert s.of_type("keepalive/stop")


@posix_only
def test_ctrl_c_during_a_run_stops_and_never_continues(make):
    hold = threading.Event()
    s = make(in_order(hold, then=done("too late")))
    proc = s.start("--keep-alive", "Compute 6*7.")
    wait_until(lambda: s.server.count() == 1, what="the first request")
    os.killpg(proc.pid, signal.SIGINT)                        # Ctrl+C reaches the whole group
    proc.wait(30)
    hold.set()
    assert proc.returncode == 130, s.stderr(proc)
    time.sleep(1.5)
    assert s.server.count() == 1
    assert s.of_type("keepalive/continue") == []


@posix_only
def test_sigterm_to_the_supervisor_during_a_run_stops_the_run_too(make):
    hold = threading.Event()
    s = make(in_order(hold, then=done("too late")))
    proc = s.start("--keep-alive", "Compute 6*7.")
    wait_until(lambda: s.server.count() == 1, what="the first request")
    child = wait_until(lambda: children(proc.pid), what="the run's process")[0]
    os.kill(proc.pid, signal.SIGTERM)
    proc.wait(30)
    assert proc.returncode == 143, s.stderr(proc)
    wait_until(lambda: not alive(child), timeout=10, what="the run to stop with its supervisor")
    hold.set()
    time.sleep(1.5)
    assert s.server.count() == 1
    assert s.of_type("keepalive/continue") == []


# ---------- 5. one process per session ----------

def test_a_second_process_on_the_same_session_is_refused(make):
    s = make(lambda request, n: DROP, waits="30")
    first = s.start("--keep-alive", "Compute 6*7.")
    wait_until(lambda: s.of_type("keepalive/continue"), what="the first wait")
    for args in (["-c", "--keep-alive"], ["-c"]):
        second = s.run(*args, timeout=60)
        shown = s.stdout(second) + s.stderr(second)
        assert second.returncode == 4, shown
        assert "in use by another mini_agent process" in shown and str(first.pid) in shown
    assert first.poll() is None                              # the first one carries on


def test_a_hard_kill_leaves_no_lock_in_the_way(make):
    mode = {"answer": DROP}
    s = make(lambda request, n: mode["answer"], waits="30")
    first = s.start("--keep-alive", "Compute 6*7.")
    wait_until(lambda: s.of_type("keepalive/continue"), what="the first wait")
    pinned = s.pinned()
    hard_kill(first.pid)
    first.wait(30)
    mode["answer"] = done("continued")
    again = s.run("-c", "--keep-alive")
    assert again.returncode == 0, s.stdout(again) + s.stderr(again)
    assert "continued" in s.stdout(again)
    assert not Path(str(pinned) + ".lock").exists()          # released cleanly this time


# ---------- 6. a new task in the same folder during a wait ----------

ALPHA = "Alpha task: answer with the word อัลฟา."
BETA = "Beta task: answer with the word בטא."


def test_a_new_task_during_a_wait_does_not_replace_the_task_continued(make):
    def answer(request, n):
        if request.goal == BETA:
            return done("beta answer")
        return DROP if s.server.count(ALPHA) == 1 else done("alpha answer")

    s = make(answer, waits="4")
    first = s.start("--keep-alive", ALPHA)
    wait_until(lambda: s.of_type("keepalive/continue"), what="the first wait")
    other = s.run(BETA)
    assert other.returncode == 0 and "beta answer" in s.stdout(other)
    first.wait(60)
    assert first.returncode == 0, s.stderr(first)
    assert "alpha answer" in s.stdout(first) and "beta answer" not in s.stdout(first)
    assert s.server.count(ALPHA) == 2 and s.server.count(BETA) == 1
    ends = s.of_type("turn/end", s.pinned())
    assert ends and ends[-1].get("answer") == "alpha answer"


# ---------- 7. the token budget ----------

def test_tokens_across_continues_stay_within_the_original_budget(make):
    call = 30_000
    s = make(in_order(costly(CALC, call), DROP, costly(CALC, call), DROP,
                      then=costly(CALC, call)))
    proc = s.run("--budget", "100000", "--keep-alive", "Keep computing.")
    assert proc.returncode == 1, s.stderr(proc)                # token_limit: permanent
    # run 1: 30k, failed; run 2 (70k left): 30k, failed; run 3 (40k left): 30k, 30k -> over
    assert s.server.count() == 6
    served = sum(r.tokens for r in s.server.requests)
    assert served == 120_000 and served <= 100_000 + call      # at most one call past the budget
    budgets = [c.get("budget_left") for c in s.of_type("keepalive/continue")]
    assert budgets == [70_000, 40_000]
    assert [e.get("reason") for e in s.of_type("turn/end")][-1] == "token_limit"


# ---------- 8. approval ----------

def test_needs_approval_without_a_terminal_exits_3_without_continuing(make):
    s = make(in_order(act("bash", cmd="echo hi")))
    proc = s.run("--keep-alive", "Say hi with bash.")
    assert proc.returncode == 3, s.stderr(proc)
    assert s.server.count() == 1
    assert s.of_type("keepalive/continue") == []


# ---------- 9. progress, giving up, and the limits ----------

def test_a_continue_that_makes_progress_resets_the_count(make):
    s = make(in_order(DROP, CALC, DROP, CALC, DROP, CALC, then=done("made it")))
    proc = s.run("--keep-alive", "1", "Compute 6*7 three times.")
    assert proc.returncode == 0, s.stderr(proc)
    assert "made it" in s.stdout(proc)
    assert len(s.of_type("keepalive/continue")) == 3            # more than 1: progress reset it


def test_continues_without_progress_give_up_with_75(make):
    s = make(lambda request, n: DROP)
    proc = s.run("--keep-alive", "2", "Compute 6*7.")
    assert proc.returncode == 75, s.stderr(proc)
    assert s.server.count() == 3                               # the first run and 2 continues
    assert len(s.of_type("keepalive/continue")) == 2
    stops = s.of_type("keepalive/stop")
    assert stops and stops[-1].get("exit_status") == 75
    shown = s.stderr(proc)
    assert "2 continues in a row made no progress" in shown
    assert "mini_agent -c --keep-alive" in shown


def test_the_step_limit_still_ends_the_task(make):
    s = make(lambda request, n: CALC)                          # never done
    proc = s.run("--keep-alive", "Compute 6*7 forever.")
    assert proc.returncode == 1, s.stderr(proc)
    assert s.of_type("keepalive/continue") == []
    assert [e.get("reason") for e in s.of_type("turn/end")] == ["max_steps"]


# ---------- the command line ----------

@pytest.mark.parametrize("args, waits", [
    (["--keep-alive", "0", "Compute 6*7."], "0"),
    (["--keep-alive=0", "Compute 6*7."], "0"),
    (["--tui", "--keep-alive", "Compute 6*7."], "0"),
    (["--keep-alive"], "0"),
    (["--keep-alive", "Compute 6*7."], "soon"),
    (["--keep-alive", "Compute 6*7."], "-1"),
])
def test_usage_errors_exit_2_and_ask_nothing(make, args, waits):
    s = make(lambda request, n: done("no"), waits=waits)
    proc = s.run(*args)
    assert proc.returncode == 2, s.stdout(proc) + s.stderr(proc)
    assert s.server.count() == 0


@pytest.mark.parametrize("args, limit", [
    (["--keep-alive", "Compute 6*7."], 5),
    (["--keep-alive", "3", "Compute 6*7."], 3),
    (["--keep-alive=2", "Compute 6*7."], 2),
])
def test_the_number_after_keep_alive_and_the_task(make, args, limit):
    s = make(in_order(DROP, then=done("ok")))
    proc = s.run(*args)
    assert proc.returncode == 0, s.stderr(proc)
    continues = s.of_type("keepalive/continue")
    assert len(continues) == 1 and continues[0].get("of") == limit
    assert s.server.requests[0].goal == "Compute 6*7."


def test_continue_with_nothing_unfinished_says_so_and_exits_0(make):
    s = make(lambda request, n: done("no"))
    proc = s.run("-c", "--keep-alive")
    assert proc.returncode == 0, s.stderr(proc)
    assert "nothing unfinished to continue in this folder" in s.stdout(proc)
    assert s.server.count() == 0
