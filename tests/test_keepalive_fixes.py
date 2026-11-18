"""Keep-alive: a log line cut short by a kill, providers that drop the HTTP status, a stop while
a run starts, a stop while a tool runs, and the runs' way into a held session."""
import contextlib
import inspect
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from mini_agent.cli import keepalive as K
from mini_agent.cli.main import build_parser
from mini_agent.core.agent.run import run_agent
from mini_agent.session import SessionLog
from mini_agent.session_lock import TOKEN_ENV, SessionLock
from test_cli import cli, env_for
from test_keepalive import Flaky, env_of, events, sessions, start, wait_for

POSIX = pytest.mark.skipif(os.name == "nt", reason="POSIX signals")


def lines_are_events(path: Path) -> bool:
    raw = path.read_bytes().decode("utf-8")
    return raw.endswith("\n") and all(json.loads(line) for line in raw.splitlines())


def supervise(tmp_path, runner):
    """K.supervise for one task, with runner(argv, lock) standing in for a run's process."""
    args = build_parser().parse_args(["--keep-alive=1", "a task"])
    args.keep_alive = int(args.keep_alive)
    with patch.object(K, "SESS_DIR", tmp_path), patch.object(K, "_run", runner), \
         patch.object(K, "waits", return_value=[0]), contextlib.redirect_stderr(io.StringIO()):
        return K.supervise(args, 100)


@pytest.fixture
def stop_handlers():
    saved = {s: signal.getsignal(s) for s in K.ASKED_TO_STOP}
    yield
    for s, handler in saved.items():
        signal.signal(s, handler)


# ---------- 1. a kill during a write leaves a last line without its newline ----------

def test_a_cut_last_line_is_skipped_and_the_next_writer_removes_it(tmp_path):
    path = tmp_path / "s.jsonl"
    log = SessionLog(path)
    log.append("user/message", content="a task")
    log.append("turn/start", goal="a task")
    with path.open("a", encoding="utf-8") as f:
        f.write('{"seq": 3, "time": 1.0, "type": "assistant/chunk", "data": {"text": "ab')
    assert [e["seq"] for e in SessionLog(path).events()] == [1, 2]
    again = SessionLog(path)
    assert again.append("turn/end", reason="done", answer="x")["seq"] == 3
    assert lines_are_events(path)
    assert [e["type"] for e in SessionLog(path).events()] == ["user/message", "turn/start", "turn/end"]


def test_a_whole_last_event_without_its_newline_is_kept(tmp_path):
    path = tmp_path / "s.jsonl"
    SessionLog(path).append("user/message", content="a task")
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    assert [e["seq"] for e in SessionLog(path).events()] == [1]
    SessionLog(path).append("turn/start", goal="a task")
    assert lines_are_events(path)
    assert [e["seq"] for e in SessionLog(path).events()] == [1, 2]


def test_a_cut_multibyte_character_is_skipped_too(tmp_path):
    path = tmp_path / "s.jsonl"
    SessionLog(path).append("user/message", content="a task")
    with path.open("ab") as f:
        f.write('{"seq": 2, "data": {"text": "Ωμέγα'.encode("utf-8")[:-1])
    assert [e["seq"] for e in SessionLog(path).events()] == [1]
    SessionLog(path).append("turn/end", reason="done", answer="x")
    assert lines_are_events(path)


def test_the_supervisor_continues_a_run_killed_in_the_middle_of_a_write(tmp_path):
    runs = []

    def run(argv, lock):
        runs.append(argv)
        log = SessionLog(lock.session)
        if len(runs) == 1:
            log.append("user/message", content="a task")
            log.append("turn/start", goal="a task")
            with lock.session.open("a", encoding="utf-8") as f:
                f.write('{"seq": 3, "type": "assistant/chunk", "data": {"text": "x')
            return -signal.SIGKILL if hasattr(signal, "SIGKILL") else 1
        log.append("turn/end", reason="done", answer="finished")
        return 0

    assert supervise(tmp_path, run) == 0
    assert len(runs) == 2
    assert all(lines_are_events(p) for p in tmp_path.glob("*.jsonl"))


# ---------- 2. every backend keeps the HTTP status of a failed call ----------

def test_ollama_keeps_the_http_status():
    from mini_agent.llm.backends.ollama import OllamaLLM
    reply = httpx.Response(404, json={"error": "model 'missing' not found"})
    with patch("mini_agent.llm.backends.ollama.httpx.post", return_value=reply):
        text = OllamaLLM(model="missing").call("hi")
    assert text.startswith("Error: HTTP 404: ") and "not found" in text
    assert not K.passing(text)


@pytest.mark.parametrize("status,again", [(401, False), (404, False), (529, True)])
def test_claude_without_streaming_keeps_the_http_status(status, again):
    from mini_agent.llm.backends.claude import ClaudeLLM
    body = {"type": "error", "error": {"type": "some_error", "message": "it failed"}}
    reply = httpx.Response(status, json=body)
    llm = ClaudeLLM(api_key="test-key", model="claude-test", stream=False)
    llm.retry_waits = ()
    with patch("mini_agent.llm.backends.claude.httpx.post", return_value=reply):
        text = llm.call("hi")
    assert text == f"Error: HTTP {status}: it failed"
    assert K.passing(text) is again


def test_an_unknown_ollama_model_stops_keep_alive_at_once(tmp_path):
    from mini_agent.llm.backends.ollama import OllamaLLM
    runs = []

    def run(argv, lock):
        runs.append(argv)
        log = SessionLog(lock.session)
        if len(runs) == 1:
            log.append("user/message", content="a task")
        reply = httpx.Response(404, json={"error": "model 'missing' not found"})
        with patch("mini_agent.llm.backends.ollama.httpx.post", return_value=reply):
            run_agent(llm=OllamaLLM(model="missing"), tools={}, tool_desc="", goal="a task",
                      verbose=False, on_event=lambda kind, /, **data: log.append(kind, **data))
        return 1

    assert supervise(tmp_path, run) == 1
    assert len(runs) == 1


# ---------- 3. a stop while a run is being started stops that run too ----------

def _sleeper():
    return [sys.executable, "-c", "import time; time.sleep(30)"]


@POSIX
def test_a_stop_inside_popen_stops_the_new_run(tmp_path, stop_handlers):
    lock = SessionLock(tmp_path / "s.jsonl").acquire()
    made = []
    real = subprocess.Popen

    def popen(*a, **kw):
        made.append(real(*a, **kw))
        signal.raise_signal(signal.SIGTERM)        # arrives before Popen has returned
        return made[-1]

    signal.signal(signal.SIGTERM, K._stopped)
    try:
        with patch.object(K.subprocess, "Popen", popen), pytest.raises(K.Stopped):
            K._run(_sleeper(), lock)
        assert made[0].poll() is not None
    finally:
        for p in made:
            if p.poll() is None:
                p.kill()
            p.wait()
        lock.release()


@POSIX
def test_a_stop_right_after_popen_stops_the_new_run(tmp_path, stop_handlers):
    lock = SessionLock(tmp_path / "s.jsonl").acquire()
    lines, first = inspect.getsourcelines(K._run)
    after = first + next(i for i, s in enumerate(lines) if "if _held:" in s)
    made = []
    real = subprocess.Popen

    def popen(*a, **kw):
        made.append(real(*a, **kw))
        return made[-1]

    def trace(frame, event, arg):
        if frame.f_code is K._run.__code__ and event == "line" and frame.f_lineno == after:
            signal.raise_signal(signal.SIGTERM)
        return trace

    signal.signal(signal.SIGTERM, K._stopped)
    try:
        with patch.object(K.subprocess, "Popen", popen), pytest.raises(K.Stopped):
            sys.settrace(trace)
            try:
                K._run(_sleeper(), lock)
            finally:
                sys.settrace(None)
        assert made and made[0].poll() is not None
    finally:
        for p in made:
            if p.poll() is None:
                p.kill()
            p.wait()
        lock.release()


# ---------- 4. a stop while a tool runs stops the tool too ----------

def _slow_tool(folder: Path) -> str:
    code = (f"from pathlib import Path\nimport time\np = Path({str(folder)!r})\n"
            "(p / 'ready').touch()\n"
            "while not (p / 'release').exists(): time.sleep(0.02)\n"
            "(p / 'side_effect').touch()\n")
    return json.dumps({"thought": "run it", "action": "run_python", "args": {"code": code}})


@POSIX
@pytest.mark.parametrize("keep_alive", [False, True])
def test_sigterm_stops_the_running_tool(tmp_path, keep_alive):
    flag = tmp_path / "flags"
    flag.mkdir()
    with Flaky([_slow_tool(flag)]) as server:
        env = env_of(server)
        args = (["--keep-alive"] if keep_alive else []) + ["--yes", "run the slow tool"]
        proc = start(args, env, tmp_path)
        try:
            assert wait_for(lambda: (flag / "ready").exists(), 60), proc.communicate(timeout=5)
            proc.send_signal(signal.SIGTERM)
            out, err = proc.communicate(timeout=60)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        (flag / "release").touch()
        time.sleep(1.5)
        assert proc.returncode == 128 + signal.SIGTERM, (out, err)
        assert not (flag / "side_effect").exists()
        (log,) = sessions(env)
        assert events(log, "tool/call") and not events(log, "tool/result")
        assert not events(log, "turn/end")         # unfinished: `mini_agent -c` continues it
        if keep_alive:
            assert [e["data"]["exit_status"] for e in events(log, "keepalive/stop")] == [143]


# ---------- 5. --keepalive-session works only with the supervisor's token ----------

@pytest.mark.parametrize("token", [None, "not-the-token"])
def test_the_run_option_needs_the_supervisors_token(tmp_path, token):
    session = tmp_path / "held.jsonl"
    SessionLog(session).append("user/message", content="the task", workspace=str(tmp_path))
    lock = SessionLock(session).acquire()
    before = session.read_bytes()
    env = env_for(LLM_BACKEND="offline")
    if token:
        env[TOKEN_ENV] = token
    try:
        result = cli("--no-lifecycle", "--keepalive-session", str(session), "another task",
                     env=env, cwd=str(tmp_path))
    finally:
        lock.release()
    assert result.returncode == 4, result.stderr
    assert "--keepalive-session" in result.stderr
    assert session.read_bytes() == before


def test_the_run_option_is_not_in_the_help():
    assert "keepalive-session" not in cli("--help", env=env_for()).stdout
