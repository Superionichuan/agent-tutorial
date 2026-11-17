"""`mini_agent -c`: the turn that did not finish is found in the session log and continued
from its last completed step; a tool call cut off before its result is not run again."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from mini_agent.session import SessionLog
from test_cli import FakeServer, cli, env_for


def log_with(*events):
    log = SessionLog(Path(tempfile.mkdtemp()) / "s.jsonl")
    for type, data in events:
        log.append(type, **data)
    return log


START = ("turn/start", {"goal": "g", "name": "agent"})


def test_a_finished_turn_is_not_continued():
    log = log_with(("user/message", {"content": "g"}), START,
                   ("turn/end", {"answer": "ok", "reason": "done"}))
    assert log.unfinished_turn() is None
    assert log_with().unfinished_turn() is None


@pytest.mark.parametrize("ending,reason", [
    ([], "interrupted"),
    ([("turn/end", {"answer": "x", "reason": "token_limit"})], "token_limit"),
    ([("turn/end", {"answer": "x", "reason": "llm_error"})], "llm_error"),
])
def test_a_cut_off_or_stopped_turn_is_found(ending, reason):
    log = log_with(("user/message", {"content": "first"}), START,
                   ("turn/end", {"answer": "a", "reason": "done"}),
                   ("plan/mode", {"active": False}),
                   ("user/message", {"content": "second"}),
                   ("turn/start", {"goal": "second", "name": "agent"}),
                   ("step/start", {"step": 1}), *ending)
    info = log.unfinished_turn()
    assert info["goal"] == "second" and info["reason"] == reason
    assert [e["type"] for e in info["events"]][:2] == ["turn/start", "step/start"]
    assert info["before_seq"] == 4                 # up to plan/mode; the turn's message is excluded
    texts = [m["content"] for m in log.derive_messages(until_seq=info["before_seq"])]
    assert texts == ["first"]


def test_continued_segments_belong_to_the_turn():
    log = log_with(("user/message", {"content": "g"}), START, ("step/start", {"step": 1}),
                   ("turn/start", {"goal": "g", "name": "agent", "resumed": True}),
                   ("step/start", {"step": 2}),
                   ("turn/end", {"answer": "x", "reason": "max_steps"}))
    info = log.unfinished_turn()
    assert info["reason"] == "max_steps" and len(info["events"]) == 5
    log.append("turn/start", goal="g", name="agent", resumed=True)
    log.append("turn/end", answer="ok", reason="done")
    assert log.unfinished_turn() is None


def test_continue_takes_no_task():
    r = cli("-c", "a new task", env=env_for(LLM_BACKEND="offline"))
    assert r.returncode == 2 and "without a task" in r.stderr


def test_a_one_shot_run_is_logged():
    server = FakeServer(['{"thought": "t", "action": "done", "args": {"answer": "42"}}'])
    try:
        env = env_for(server)
        r = cli("What is 6*7?", env=env)
        assert r.returncode == 0 and "Answer: 42" in r.stdout, r.stdout + r.stderr
        logs = list((Path(env["HOME"]) / ".mini_agent_sessions").glob("*.jsonl"))
        assert len(logs) == 1
        types = [json.loads(line)["type"] for line in logs[0].read_text().splitlines()]
        assert types[0] == "user/message" and types[-1] == "turn/end"
    finally:
        server.close()


@pytest.mark.skipif(os.name == "nt", reason="SIGINT to a child process: POSIX only")
def test_ctrl_c_in_a_one_shot_run_says_how_to_continue():
    import signal
    import time
    step = json.dumps({"thought": "t", "action": "bash", "args": {"cmd": "sleep 20"}})
    server = FakeServer([step])
    try:
        env = env_for(server)
        proc = subprocess.Popen([sys.executable, "-m", "mini_agent", "--yes", "wait"], env=env,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, cwd=tempfile.mkdtemp())
        log_dir = Path(env["HOME"]) / ".mini_agent_sessions"
        for _ in range(200):                         # wait until the tool call is under way
            logs = list(log_dir.glob("*.jsonl"))
            if logs and '"tool/call"' in logs[0].read_text():
                break
            time.sleep(0.05)
        proc.send_signal(signal.SIGINT)
        out, err = proc.communicate(timeout=30)
        assert proc.returncode == 130, out + err
        assert "mini_agent -c" in err and "Traceback" not in err
        assert SessionLog(logs[0]).unfinished_turn()["reason"] == "interrupted"
    finally:
        server.close()


def test_killed_mid_tool_then_continued():
    """The process dies inside a tool call; -c finishes the task; no tool runs twice."""
    pytest.importorskip("mini_agent.lifecycle")
    step = lambda cmd: json.dumps({"thought": "t", "action": "bash", "args": {"cmd": cmd}})
    server = FakeServer([
        step("echo one >> count.txt"),
        step("echo two >> count.txt; kill -9 $PPID"),        # the agent process dies here
        json.dumps({"thought": "t", "action": "done", "args": {"answer": "finished"}}),
    ])
    try:
        env = env_for(server)
        work = tempfile.mkdtemp()
        r = cli("--yes", "count to two", env=env, cwd=work)
        assert r.returncode != 0                                # killed
        assert (Path(work) / "count.txt").read_text() == "one\ntwo\n"
        r = cli("-c", "--yes", env=env, cwd=work)              # stdin closed: exits after the turn
        assert "finished" in r.stdout, r.stdout + r.stderr
        assert (Path(work) / "count.txt").read_text() == "one\ntwo\n"   # nothing ran twice
        assert len(server.requests) == 3
        told = server.requests[2]["messages"][0]["content"]
        assert "interrupted before this call finished" in told
        log = sorted((Path(env["HOME"]) / ".mini_agent_sessions").glob("*.jsonl"))[-1]
        assert SessionLog(log).unfinished_turn() is None
    finally:
        server.close()
