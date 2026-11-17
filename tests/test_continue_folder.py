"""`mini_agent -c` works on the current folder only; with no terminal it behaves like a
one-shot task; an offline task with no recording, and a key the provider refuses, say
plainly what to do. All run for real: python -m mini_agent against a fake server."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from mini_agent.session import SessionLog
from mini_agent.ui.common import latest_folder_elsewhere, latest_session, nothing_to_continue
from test_cli import BASH_THEN_DONE, FakeServer, cli, env_for

WRITE = json.dumps({"thought": "write it", "action": "write_file",
                    "args": {"path": "note.txt", "content": "hello"}})
DONE = json.dumps({"thought": "done", "action": "done", "args": {"answer": "finished"}})


def sessions(env):
    return sorted((Path(env["MINI_AGENT_HOME"]) / "sessions").glob("*.jsonl"))


def test_continue_in_another_folder_runs_nothing_and_names_the_folder():
    server = FakeServer([WRITE, WRITE, DONE])     # refused (no terminal), then run with --yes
    try:
        env = env_for(server)
        first, other = tempfile.mkdtemp(), tempfile.mkdtemp()
        r = cli("write hello into note.txt", env=env, cwd=first)
        assert r.returncode == 3, r.stdout + r.stderr          # needs approval, no terminal
        asked = len(server.requests)
        r = cli("-c", "--yes", env=env, cwd=other)
        assert r.returncode == 2, r.stdout + r.stderr
        assert "no unfinished task to continue in this folder" in r.stderr
        assert str(Path(first).resolve()) in r.stderr and "cd there" in r.stderr
        assert len(server.requests) == asked                    # no model call
        assert os.listdir(other) == []                          # nothing written here
        r = cli("-c", "--yes", env=env, cwd=first)              # in its own folder it continues
        assert r.returncode == 0 and "finished" in r.stdout, r.stdout + r.stderr
        assert (Path(first) / "note.txt").read_text() == "hello"
    finally:
        server.close()


def test_continue_without_a_terminal_exits_with_the_task_status():
    """No prompt after the continued task; the status is the task's (here 3: it stops
    again, because without --yes nobody can approve the call)."""
    server = FakeServer([WRITE, WRITE])
    try:
        env = env_for(server)
        work = tempfile.mkdtemp()
        assert cli("write hello into note.txt", env=env, cwd=work).returncode == 3
        proc = subprocess.run([sys.executable, "-m", "mini_agent", "-c"], env=env, cwd=work,
                              input="", capture_output=True, encoding="utf-8", timeout=60)
        assert proc.returncode == 3, proc.stdout + proc.stderr
        assert "❯" not in proc.stdout                           # no interactive prompt
    finally:
        server.close()


def test_continue_with_stdin_left_open_does_not_wait():
    """A script that keeps stdin open (a pipe nobody writes to) still gets its exit."""
    server = FakeServer([WRITE, DONE])
    try:
        env = env_for(server)
        work = tempfile.mkdtemp()
        assert cli("write hello into note.txt", env=env, cwd=work).returncode == 3
        out_path = Path(tempfile.mkdtemp()) / "out.txt"
        with open(out_path, "w", encoding="utf-8") as out:
            proc = subprocess.Popen([sys.executable, "-m", "mini_agent", "-c", "--yes"], env=env,
                                    cwd=work, stdin=subprocess.PIPE, stdout=out, stderr=out)
            try:
                proc.wait(timeout=60)                 # stdin stays open: nobody writes or closes it
            except subprocess.TimeoutExpired:
                proc.kill()
                raise AssertionError("mini_agent -c waited for input")
            finally:
                proc.stdin.close()
        text = out_path.read_text(encoding="utf-8")
        assert proc.returncode == 0 and "finished" in text, text
    finally:
        server.close()


def test_nothing_unfinished_here_without_a_terminal():
    server = FakeServer([DONE])
    try:
        env = env_for(server)
        work = tempfile.mkdtemp()
        assert cli("say finished", env=env, cwd=work).returncode == 0
        r = cli("-c", env=env, cwd=work)
        assert r.returncode == 2 and "no unfinished task to continue in this folder" in r.stderr
        assert "Traceback" not in r.stderr
    finally:
        server.close()


def test_no_session_anywhere():
    r = cli("-c", env=env_for(LLM_BACKEND="offline"))
    assert r.returncode == 2 and "no unfinished task to continue" in r.stderr
    assert "Start one with" in r.stderr


def test_session_logs_record_their_folder_and_live_in_the_mini_agent_home():
    server = FakeServer([DONE])
    try:
        env = env_for(server)
        work = tempfile.mkdtemp()
        assert cli("say finished", env=env, cwd=work).returncode == 0
        logs = sessions(env)
        assert len(logs) == 1 and not (Path(env["HOME"]) / ".mini_agent_sessions").exists()
        message = next(e for e in SessionLog(logs[0]).events() if e["type"] == "user/message")
        assert message["data"]["workspace"] == str(Path(work).resolve())
    finally:
        server.close()


def test_finding_sessions_by_folder(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    logs = tmp_path / "sessions"
    logs.mkdir()

    def session(name, folder, finished):
        log = SessionLog(logs / f"{name}.jsonl")
        log.append("user/message", content="task", **({"workspace": str(folder)} if folder else {}))
        log.append("turn/start", goal="task", name="agent")
        if finished:
            log.append("turn/end", answer="ok", reason="done")

    session("100", a, finished=False)
    session("200", b, finished=True)
    session("300", None, finished=False)            # an old log without its folder
    assert latest_session(logs, workspace=a) == logs / "100.jsonl"
    assert latest_session(logs, workspace=b) == logs / "200.jsonl"
    assert latest_session(logs, workspace=b, unfinished=True) is None
    assert latest_session(logs) == logs / "300.jsonl"        # no folder given: any session
    assert latest_folder_elsewhere(b, logs) == str(a)
    assert latest_folder_elsewhere(a, logs) is None           # b's task is finished
    assert latest_folder_elsewhere(a, logs, unfinished=False) == str(b)
    assert nothing_to_continue(a, terminal=False, directory=logs) == ""
    why = nothing_to_continue(b, terminal=False, directory=logs)
    assert "no unfinished task" in why and str(a) in why
    assert nothing_to_continue(b, terminal=True, directory=logs) == ""   # b's session resumes
    why = nothing_to_continue(tmp_path, terminal=True, directory=logs)
    assert "no session" in why and str(a) in why             # the unfinished task first


def test_offline_task_with_no_recording_from_the_command_line():
    r = cli("Something that was never recorded.", env=env_for(LLM_BACKEND="offline"))
    assert r.returncode == 1, r.stdout + r.stderr
    assert ("Offline mode only replays the tutorial's recorded tasks; run `mini_agent setup` "
            "to use a model.") in r.stdout
    # not the notebook's wording, not counted as invalid replies, no internal line
    assert "setup cell" not in r.stdout and "invalid replies" not in r.stdout
    assert "Action:" not in r.stdout and "Answer:" not in r.stdout
