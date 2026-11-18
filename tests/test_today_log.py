"""Today's log holds what the agent writes there with its log tool, nothing else.

The interactive modes used to add "Q: <goal> -> A: <answer>" there after every task, and
today's log goes into every prompt. When the same task ran again later that day, a model
read that line as a claim about the task in front of it and kept re-checking it (one
approval per check). The session log already records every task and answer, so the
interactive modes no longer write the line.
Run: python -m pytest tests/test_today_log.py
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from mini_agent import Agent  # noqa: E402
from mini_agent.llm.base import BaseLLM  # noqa: E402
from test_cli import FakeServer, cli, env_for  # noqa: E402

WRITE = json.dumps({"thought": "write it", "action": "write_file",
                    "args": {"path": "note.txt", "content": "hello"}})


def done(answer):
    return json.dumps({"thought": "t", "action": "done", "args": {"answer": answer}})


def daily_logs(home):
    """Every day's log in a mini_agent home (not the long-term file), as {name: text}."""
    folder = Path(home) / "memory"
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.md"))
            if p.name != "long_term.md"} if folder.is_dir() else {}


def memory_of(prompt):
    return prompt.split("[Memory]\n", 1)[1].split("\n\n[Steps taken so far]", 1)[0]


def prompt_of(request):
    return "\n".join(str(m.get("content", "")) for m in request["messages"])


class Scripted(BaseLLM):
    def __init__(self, replies):
        self.replies, self.prompts, self.model = list(replies), [], "scripted"
        self.total_prompt_tokens = self.total_completion_tokens = 0

    def call(self, prompt):
        self.prompts.append(prompt)
        self.total_prompt_tokens += 10
        self.total_completion_tokens += 5
        return self.replies.pop(0)


def test_the_log_tool_still_writes_todays_log_and_later_prompts_show_it(tmp_path, monkeypatch):
    monkeypatch.setenv("MINI_AGENT_HOME", str(tmp_path / "home"))
    note = json.dumps({"thought": "note it", "action": "log", "args": {"content": "the build takes 4 min"}})
    first = Scripted([note, done("noted")])
    assert Agent(llm=first, workspace=str(tmp_path / "ws"), skills=[]).run("note it", verbose=False) == "noted"
    [(name, text)] = daily_logs(tmp_path / "home").items()
    assert "] the build takes 4 min" in text and "Q: " not in text      # the log tool's own line
    later = Scripted([done("ok")])
    Agent(llm=later, workspace=str(tmp_path / "ws"), skills=[]).run("anything", verbose=False)
    assert memory_of(later.prompts[0]) == "### Today's log\n" + text


def test_no_memory_no_section(tmp_path, monkeypatch):
    monkeypatch.setenv("MINI_AGENT_HOME", str(tmp_path / "home"))
    llm = Scripted([done("ok")])
    Agent(llm=llm, workspace=str(tmp_path / "ws"), skills=[]).run("anything", verbose=False)
    assert memory_of(llm.prompts[0]) == "(none)"


def test_a_task_continued_with_c_leaves_todays_log_as_it_was():
    # no terminal: the task stops for approval; -c --yes continues it once (the interactive
    # mode's code, without a prompt) and finishes it
    server = FakeServer([WRITE, WRITE, done("finished")])
    try:
        env = env_for(server)
        work = tempfile.mkdtemp()
        assert cli("write hello into note.txt", env=env, cwd=work).returncode == 3
        r = cli("-c", "--yes", env=env, cwd=work)
        assert r.returncode == 0 and "finished" in r.stdout, r.stdout + r.stderr
    finally:
        server.close()
    assert daily_logs(env["MINI_AGENT_HOME"]) == {}


def one_interactive_session(env, task, answer):
    """mini_agent in a pseudo-terminal, in a new folder: one task, then /exit."""
    import pty
    import select

    master, slave = pty.openpty()
    proc = subprocess.Popen([sys.executable, "-m", "mini_agent"], env=env, stdin=slave,
                            stdout=slave, stderr=slave, cwd=tempfile.mkdtemp())
    os.close(slave)
    seen = b""

    def read_some(timeout):
        nonlocal seen
        if select.select([master], [], [], timeout)[0]:
            try:
                seen += os.read(master, 65536)
            except OSError:
                return False
        return True

    def wait_for(text, start=0, timeout=60):
        end = time.time() + timeout
        while text not in seen[start:]:
            if time.time() > end:
                raise AssertionError(f"did not see {text!r}; output so far:\n{seen[-2000:]!r}")
            if not read_some(0.2):
                break
        return seen.index(text, start) + len(text)

    try:
        wait_for(b"\xe2\x9d\xaf")                      # the prompt
        os.write(master, task.encode() + b"\r")
        after = wait_for(answer.encode(), start=len(seen))
        wait_for(b"\xe2\x9d\xaf", start=after)         # the next prompt, then /exit
        for _ in range(10):                            # again every few seconds until it ends
            try:
                os.write(master, b"/exit\r")
                proc.wait(timeout=3)
                break
            except OSError:
                break
            except subprocess.TimeoutExpired:
                while select.select([master], [], [], 0)[0] and read_some(0):
                    pass                               # keep reading what it prints
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
        os.close(master)


@pytest.mark.skipif(os.name == "nt", reason="pseudo-terminals are POSIX-only")
def test_the_same_task_in_a_later_session_starts_from_the_same_memory():
    # two interactive sessions on one day, each in its own folder, the same task in both
    server = FakeServer([done("42"), done("42")])
    env = env_for(server, TERM="xterm")
    try:
        one_interactive_session(env, "Compute 6*7.", "42")
        one_interactive_session(env, "Compute 6*7.", "42")
    finally:
        server.close()
    assert daily_logs(env["MINI_AGENT_HOME"]) == {}
    first, second = (memory_of(prompt_of(r)) for r in server.requests)
    assert second == first and "Compute 6*7." not in second, second


def test_a_task_in_the_full_screen_interface_leaves_todays_log_as_it_was(monkeypatch, tmp_path):
    pytest.importorskip("textual")
    server = FakeServer([done("42")])
    try:
        for key, value in env_for(server).items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("MINI_AGENT_HOME", str(tmp_path / ".mini_agent"))
        monkeypatch.chdir(tmp_path)
        from mini_agent.ui import tui
        monkeypatch.setattr(tui, "SESS_DIR", tmp_path / "sessions")

        async def scenario():
            app = tui.MiniAgentApp()
            async with app.run_test() as pilot:
                app.query_one(tui.Input).value = "Compute 6*7."
                await pilot.press("enter")
                for _ in range(300):
                    await pilot.pause(0.05)
                    if server.requests and not any(w.is_running for w in app.workers):
                        break
                await app.workers.wait_for_complete()

        asyncio.run(scenario())
    finally:
        server.close()
    assert len(server.requests) == 1
    assert daily_logs(tmp_path / ".mini_agent") == {}
