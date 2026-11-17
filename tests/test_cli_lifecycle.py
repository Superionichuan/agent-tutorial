"""The command line with the lifecycle: written memory, a token budget per task, -c after a stop."""
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from mini_agent.cli.main import DEFAULT_BUDGET, agent_options, build_parser
from mini_agent.core.agent.run import console_renderer
from mini_agent.ui.common import compact_notice, session_conversation, stop_notice
from test_cli import FakeServer, cli, env_for

CALC = json.dumps({"thought": "compute", "action": "calc", "args": {"expr": "6*7"}})
DONE = json.dumps({"thought": "done", "action": "done", "args": {"answer": "finished"}})


def options(*argv, env=None, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.delenv("MINI_AGENT_BUDGET", raising=False)
        for k, v in (env or {}).items():
            monkeypatch.setenv(k, v)
    return agent_options(build_parser().parse_args(list(argv)))


def test_the_command_line_runs_with_a_lifecycle_and_a_budget(monkeypatch):
    o = options("task", monkeypatch=monkeypatch)
    assert o["lifecycle"] is True and o["max_tokens"] == DEFAULT_BUDGET == 1_000_000
    o = options("--no-lifecycle", "task", monkeypatch=monkeypatch)
    assert o["lifecycle"] is None and "max_tokens" not in o       # the agent's own limit
    assert options("--budget", "5000", "task", monkeypatch=monkeypatch)["max_tokens"] == 5000
    assert options("task", env={"MINI_AGENT_BUDGET": "2_000_000"},
                   monkeypatch=monkeypatch)["max_tokens"] == 2_000_000
    assert options("--no-lifecycle", "--budget", "7", "task", monkeypatch=monkeypatch)["max_tokens"] == 7


@pytest.mark.parametrize("argv, env, message", [
    (["--budget", "0", "x"], {}, "--budget must be a positive number"),
    (["x"], {"MINI_AGENT_BUDGET": "lots"}, "MINI_AGENT_BUDGET must be a whole number"),
])
def test_a_bad_budget_is_explained(argv, env, message):
    r = cli(*argv, env=env_for(LLM_BACKEND="offline", **env))
    assert r.returncode == 2 and message in r.stderr and "Traceback" not in r.stderr


def test_a_task_writes_the_user_message_and_closes_its_handoff():
    server = FakeServer([CALC, DONE])
    work = tempfile.mkdtemp()
    try:
        r = cli("compute", "six", "times", "seven", env=env_for(server), cwd=work)
    finally:
        server.close()
    assert r.returncode == 0 and "Answer: finished" in r.stdout, r.stderr
    folder = Path(work) / ".mini_agent"
    assert "compute six times seven" in (folder / "user.md").read_text(encoding="utf-8")
    handoff = folder / "handoff.md"
    assert not handoff.exists() or "closed" in handoff.read_text(encoding="utf-8").lower()


def test_no_lifecycle_writes_nothing_in_the_workspace():
    server = FakeServer([DONE])
    work = tempfile.mkdtemp()
    try:
        r = cli("--no-lifecycle", "hello", env=env_for(server), cwd=work)
    finally:
        server.close()
    assert r.returncode == 0 and not (Path(work) / ".mini_agent").exists()


def test_budget_stop_writes_a_handoff_and_c_continues_with_a_new_budget():
    # each fake call uses 10 + 5 tokens: the second call passes a budget of 20
    server = FakeServer([CALC, CALC, DONE])
    work = tempfile.mkdtemp()
    env = env_for(server, MINI_AGENT_BUDGET="20")
    try:
        r = cli("--yes", "a long task", env=env, cwd=work)
        assert r.returncode == 1, r.stderr                  # stopped before the end
        assert "used up its token budget (30/20 tokens)" in r.stdout
        assert "`mini_agent -c` continues it with a new budget" in r.stdout
        handoff = (Path(work) / ".mini_agent" / "handoff.md").read_text(encoding="utf-8")
        assert "a long task" in handoff
        env.pop("MINI_AGENT_BUDGET")                    # -c: the default budget again
        r2 = cli("-c", "--yes", env=env, cwd=work)
    finally:
        server.close()
    assert r2.returncode == 0, r2.stderr
    # step 1 ran; step 2 was decided but stopped before its tool ran, so it is dropped
    assert "Continuing the unfinished task (token_limit, 1 step kept): a long task" in r2.stdout
    assert "finished" in r2.stdout and len(server.requests) == 3
    # the continued turn's first prompt carries the steps already done
    last = server.requests[-1]["messages"][0]["content"]
    assert last.count("calc") >= 2 and "42" in last


def test_session_conversation_uses_the_lifecycle_budget():
    class Lc:
        def conversation(self, log, until_seq=None):
            return f"budgeted:{log}:{until_seq}"

    class WithLc:
        lifecycle = Lc()

    class Plain:
        lifecycle = None
        llm = object()

    class Log:
        compacted = False

        def compact(self, llm):
            self.compacted = True

    assert session_conversation(WithLc(), "LOG", lambda limit=1600: "x") == "budgeted:LOG:None"
    assert session_conversation(WithLc(), "LOG", lambda limit=1600: "x",
                                until_seq=5) == "budgeted:LOG:5"   # -c: the text it started with
    log = Log()
    assert session_conversation(Plain(), log, lambda limit=1600: "x" * min(limit, 3000)) == "x" * 1600
    assert log.compacted                                 # the old rule: over 2,500 characters


def test_compaction_and_stop_notices():
    d = {"step": 12, "before_tokens": 61000, "after_tokens": 30500, "method": "mechanical"}
    assert compact_notice(d) == "✻ context compacted before step 12: 61,000 → 30,500 tokens (mechanical)"
    out = io.StringIO()
    with redirect_stdout(out):
        console_renderer("context/compact", **d)
    assert "[context compacted before step 12: 61,000 -> 30,500 tokens, mechanical]" in out.getvalue()
    assert stop_notice({"reason": "done"}) == "" and "-c" in stop_notice({"reason": "max_steps"})
