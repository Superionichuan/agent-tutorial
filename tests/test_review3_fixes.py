"""Regression tests for the third review: replies, sessions, the command line, the loop."""
import importlib
import json

import httpx
import pytest

from mini_agent.core.agent.run import run_agent
from mini_agent.core.state import State
from mini_agent.lifecycle import Lifecycle
from mini_agent.llm import factory
from mini_agent.llm.backends.claude import ClaudeLLM
from mini_agent.llm.backends.openai_compat import OpenAICompatLLM
from mini_agent.llm.base import LLMError
from mini_agent.session import SessionLog
from mini_agent.ui import common
from mini_agent.ui.common import latest_session, mark_message

cli = importlib.import_module("mini_agent.cli.main")   # the module (mini_agent.cli.main is also a function)

DONE = '{"thought":"","action":"done","args":{"answer":"ok"}}'
CALL = '{"thought":"","action":"read","args":{}}'


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    for key in factory.CONFIG_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("MINI_AGENT_BUDGET", raising=False)
    monkeypatch.setenv("MINI_AGENT_HOME", str(tmp_path / "agent_home"))
    monkeypatch.setattr(factory, "PACKAGE_ENV", tmp_path / "absent.env")
    monkeypatch.setattr(common, "SESS_DIR", tmp_path / "sessions")
    monkeypatch.chdir(tmp_path)


def compat(status=200, **response):
    return OpenAICompatLLM(model="m", base_url="https://mock.invalid/v1",
                           transport=httpx.MockTransport(lambda r: httpx.Response(status, **response)))


# ---- replies: malformed counts and indexes are errors, not crashes or wrong totals ----

@pytest.mark.parametrize("index", [[], {}, "0", -1, True, 1.5])
def test_malformed_stream_index_is_an_error(index):
    data = {"choices": [{"delta": {"tool_calls": [
        {"index": index, "function": {"name": "calc", "arguments": "{}"}}]}}]}
    llm = compat(text="data: " + json.dumps(data) + "\n\ndata: [DONE]\n\n")
    assert isinstance(llm.call_stream("hi", lambda *a, **k: None), LLMError)


@pytest.mark.parametrize("usage", [{"prompt_tokens": -100, "completion_tokens": 1},
                                   {"prompt_tokens": 1, "completion_tokens": -1}])
def test_negative_usage_is_an_error(usage):
    llm = compat(json={"choices": [{"message": {"content": "ok"}}], "usage": usage})
    reply = llm("hi")
    assert isinstance(reply, LLMError) and llm.total_tokens == 0


def test_a_well_formed_index_and_usage_still_work():
    llm = compat(json={"choices": [{"message": {"content": "ok"}}],
                       "usage": {"prompt_tokens": 0, "completion_tokens": 3}})
    assert llm("hi") == "ok" and llm.total_tokens == 3


@pytest.mark.parametrize("usage", [{"input_tokens": -5, "output_tokens": 1},
                                   {"input_tokens": "5", "output_tokens": 1},
                                   {"input_tokens": 2, "output_tokens": None}, "lots"])
def test_claude_malformed_counts_are_an_error(monkeypatch, usage):
    body = {"content": [{"type": "text", "text": "hello"}], "usage": usage}
    real_client = httpx.Client

    def post(url, **kw):
        with real_client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))) as c:
            kw.pop("timeout", None)
            return c.post(url, **kw)
    monkeypatch.setattr(httpx, "post", post)
    llm = ClaudeLLM(api_key="sk-ant-fake-key", model="m")
    reply = llm("hi")
    assert isinstance(reply, LLMError) and llm.total_prompt_tokens == 0


# ---- sessions: -c finds the task that did not finish ----

def test_a_settings_only_session_does_not_hide_an_unfinished_task(tmp_path):
    used = tmp_path / "100.jsonl"
    log = SessionLog(used)
    log.append("user/message", content="unfinished task")
    log.append("turn/start", goal="unfinished task")
    SessionLog(tmp_path / "101.jsonl").append("plan/mode", active=True)
    (tmp_path / "102.jsonl").write_text("")                   # opened and left at once
    assert latest_session(tmp_path) == used


def test_a_pending_task_names_its_message(tmp_path):
    log = SessionLog(tmp_path / "s.jsonl")
    log.append("plan/mode", active=False)
    sent = log.append("user/message", content="task")
    unfinished = log.unfinished_turn()
    assert unfinished["pending"] and unfinished["message_seq"] == sent["seq"]


def test_mark_message_names_session_and_seq(tmp_path):
    class A:
        lifecycle = Lifecycle(workspace=tmp_path)
    log = SessionLog(tmp_path / "1800000000_02.jsonl")
    mark_message(A, log, log.append("user/message", content="hi")["seq"])
    assert A.lifecycle.message_id == "1800000000_02.jsonl#1"
    mark_message(type("NoLifecycle", (), {"lifecycle": None}), log, 1)   # nothing to mark


# ---- the command line: settings that cannot work are a message ----

@pytest.mark.parametrize("args", [["--backend", "openai", "--model", "m"],
                                  ["--tui", "--backend", "openai", "--model", "m"]])
def test_interactive_foreign_key_is_a_message(monkeypatch, capsys, args):
    monkeypatch.setattr(cli, "setup_problem", lambda: "")
    monkeypatch.setenv("LLM_API_KEY", "sk-or-v1-OPENROUTER-ONLY-TEST-KEY")
    assert cli.main(args) == 2
    assert "not sent to OpenAI" in capsys.readouterr().err


# ---- the loop: step numbers and the handoff ----

def test_a_reused_state_goes_on_numbering():
    state, events = State("task"), []
    emit = lambda t, **d: events.append((t, d))   # noqa: E731
    run_agent(lambda p: CALL, {"read": lambda: "result"}, "", "task", state=state,
              max_steps=1, on_event=emit)
    run_agent(lambda p: DONE, {}, "", "task", state=state, max_steps=1, on_event=emit)
    assert [d["step"] for t, d in events if t == "step/start"] == [1, 2]
    assert [d["step"] for t, d in events if t == "tool/call"] == [1]


@pytest.mark.parametrize("error", [KeyboardInterrupt, RuntimeError])
def test_run_agent_leaves_a_handoff_when_the_call_raises(tmp_path, error):
    def model(prompt):
        raise error("stopped")
    lc = Lifecycle(workspace=tmp_path, llm=model)
    with pytest.raises(error):
        run_agent(model, {}, "", "unfinished", lifecycle=lc, verbose=False)
    handoff = tmp_path / ".mini_agent" / "handoff.md"
    assert handoff.exists() and "unfinished" in handoff.read_text()


def test_a_pending_task_run_again_is_recorded_once(tmp_path):
    """Sent, interrupted before it started, then run again by -c: user.md has it once, because
    the command line marks both runs with the logged message (as the REPL and TUI do)."""
    from mini_agent import Agent
    from mini_agent.lifecycle import notes
    from mini_agent.ui.common import resume_state
    log = SessionLog(tmp_path / "session.jsonl")
    seq = log.append("user/message", content="task")["seq"]
    agent = Agent(llm=lambda p: DONE, workspace=tmp_path, lifecycle=True,
                  use_memory=False, skills=[])

    def interrupt(kind, **data):
        if kind == "turn/start":
            raise KeyboardInterrupt
        log.append(kind, **data)

    mark_message(agent, log, seq)
    with pytest.raises(KeyboardInterrupt):
        agent.run("task", on_event=interrupt)
    pending = log.unfinished_turn()
    assert pending["pending"] and pending["message_seq"] == seq
    mark_message(agent, log, pending["message_seq"])
    agent.run(pending["goal"], state=resume_state(pending), on_event=log.append)
    recorded = [m[2] for m in notes.user_messages(tmp_path / ".mini_agent" / "user.md")]
    assert recorded == ["task"]


def test_continue_without_a_lifecycle_gets_the_text_it_started_with():
    calls = []

    class Plain:
        lifecycle = None
        llm = None

    def plain_text(limit=1600, until_seq=None):
        calls.append(until_seq)
        return f"text up to {until_seq}"

    assert common.session_conversation(Plain(), None, plain_text, until_seq=7) == "text up to 7"
    assert calls == [7]


def test_a_full_context_window_has_its_own_notice():
    notice = common.stop_notice({"reason": "context_full"})
    assert "context window" in notice and "mini_agent -c" in notice
    assert common.stop_notice({"reason": "done"}) == ""
