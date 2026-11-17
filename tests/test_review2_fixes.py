"""Regression tests for the findings of the second independent review of v2.

Retries after a timeout while reading an error; malformed and corrupt replies; a key
never sent to a provider that did not issue it; a key echoed by a server never printed;
setup with a broken address; session files and the task -c continues; tool schemas
and wrapped risky tools.
"""
import functools
import io
import json
import os
import sys
import time

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mini_agent import Agent, tool
from mini_agent.cli import doctor, setup
from mini_agent.core.executor import execute
from mini_agent.core.policy import Approval
from mini_agent.llm import factory
from mini_agent.llm.backends.claude import ClaudeLLM
from mini_agent.llm.backends.claude_stream import _Masked, stream_claude
from mini_agent.llm.backends.openai_compat import OpenAICompatLLM
from mini_agent.llm.base import LLMError, mask
from mini_agent.session import SessionLog
from mini_agent.tools.spec import ToolSet
from mini_agent.ui import common
from mini_agent.ui.common import latest_session, new_session_path, resume_state


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """No settings from the real environment, home, working directory or package."""
    for key in factory.CONFIG_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MINI_AGENT_HOME", str(tmp_path / "agent_home"))
    monkeypatch.setattr(factory, "PACKAGE_ENV", tmp_path / "absent.env")
    monkeypatch.setattr(common, "SESS_DIR", tmp_path / "sessions")
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def mock_http(monkeypatch):
    """Route every httpx request (Client and httpx.post) to handler; returns the requests."""
    sent = []

    def install(handler):
        def server(request):
            sent.append(request)
            return handler(request)
        transport = httpx.MockTransport(server)
        real_client = httpx.Client
        monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(**dict(kw, transport=transport)))

        def post(url, **kw):
            kw.pop("timeout", None)
            with real_client(transport=transport) as client:
                return client.post(url, **kw)
        monkeypatch.setattr(httpx, "post", post)
        return sent
    return install


class Cut(httpx.SyncByteStream):
    """A body that times out after an optional first piece."""
    def __init__(self, prefix=b""):
        self.prefix = prefix

    def __iter__(self):
        if self.prefix:
            yield self.prefix
        raise httpx.ReadTimeout("cut off")


def no_wait(llm):
    llm.retry_waits = (0,)
    llm._sleep = lambda _: None
    return llm


# ---- retries: a final error whose text times out is still final ----

@pytest.mark.parametrize("backend", ["compat-stream", "compat", "claude-stream"])
@pytest.mark.parametrize("status", [400, 401, 403])
def test_error_status_body_timeout_is_not_retried(mock_http, backend, status):
    sent = mock_http(lambda request: httpx.Response(status, stream=Cut()))
    if backend == "claude-stream":
        llm = no_wait(ClaudeLLM(api_key="sk-ant-test-key", model="m", stream=True))
        reply = llm("hi")
    else:
        llm = no_wait(OpenAICompatLLM(model="m", base_url="https://mock.invalid/v1"))
        reply = llm.call_stream("hi", lambda *a, **kw: None) if backend == "compat-stream" else llm("hi")
    assert isinstance(reply, LLMError) and f"HTTP {status}" in reply
    assert len(sent) == 1, f"HTTP {status} was requested {len(sent)} times"


def test_retryable_status_body_timeout_is_retried(mock_http):
    replies = iter([httpx.Response(503, stream=Cut()),
                    httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})])
    sent = mock_http(lambda request: next(replies))
    llm = no_wait(OpenAICompatLLM(model="m", base_url="https://mock.invalid/v1"))
    assert (llm("hi"), len(sent)) == ("ok", 2)


def test_role_only_event_does_not_disable_timeout_retry():
    requests, shown = [], []

    def server(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, stream=Cut(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'))
        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n')
    llm = no_wait(OpenAICompatLLM(model="m", base_url="https://mock.invalid/v1",
                                  transport=httpx.MockTransport(server)))
    reply = llm.call_stream("hi", lambda text, **kw: shown.append(text))
    assert (reply, shown, len(requests)) == ("ok", ["ok"], 2)


def test_timeout_after_shown_text_is_not_repeated():
    requests = []

    def server(request):
        requests.append(request)
        return httpx.Response(200, stream=Cut(b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'))
    llm = no_wait(OpenAICompatLLM(model="m", base_url="https://mock.invalid/v1",
                                  transport=httpx.MockTransport(server)))
    reply = llm.call_stream("hi", lambda *a, **kw: None)
    assert isinstance(reply, LLMError) and "Timeout" in reply and len(requests) == 1


# ---- malformed and corrupt replies are errors, not crashes or silent successes ----

def backend(text):
    return OpenAICompatLLM(model="m", base_url="https://mock.invalid/v1",
                           transport=httpx.MockTransport(lambda request: httpx.Response(200, text=text)))


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("bad_field", ["function", "usage", "calls-not-list", "name"])
def test_nested_malformed_reply_returns_error(stream, bad_field):
    message = {"content": "hello"}
    usage = {"prompt_tokens": 2, "completion_tokens": 1}
    if bad_field == "function":
        message["tool_calls"] = [{"index": 0, "function": "not-an-object"}]
    elif bad_field == "calls-not-list":
        message["tool_calls"] = {"index": 0}
    elif bad_field == "name":
        message["tool_calls"] = [{"index": 0, "function": {"name": 7}}]
    else:
        usage["prompt_tokens"] = "not-a-count"
    data = {"choices": [{"delta" if stream else "message": message}], "usage": usage}
    text = "data: " + json.dumps(data) + "\n\n" if stream else json.dumps(data)
    llm = backend(text)
    result = llm.call_stream("hi", lambda *a, **kw: None) if stream else llm("hi")
    assert isinstance(result, LLMError) and "unexpected reply" in result


def test_null_counts_are_allowed():
    data = {"choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": None, "completion_tokens": 3}}
    llm = backend(json.dumps(data))
    assert llm("hi") == "hello" and llm.total_completion_tokens == 3


def test_malformed_event_is_not_silent_success():
    text = ('data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"lost"}}]\n\n'
            'data: [DONE]\n\n')
    result = backend(text).call_stream("hi", lambda *a, **kw: None)
    assert isinstance(result, LLMError) and "not JSON" in result


def test_json_fallback_rejects_missing_message():
    for stream in (True, False):
        llm = backend('{"choices":[{}]}')
        result = llm.call_stream("hi", lambda *a, **kw: None) if stream else llm("hi")
        assert isinstance(result, LLMError), f"missing message reported success: {result!r}"


# ---- a key is never sent to a provider that did not issue it ----

FOREIGN = "sk-or-v1-OPENROUTER-ONLY-TEST-KEY"


@pytest.mark.parametrize("target, label", [("openai", "OpenAI"), ("claude", "Anthropic"),
                                           ("deepseek", "DeepSeek"), ("groq", "Groq")])
def test_explicit_backend_refuses_a_foreign_generic_key(mock_http, target, label):
    sent = mock_http(lambda request: httpx.Response(401, json={"error": {"message": "rejected"}}))
    with pytest.raises(ValueError, match=f"not sent to {label}") as info:
        factory.create_llm({"LLM_BACKEND": target, "LLM_API_KEY": FOREIGN, "LLM_MODEL": "m"})
    assert "LLM_BACKEND=openrouter" in str(info.value) and FOREIGN not in str(info.value)
    assert sent == []


def test_explicit_backend_uses_its_own_key_instead(mock_http):
    sent = mock_http(lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}))
    llm = factory.create_llm({"LLM_BACKEND": "openai", "LLM_API_KEY": FOREIGN,
                              "OPENAI_API_KEY": "sk-proj-OPENAI-KEY-1234", "LLM_MODEL": "m"})
    assert llm("hi") == "ok"
    assert [r.headers["authorization"] for r in sent] == ["Bearer sk-proj-OPENAI-KEY-1234"]


def test_explicit_address_takes_the_key_where_it_points(mock_http):
    sent = mock_http(lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}))
    llm = factory.create_llm({"LLM_BACKEND": "deepseek", "DEEPSEEK_BASE_URL": "https://gateway.invalid/v1",
                              "LLM_API_KEY": FOREIGN})
    assert llm("hi") == "ok" and sent[0].url.host == "gateway.invalid"
    assert sent[0].headers["authorization"] == f"Bearer {FOREIGN}"


def test_local_server_gets_no_foreign_key(mock_http):
    sent = mock_http(lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}))
    llm = factory.create_llm({"LLM_BACKEND": "lmstudio", "LLM_API_KEY": FOREIGN, "LLM_MODEL": "m"})
    assert llm("hi") == "ok" and "authorization" not in sent[0].headers


def test_matching_generic_key_is_used(mock_http):
    sent = mock_http(lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}))
    llm = factory.create_llm({"LLM_BACKEND": "openrouter", "LLM_API_KEY": FOREIGN})
    assert llm("hi") == "ok" and sent[0].url.host == "openrouter.ai"


def test_explicit_url_wins_over_anthropic_hint():
    url = "https://company-gateway.invalid/v1"
    llm = factory.create_llm({"LLM_BASE_URL": url, "LLM_PROVIDER": "anthropic",
                              "LLM_API_KEY": "sk-ant-test-key", "LLM_MODEL": "m"})
    assert isinstance(llm, ClaudeLLM) and llm.base_url == url


# ---- a key echoed by a server is never printed ----

SECRET = "sk-ant-FAKE-KEY-DO-NOT-DISCLOSE"


def test_doctor_stream_does_not_print_echoed_key(mock_http, monkeypatch, capsys):
    monkeypatch.setenv("LLM_BACKEND", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("LLM_STREAM", "true")
    event = {"type": "content_block_delta", "delta": {"text": SECRET}}

    def server(request):
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, text="data: " + json.dumps(event) + "\n\n")
        return httpx.Response(200, json={"content": [{"type": "text", "text": "echo " + SECRET}]})
    sent = mock_http(server)
    doctor.main(["--test"])
    out = capsys.readouterr()
    assert SECRET not in out.out + out.err and mask(SECRET) in out.out and len(sent) == 1


def test_claude_stream_masks_a_key_split_across_chunks(mock_http, capsys):
    key = "sk-ant-api03-SPLIT-ACROSS-CHUNKS-xyz"
    pieces = ["the key is sk-ant-ap", "i03-SPLIT-ACROSS-", "CHUNKS-xyz, done"]
    body = "".join("data: " + json.dumps({"type": "content_block_delta", "delta": {"text": p}}) + "\n\n"
                   for p in pieces)
    mock_http(lambda request: httpx.Response(200, text=body))
    content, _, _, error = stream_claude(key, "m", "hi")
    out = capsys.readouterr().out
    assert error == "" and content == "".join(pieces)        # the reply itself is unchanged
    assert key not in out and out == f"the key is {mask(key)}, done\n"


def test_masked_printer_releases_text_that_is_not_the_key():
    shown = _Masked("sk-ant-SECRET-KEY-123")
    assert shown.feed("costs sk-") == "costs "               # "sk-" could start the key: held
    assert shown.feed("learn") == "sk-learn"
    assert shown.flush() == ""


def test_setup_bad_url_is_a_message_not_traceback(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("test-secret-key\n"))
    result = setup.main(["--provider", "compat", "--base-url", "http://host:bad-port/v1",
                         "--model", "m", "--key-stdin"])
    out = capsys.readouterr().out
    assert result != 0 and "failed" in out and "test-secret-key" not in out


# ---- session files and the task -c continues ----

def test_idle_sessions_do_not_share_log(tmp_path, monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1800000000)
    first = SessionLog(new_session_path(tmp_path))
    second = SessionLog(new_session_path(tmp_path))
    first.append("user/message", content="first session")
    second.append("user/message", content="second session")
    assert first.path != second.path, f"both sessions use {first.path.name}"


def test_latest_session_after_suffix_rollover(tmp_path, monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1800000000)
    for _ in range(100):
        newest = new_session_path(tmp_path)
        SessionLog(newest).append("user/message", content="task")
    assert newest.name == "1800000000_100.jsonl" and latest_session(tmp_path) == newest


def test_latest_session_skips_a_session_left_without_a_message(tmp_path, monkeypatch):
    now = [1800000000]
    monkeypatch.setattr(time, "time", lambda: now[0])
    used = new_session_path(tmp_path)
    SessionLog(used).append("user/message", content="task")
    now[0] += 1
    new_session_path(tmp_path)                    # opened later and left: an empty file
    assert latest_session(tmp_path) == used


def test_pending_user_message_is_not_finished(tmp_path):
    log = SessionLog(tmp_path / "session.jsonl")
    log.append("user/message", content="old task")
    log.append("turn/start", goal="old task")
    end = log.append("turn/end", reason="done", answer="finished")
    log.append("user/message", content="new task")
    # The process dies after the UI logs the new input, before turn/start.
    pending = log.unfinished_turn()
    assert pending is not None, "new logged user input was discarded"
    assert (pending["goal"], pending["pending"], pending["events"]) == ("new task", True, [])
    assert pending["before_seq"] == end["seq"]
    assert resume_state(pending) is None          # run as a new turn, not continued


def test_answered_user_message_is_not_pending(tmp_path):
    log = SessionLog(tmp_path / "session.jsonl")
    log.append("user/message", content="hi")
    log.append("assistant/message", content="hello")
    assert log.unfinished_turn() is None


def test_before_seq_excludes_user_even_with_metadata(tmp_path):
    log = SessionLog(tmp_path / "session.jsonl")
    log.append("assistant/message", content="previous conversation")
    log.append("user/message", content="current task")
    log.append("plan/mode", active=True)
    log.append("turn/start", goal="current task")
    pending = log.unfinished_turn()
    assert pending["pending"] is False
    assert log.derive_messages(until_seq=pending["before_seq"]) == [
        {"role": "assistant", "content": "previous conversation"}]


# ---- tool schemas and wrapped risky tools ----

def test_positional_defaults_can_be_skipped_by_name():
    @tool
    def f(x: int = 1, y: int = 2, /):
        return (x, y)
    tools = ToolSet.build([f])
    assert execute("f", {"y": 9}, tools.funcs) == str(f(1, 9))
    assert execute("f", {}, tools.funcs) == str(f())
    assert execute("f", {"x": 5}, tools.funcs) == str(f(5))


@pytest.mark.parametrize("depth", [3, 32, 300])
@pytest.mark.parametrize("policy", ["ask", "deny"])
def test_deep_wrapper_preserves_risk(tmp_path, policy, depth):
    called = []

    @tool(risky=True)
    def dangerous():
        called.append("executed")
        return "ok"
    wrapped = dangerous
    for _ in range(depth):
        def wrap(fn):
            @functools.wraps(fn, updated=())
            def wrapper():
                return fn()
            return wrapper
        wrapped = wrap(wrapped)
    replies = iter(['{"thought":"t","action":"dangerous","args":{}}',
                    '{"thought":"t","action":"done","args":{"answer":"ok"}}'])
    Agent(llm=lambda _: next(replies), tools=[wrapped], use_memory=False, skills=[],
          workspace=tmp_path, approve=Approval(policy, ask=lambda *a: "no")).run("task", verbose=False)
    assert called == [], f"{policy} allowed the risky wrapped function: {called!r}"


def test_empty_tuple_schema_rejects_nonempty():
    import typing

    @tool
    def f(value: tuple[()], other: typing.Tuple[()], any_tuple: typing.Tuple):
        return value
    props = f.__tool__.parameters["properties"]
    assert props["value"] == props["other"] == {"type": "array", "maxItems": 0}
    assert props["any_tuple"] == {"type": "array"}
