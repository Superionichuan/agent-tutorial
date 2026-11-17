"""Retrying a model call: 429, 5xx and timeouts are tried again (at most 3 times, growing
waits, Retry-After honoured); 400, 401, 403 and connection errors stop at once."""
import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from mini_agent import Agent
from mini_agent.llm import retry
from mini_agent.llm.backends.claude import ClaudeLLM
from mini_agent.llm.backends.ollama import OllamaLLM
from mini_agent.llm.backends.openai_compat import OpenAICompatLLM
from mini_agent.llm.base import LLMError

OK = {"choices": [{"message": {"content": "hello"}}], "usage": {"prompt_tokens": 2, "completion_tokens": 1}}
DONE = {"choices": [{"message": {"content": '{"thought": "t", "action": "done", "args": {"answer": "42"}}'}}]}


class Script:
    """A fake server answering from a list: a status, (status, headers), an exception, or a body."""

    def __init__(self, *answers, body=OK, stream=False):
        self.answers, self.body, self.stream, self.requests = list(answers), body, stream, 0

    def __call__(self, request):
        self.requests += 1
        answer = self.answers.pop(0) if self.answers else 200
        if isinstance(answer, Exception):
            raise answer
        status, headers = answer if isinstance(answer, tuple) else (answer, {})
        if status != 200:
            return httpx.Response(status, json={"error": {"message": f"status {status}"}}, headers=headers)
        if self.stream:
            text = ("data: " + json.dumps({"choices": [{"delta": {"content": "hello"}}]}) + "\n\n"
                    "data: [DONE]\n\n")
            return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=self.body)


def compat(server):
    llm = OpenAICompatLLM(api_key="fake-key-12345", model="m", base_url="https://fake.invalid/v1",
                          transport=httpx.MockTransport(server))
    waits = []
    llm._sleep = waits.append
    return llm, waits


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 529])
@pytest.mark.parametrize("stream", [False, True])
def test_a_passing_failure_is_tried_again(status, stream):
    server = Script(status, status, stream=stream)
    llm, waits = compat(server)
    reply = llm.call_stream("hi", lambda *a, **k: None) if stream else llm("hi")
    assert reply == "hello" and not isinstance(reply, LLMError)
    assert server.requests == 3 and waits == [1.0, 2.0]


@pytest.mark.parametrize("stream", [False, True])
def test_at_most_three_retries(stream):
    server = Script(503, 503, 503, 503, 503, stream=stream)
    llm, waits = compat(server)
    reply = llm.call_stream("hi", lambda *a, **k: None) if stream else llm("hi")
    assert isinstance(reply, LLMError) and "HTTP 503" in reply
    assert server.requests == 4 and waits == [1.0, 2.0, 4.0]


@pytest.mark.parametrize("status", [400, 401, 403, 404])
@pytest.mark.parametrize("stream", [False, True])
def test_a_request_error_stops_at_once(status, stream):
    server = Script(status, stream=stream)
    llm, waits = compat(server)
    reply = llm.call_stream("hi", lambda *a, **k: None) if stream else llm("hi")
    assert isinstance(reply, LLMError) and f"HTTP {status}" in reply
    assert server.requests == 1 and waits == []


@pytest.mark.parametrize("stream", [False, True])
def test_a_timeout_is_tried_again(stream):
    server = Script(httpx.ReadTimeout("slow"), httpx.ConnectTimeout("slow"), stream=stream)
    llm, waits = compat(server)
    reply = llm.call_stream("hi", lambda *a, **k: None) if stream else llm("hi")
    assert reply == "hello" and server.requests == 3 and waits == [1.0, 2.0]


@pytest.mark.parametrize("stream", [False, True])
def test_a_refused_connection_is_not_tried_again(stream):
    server = Script(httpx.ConnectError("refused"), stream=stream)
    llm, waits = compat(server)
    reply = llm.call_stream("hi", lambda *a, **k: None) if stream else llm("hi")
    assert isinstance(reply, LLMError) and "ConnectError" in reply
    assert server.requests == 1 and waits == []


def test_retry_after_is_honoured_up_to_a_limit():
    server = Script((429, {"Retry-After": "7"}), (429, {"Retry-After": "120"}), (503, {"Retry-After": "soon"}))
    llm, waits = compat(server)
    assert llm("hi") == "hello"
    assert waits == [7.0, retry.MAX_WAIT, 4.0]


def test_no_retries_when_switched_off():
    server = Script(429)
    llm, waits = compat(server)
    llm.retry_waits = ()
    assert isinstance(llm("hi"), LLMError) and server.requests == 1 and waits == []


def test_a_stream_cut_off_after_some_text_is_not_repeated():
    calls = []

    class Cut(httpx.SyncByteStream):
        def __iter__(self):
            yield b'data: {"choices": [{"delta": {"content": "hel"}}]}\n\n'
            raise httpx.ReadTimeout("cut off")

    def server(request):
        calls.append(1)
        return httpx.Response(200, stream=Cut(), headers={"content-type": "text/event-stream"})
    llm, waits = compat(server)
    shown = []
    reply = llm.call_stream("hi", lambda text, kind="content": shown.append(text))
    assert isinstance(reply, LLMError) and "ReadTimeout" in reply
    assert calls == [1] and waits == [] and shown == ["hel"]


def test_the_agent_finishes_after_a_rate_limit():
    server = Script(429, body=DONE)
    llm, waits = compat(server)
    events = []
    answer = Agent(llm=llm, use_memory=False, skills=[], workspace=str(os.getcwd())).run(
        "goal", on_event=lambda t, **d: events.append((t, d)))
    assert answer == "42" and server.requests == 2 and waits == [1.0]
    assert [d["reason"] for t, d in events if t == "turn/end"] == ["done"]


# ---- the other backends ----

def patch_httpx(monkeypatch, server):
    real_client, real_post = httpx.Client, httpx.post
    transport = httpx.MockTransport(server)
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: real_client(*a, **dict(kw, transport=transport)))

    def post(url, **kw):
        with real_client(transport=transport) as client:
            kw.pop("timeout", None)
            return client.post(url, **kw)
    monkeypatch.setattr(httpx, "post", post)


def test_ollama_retries_a_server_error(monkeypatch):
    server = Script(500, 503, body={"response": "hello", "prompt_eval_count": 2, "eval_count": 1})
    patch_httpx(monkeypatch, server)
    llm = OllamaLLM()
    waits = []
    llm._sleep = waits.append
    assert llm("hi") == "hello" and server.requests == 3 and waits == [1.0, 2.0]


def test_claude_retries_an_overload(monkeypatch):
    server = Script(529, body={"content": [{"type": "text", "text": "hello"}],
                               "usage": {"input_tokens": 2, "output_tokens": 1}})
    patch_httpx(monkeypatch, server)
    llm = ClaudeLLM(api_key="sk-ant-fake-key", model="m")
    waits = []
    llm._sleep = waits.append
    assert llm("hi") == "hello" and server.requests == 2 and waits == [1.0]


def test_claude_stream_reports_a_bad_key(monkeypatch, capsys):
    secret = "sk-ant-fake-THIS_MUST_NOT_BE_PRINTED"

    def server(request):
        return httpx.Response(401, json={"type": "error",
                                         "error": {"type": "authentication_error",
                                                   "message": "invalid x-api-key " + secret}})
    patch_httpx(monkeypatch, server)
    llm = ClaudeLLM(api_key=secret, model="m", stream=True)
    reply = llm("hi")
    assert isinstance(reply, LLMError) and "HTTP 401" in reply and secret not in reply
