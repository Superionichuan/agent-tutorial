"""The OpenAI-compatible backend and backend selection, against a fake server (no network)."""
import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mini_agent.llm import (ClaudeLLM, DeepSeekLLM, OllamaLLM, OpenAICompatLLM, OfflineLLM,
                            choose_backend, create_llm, detect_provider)
from mini_agent.tools import BUILTIN_SPECS


class FakeServer:
    """An httpx transport that answers /chat/completions and records the requests."""

    def __init__(self, reply=None, stream=None, status=200, raise_error=None):
        self.reply, self.stream, self.status, self.raise_error = reply, stream, status, raise_error
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.raise_error:
            raise self.raise_error
        body = json.loads(request.content)
        self.requests.append((str(request.url), dict(request.headers), body))
        if body.get("stream"):
            text = "".join(f"{line}\n" for line in self.stream)
            return httpx.Response(self.status, text=text, headers={"content-type": "text/event-stream"})
        return httpx.Response(self.status, json=self.reply)

    def llm(self, **kw):
        kw.setdefault("model", "m")
        kw.setdefault("base_url", "https://example.test/v1")
        return OpenAICompatLLM(transport=httpx.MockTransport(self), **kw)


def chunk(delta=None, usage=None, finish=None):
    data = {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}] if delta is not None
            else []}
    if usage:
        data["usage"] = usage
    return "data: " + json.dumps(data)


OK = {"choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
      "usage": {"prompt_tokens": 7, "completion_tokens": 2}, "model": "m"}


def test_call_sends_one_user_message_and_counts_tokens():
    server = FakeServer(OK)
    llm = server.llm(api_key="k")
    assert llm("hi") == "hello"
    url, headers, body = server.requests[0]
    assert url == "https://example.test/v1/chat/completions"
    assert headers["authorization"] == "Bearer k"
    assert body == {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    assert (llm.total_prompt_tokens, llm.total_completion_tokens) == (7, 2)
    r = llm.call_with_usage("hi")
    assert (r.content, r.prompt_tokens, r.completion_tokens, r.total_tokens) == ("hello", 7, 2, 9)


def test_no_key_sends_no_authorization_header():
    server = FakeServer(OK)
    server.llm()("hi")
    assert "authorization" not in server.requests[0][1]


def test_stream_text_reasoning_usage_and_comment_lines():
    server = FakeServer(stream=[
        ": OPENROUTER PROCESSING", "",
        chunk({"reasoning": "think "}), chunk({"reasoning_content": "more"}),
        chunk({"content": "hel"}), chunk({"content": "lo"}, finish="stop"),
        chunk(None, usage={"prompt_tokens": 3, "completion_tokens": 4}), "data: [DONE]"])
    llm = server.llm()
    seen = []
    assert llm.call_stream("hi", lambda text, kind="content": seen.append((kind, text))) == "hello"
    assert seen == [("reasoning", "think "), ("reasoning", "more"), ("content", "hel"), ("content", "lo")]
    body = server.requests[0][2]
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert (llm.total_prompt_tokens, llm.total_completion_tokens) == (3, 4)
    old_style = []
    server.llm().call_stream("hi", lambda text: old_style.append(text))   # one-argument callbacks
    assert "".join(old_style) == "think morehello"


def test_errors_come_back_as_text():
    bad_key = FakeServer({"error": {"message": "invalid api key"}}, status=401)
    assert bad_key.llm()("hi") == "Error: HTTP 401: invalid api key"
    down = FakeServer(raise_error=httpx.ConnectError("refused"))
    assert down.llm()("hi").startswith("Error: ConnectError")
    in_body = FakeServer({"error": {"message": "no credit"}})
    assert in_body.llm()("hi") == "Error: HTTP 200: no credit"
    stream_err = FakeServer(stream=['data: {"error": {"message": "rate limited"}}'])
    assert stream_err.llm().call_stream("hi", lambda t: None) == "Error: HTTP 200: rate limited"
    assert OpenAICompatLLM(model="m")("hi").startswith("Error: no server address")
    assert OpenAICompatLLM(base_url="https://x.test/v1")("hi").startswith("Error: no model")


def test_native_tool_calls_plain_and_streamed():
    reply = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "calc", "arguments": "{\"expr\": \"2**10\"}"}},
        {"id": "c2", "type": "function", "function": {"name": "calc", "arguments": "not json"}},
    ]}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    server = FakeServer(reply)
    out = server.llm().chat("go", [BUILTIN_SPECS["calc"]])
    sent = server.requests[0][2]
    assert sent["tools"] == [BUILTIN_SPECS["calc"].openai_schema()] and sent["tool_choice"] == "auto"
    assert out.content == "" and out.finish_reason == "tool_calls"
    assert [(c.name, c.args, c.id, bool(c.error)) for c in out.tool_calls] == [
        ("calc", {"expr": "2**10"}, "c1", False), ("calc", {}, "c2", True)]

    server = FakeServer(stream=[
        chunk({"content": "Let me compute. "}),
        chunk({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "calc", "arguments": ""}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{\"expr\": "}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": "\"3**7\"}"}}]}, finish="tool_calls"),
        "data: [DONE]"])
    out = server.llm().chat("go", [BUILTIN_SPECS["calc"]], on_chunk=lambda t, kind="content": None)
    assert out.content == "Let me compute. "
    assert [(c.name, c.args, c.id) for c in out.tool_calls] == [("calc", {"expr": "3**7"}, "c1")]


def test_detect_provider_by_key_prefix():
    assert detect_provider("sk-or-v1-abc") == "openrouter"
    assert detect_provider("sk-ant-api03-abc") == "anthropic"
    assert detect_provider("gsk_abc") == "groq"
    assert detect_provider("xai-abc") == "xai"
    assert detect_provider("AIzaSyabc") == "gemini"
    assert detect_provider("sk-proj-abc") == "openai"
    assert detect_provider("nvapi-abc") == "nvidia"
    assert detect_provider("sk-abc") is None and detect_provider("") is None


@pytest.fixture
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.endswith("_API_KEY") or key.startswith(("LLM_", "DEEPSEEK_", "ANTHROPIC_")):
            monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_backend_selection_order(clean_env):
    assert choose_backend({}) == "ollama"
    assert choose_backend({"LLM_BACKEND": "Offline"}) == "offline"
    assert choose_backend({"LLM_BASE_URL": "http://x/v1"}) == "openai_compat"
    assert choose_backend({"LLM_API_KEY": "sk-ant-x"}) == "claude"
    assert choose_backend({"LLM_API_KEY": "sk-or-x"}) == "openai_compat"
    assert choose_backend({"OPENROUTER_API_KEY": "k", "DEEPSEEK_API_KEY": "k"}) == "openrouter"
    assert choose_backend({"DEEPSEEK_API_KEY": "k", "OPENAI_API_KEY": "k"}) == "deepseek"
    assert choose_backend({"QWEN_API_KEY": "k"}) == "qwen"


def test_create_llm_builds_the_right_backend(clean_env):
    assert isinstance(create_llm({}), OllamaLLM)
    assert isinstance(create_llm({"LLM_BACKEND": "offline"}), OfflineLLM)

    llm = create_llm({"OPENROUTER_API_KEY": "sk-or-v1-x"})
    assert isinstance(llm, OpenAICompatLLM) and llm.provider == "openrouter"
    assert llm.base_url == "https://openrouter.ai/api/v1" and llm.model == "deepseek/deepseek-v4-flash"

    llm = create_llm({"LLM_API_KEY": "sk-or-v1-x", "LLM_MODEL": "anthropic/claude-haiku-5.5"})
    assert llm.provider == "openrouter" and llm.model == "anthropic/claude-haiku-5.5"

    llm = create_llm({"LLM_API_KEY": "sk-ant-x", "LLM_MODEL": "claude-x"})
    assert isinstance(llm, ClaudeLLM) and llm.api_key == "sk-ant-x" and llm.model == "claude-x"

    llm = create_llm({"LLM_BACKEND": "vllm", "LLM_MODEL": "qwen"})
    assert llm.base_url == "http://localhost:8000/v1" and llm.api_key == ""

    llm = create_llm({"LLM_BACKEND": "openai_compat", "LLM_BASE_URL": "http://h:9/v1/", "LLM_MODEL": "x"})
    assert llm.base_url == "http://h:9/v1" and llm.provider is None

    llm = create_llm({"LLM_BACKEND": "anthropic", "ANTHROPIC_API_KEY": "a",
                      "ANTHROPIC_BASE_URL": "https://proxy.test/v1"})
    assert isinstance(llm, ClaudeLLM) and llm.base_url == "https://proxy.test/v1"

    with pytest.raises(ValueError):
        create_llm({"LLM_BACKEND": "nonsense"})


def test_deepseek_address_rules(clean_env):
    llm = DeepSeekLLM(api_key="sk-plain")
    assert llm.base_url == "https://api.deepseek.com/v1" and llm.request_model == "deepseek-v4-flash"

    llm = DeepSeekLLM(api_key="sk-or-v1-x")            # an OpenRouter key goes to OpenRouter
    assert llm.base_url == "https://openrouter.ai/api/v1"
    assert llm.request_model == "deepseek/deepseek-v4-flash"

    clean_env.setenv("DEEPSEEK_BASE_URL", "https://mirror.test/v1/")
    assert DeepSeekLLM(api_key="sk-or-v1-x").base_url == "https://mirror.test/v1"
    clean_env.delenv("DEEPSEEK_BASE_URL")

    assert DeepSeekLLM(api_key="k", base_url="https://given.test/v1").base_url == "https://given.test/v1"

    llm = DeepSeekLLM("k", "deepseek-v4-flash", 30)          # positional arguments as before
    assert (llm.api_key, llm.model, llm.timeout) == ("k", "deepseek-v4-flash", 30)

    old = DeepSeekLLM.BASE_URL                               # older notebooks patch the class
    try:
        DeepSeekLLM.BASE_URL = "https://openrouter.ai/api/v1"
        llm = DeepSeekLLM(api_key="sk-or-v1-x")
        assert llm.base_url == "https://openrouter.ai/api/v1"
        assert llm.request_model == "deepseek/deepseek-v4-flash"
    finally:
        DeepSeekLLM.BASE_URL = old


def test_deepseek_request_matches_the_old_body():
    server = FakeServer(OK)
    llm = DeepSeekLLM(api_key="k", transport=httpx.MockTransport(server))
    assert llm("hi") == "hello"
    assert server.requests[0][2] == {"model": "deepseek-v4-flash",
                                     "messages": [{"role": "user", "content": "hi"}]}
