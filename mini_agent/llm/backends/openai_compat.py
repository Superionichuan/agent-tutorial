"""Any OpenAI-compatible chat-completions server

One class covers hosted services (OpenRouter, DeepSeek, OpenAI, Gemini's
OpenAI endpoint, Groq, xAI, Together, Mistral, Fireworks, NVIDIA, DashScope)
and local servers (Ollama's /v1, LM Studio, vLLM). A provider preset fills in
the address, the environment variable that holds the key, and a default model:

    OpenAICompatLLM(provider="openrouter", model="deepseek/deepseek-v4-flash")
    OpenAICompatLLM(base_url="http://localhost:8000/v1", model="my-model")   # vLLM

detect_provider(key) recognises a key by its prefix: sk-or- OpenRouter,
sk-ant- Anthropic (served by the native Claude backend), gsk_ Groq, xai- xAI,
AIza Gemini, sk-proj- / sk-svcacct- OpenAI, nvapi- NVIDIA. A plain "sk-" key
is ambiguous (OpenAI, DeepSeek and DashScope all issue them) and is not guessed.

Errors (network, HTTP, provider) come back as text starting with "Error:",
like the other backends, so a run reports them instead of crashing.
"""
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import httpx

from ..base import BaseLLM, ChatReply, LLMResponse, ToolCall


@dataclass(frozen=True)
class Preset:
    base_url: str
    key_env: Tuple[str, ...] = ()
    model: str = ""                      # default model ("" = must be given)
    key_prefixes: Tuple[str, ...] = ()
    local: bool = False                  # runs on this machine; no key needed


PRESETS: Dict[str, Preset] = {
    "openrouter": Preset("https://openrouter.ai/api/v1", ("OPENROUTER_API_KEY",),
                         "deepseek/deepseek-v4-flash", ("sk-or-",)),
    "deepseek": Preset("https://api.deepseek.com/v1", ("DEEPSEEK_API_KEY",), "deepseek-v4-flash"),
    "openai": Preset("https://api.openai.com/v1", ("OPENAI_API_KEY",), "",
                     ("sk-proj-", "sk-svcacct-")),
    "gemini": Preset("https://generativelanguage.googleapis.com/v1beta/openai",
                     ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "", ("AIza",)),
    "groq": Preset("https://api.groq.com/openai/v1", ("GROQ_API_KEY",), "", ("gsk_",)),
    "xai": Preset("https://api.x.ai/v1", ("XAI_API_KEY",), "", ("xai-",)),
    "together": Preset("https://api.together.xyz/v1", ("TOGETHER_API_KEY",)),
    "mistral": Preset("https://api.mistral.ai/v1", ("MISTRAL_API_KEY",)),
    "fireworks": Preset("https://api.fireworks.ai/inference/v1", ("FIREWORKS_API_KEY",)),
    "nvidia": Preset("https://integrate.api.nvidia.com/v1", ("NVIDIA_API_KEY",), "", ("nvapi-",)),
    "qwen": Preset("https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                   ("DASHSCOPE_API_KEY", "QWEN_API_KEY"), "qwen-plus"),
    "ollama": Preset("http://localhost:11434/v1", (), "qwen3:8b", local=True),
    "lmstudio": Preset("http://localhost:1234/v1", local=True),
    "vllm": Preset("http://localhost:8000/v1", local=True),
}

ANTHROPIC_PREFIX = "sk-ant-"


def detect_provider(api_key: str) -> Optional[str]:
    """"anthropic", a preset name, or None when the prefix is unknown or ambiguous."""
    if not api_key:
        return None
    if api_key.startswith(ANTHROPIC_PREFIX):
        return "anthropic"
    for name, preset in PRESETS.items():
        if any(api_key.startswith(p) for p in preset.key_prefixes):
            return name
    return None


def _error(status: int, text: str) -> str:
    """Provider error text: the JSON error message if there is one."""
    try:
        data = json.loads(text)
        err = data.get("error", data) if isinstance(data, dict) else data
        msg = err.get("message", err) if isinstance(err, dict) else err
        return f"Error: HTTP {status}: {str(msg)[:300]}"
    except ValueError:
        return f"Error: HTTP {status}: {text[:300]}"


def _emit(on_chunk, text: str, kind: str):
    try:
        on_chunk(text, kind=kind)
    except TypeError:                     # callbacks written as on_chunk(text)
        on_chunk(text)


def _parse_args(raw) -> Tuple[dict, str]:
    """(arguments, error) from a tool call's arguments field."""
    if isinstance(raw, dict):
        return raw, ""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, ""
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}, f"the tool arguments are not valid JSON: {str(raw)[:120]}"
    if not isinstance(value, dict):
        return {}, "the tool arguments must be a JSON object of named values"
    return value, ""


class OpenAICompatLLM(BaseLLM):
    """An OpenAI-compatible /chat/completions endpoint."""

    BASE_URL = ""                 # a fixed default address, for subclasses (DeepSeekLLM)
    supports_tools = True

    def __init__(self, api_key: str = "", model: str = "", base_url: Optional[str] = None,
                 provider: Optional[str] = None, timeout: float = 120,
                 headers: Optional[dict] = None, extra_body: Optional[dict] = None,
                 context_limit: int = 128_000, transport: Optional[httpx.BaseTransport] = None):
        if provider is not None and provider not in PRESETS:
            raise ValueError(f"unknown provider {provider!r}; known: {', '.join(PRESETS)}")
        preset = PRESETS.get(provider) if provider else None
        if not api_key and preset:
            api_key = next((os.environ[k] for k in preset.key_env if os.environ.get(k)), "")
        self.provider = provider
        self.api_key = api_key or ""
        self.model = model or (preset.model if preset else "")
        self._base_url = base_url or (preset.base_url if preset else None)
        self.timeout = timeout
        self.headers = dict(headers or {})
        self.extra_body = dict(extra_body or {})
        self.context_limit = context_limit
        self._transport = transport           # tests pass an httpx.MockTransport
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    # ---- where and what to send ----

    @property
    def base_url(self) -> str:
        return (self._base_url or self.BASE_URL or "").rstrip("/")

    @property
    def request_model(self) -> str:
        """The model name sent to the server."""
        return self.model

    def _problem(self) -> str:
        if not self.base_url:
            return ("Error: no server address: set LLM_BASE_URL (or base_url=...) or choose a "
                    "provider (LLM_PROVIDER / provider=...)")
        if not self.request_model:
            return "Error: no model: set LLM_MODEL (or model=...)"
        return ""

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _body(self, prompt: str, tools=None, stream: bool = False) -> dict:
        body = {"model": self.request_model, "messages": [{"role": "user", "content": prompt}]}
        if tools:
            body["tools"] = [t.openai_schema() if hasattr(t, "openai_schema") else t for t in tools]
            body["tool_choice"] = "auto"
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        body.update(self.extra_body)
        return body

    def _count(self, usage) -> Tuple[int, int]:
        if not isinstance(usage, dict):
            return 0, 0
        p, c = usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0
        self.total_prompt_tokens += p
        self.total_completion_tokens += c
        return p, c

    # ---- transport ----

    def _post(self, body: dict):
        """(response data, "") or (None, "Error: ...")"""
        problem = self._problem()
        if problem:
            return None, problem
        try:
            with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
                resp = client.post(f"{self.base_url}/chat/completions",
                                   headers=self._headers(), json=body)
        except httpx.HTTPError as e:
            return None, f"Error: {type(e).__name__}: {e}"
        if resp.status_code >= 400:
            return None, _error(resp.status_code, resp.text)
        try:
            data = resp.json()
        except ValueError:
            return None, f"Error: the server did not send JSON: {resp.text[:200]}"
        if isinstance(data, dict) and "error" in data:
            return None, _error(resp.status_code, json.dumps(data))
        if not isinstance(data, dict) or not data.get("choices"):
            return None, f"Error: unexpected reply: {str(data)[:200]}"
        return data, ""

    def _stream(self, body: dict, on_chunk) -> Tuple[ChatReply, Tuple[int, int]]:
        reply = ChatReply()
        problem = self._problem()
        if problem:
            reply.error = problem
            return reply, (0, 0)
        parts: List[str] = []
        thinking: List[str] = []
        calls: Dict[int, dict] = {}
        used = (0, 0)
        try:
            with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
                with client.stream("POST", f"{self.base_url}/chat/completions",
                                   headers=self._headers(), json=body) as resp:
                    if resp.status_code != 200:
                        reply.error = _error(resp.status_code, resp.read().decode("utf-8", "replace"))
                        return reply, used
                    for line in resp.iter_lines():
                        if not line.startswith("data:"):
                            continue                     # SSE comments, keep-alives
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            data = json.loads(payload)
                        except ValueError:
                            continue
                        if "error" in data:
                            reply.error = _error(200, json.dumps(data))
                            return reply, used
                        if data.get("usage"):
                            used = self._count(data["usage"])
                        for choice in data.get("choices") or []:
                            delta = choice.get("delta") or {}
                            why = delta.get("reasoning_content") or delta.get("reasoning")
                            if why:
                                thinking.append(why)
                                if on_chunk:
                                    _emit(on_chunk, why, "reasoning")
                            text = delta.get("content")
                            if text:
                                parts.append(text)
                                if on_chunk:
                                    _emit(on_chunk, text, "content")
                            for tc in delta.get("tool_calls") or []:
                                slot = calls.setdefault(tc.get("index", len(calls)),
                                                        {"id": "", "name": "", "arguments": ""})
                                slot["id"] = tc.get("id") or slot["id"]
                                fn = tc.get("function") or {}
                                if fn.get("name") and fn["name"] != slot["name"]:
                                    slot["name"] = (fn["name"] if not slot["name"]
                                                    else slot["name"] + fn["name"])
                                if fn.get("arguments"):
                                    slot["arguments"] += (fn["arguments"] if isinstance(fn["arguments"], str)
                                                          else json.dumps(fn["arguments"]))
                            if choice.get("finish_reason"):
                                reply.finish_reason = choice["finish_reason"]
        except httpx.HTTPError as e:
            reply.error = f"Error: {type(e).__name__}: {e}"
            return reply, used
        reply.content = "".join(parts)
        reply.reasoning = "".join(thinking)
        reply.tool_calls = [self._tool_call(calls[i]["name"], calls[i]["arguments"], calls[i]["id"])
                            for i in sorted(calls)]
        return reply, used

    @staticmethod
    def _tool_call(name, arguments, call_id="") -> ToolCall:
        args, error = _parse_args(arguments)
        return ToolCall(name=name or "", args=args, id=call_id or "", error=error)

    def _reply(self, data: dict) -> ChatReply:
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        calls = [self._tool_call((c.get("function") or {}).get("name"),
                                 (c.get("function") or {}).get("arguments"), c.get("id"))
                 for c in msg.get("tool_calls") or []]
        return ChatReply(content=msg.get("content") or "", tool_calls=calls,
                         reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
                         finish_reason=choice.get("finish_reason") or "")

    # ---- the BaseLLM interface ----

    def call(self, prompt: str) -> str:
        data, error = self._post(self._body(prompt))
        if error:
            return error
        self._count(data.get("usage"))
        return self._reply(data).content

    def call_stream(self, prompt: str, on_chunk) -> str:
        """Streaming call: on_chunk(text, kind) per delta (kind "content" or "reasoning")."""
        reply, _ = self._stream(self._body(prompt, stream=True), on_chunk)
        return reply.error or reply.content

    def call_with_usage(self, prompt: str) -> LLMResponse:
        data, error = self._post(self._body(prompt))
        if error:
            return LLMResponse(content=error)
        p, c = self._count(data.get("usage"))
        return LLMResponse(content=self._reply(data).content, prompt_tokens=p, completion_tokens=c,
                           total_tokens=p + c, model=data.get("model", self.request_model))

    def chat(self, prompt: str, tools, on_chunk=None) -> ChatReply:
        """Native tool calling: the prompt as one user message, the tools offered."""
        if on_chunk is not None:
            reply, _ = self._stream(self._body(prompt, tools=tools, stream=True), on_chunk)
            return reply
        data, error = self._post(self._body(prompt, tools=tools))
        if error:
            return ChatReply(error=error)
        self._count(data.get("usage"))
        return self._reply(data)
