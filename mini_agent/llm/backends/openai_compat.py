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
like the other backends, so a run reports them instead of crashing. A rate
limit (429), a server error (5xx) or a timeout is first tried again, up to 3
times with growing waits (llm/retry.py); a key echoed in an error is masked.
"""
import itertools
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import httpx

from .. import retry
from ..base import BaseLLM, ChatReply, LLMError, LLMResponse, ToolCall, redact


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
        return LLMError(f"Error: HTTP {status}: {str(msg)[:300]}")
    except ValueError:
        return LLMError(f"Error: HTTP {status}: {text[:300]}")


@dataclass(frozen=True)
class _Corrupt:
    """A server-sent event whose data is not JSON."""
    text: str


def _bad_usage(usage) -> str:
    """"" when the usage field is absent or well-formed, else what is wrong with it.
    Token counts feed the spending budget, so a count that is not a number is an error."""
    if usage is None:
        return ""
    if not isinstance(usage, dict):
        return "usage is not an object"
    for name in ("prompt_tokens", "completion_tokens"):
        value = usage.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            return f"usage.{name} is not a whole number"
    return ""


def _bad_calls(calls) -> str:
    """"" when the tool_calls field is absent or well-formed, else what is wrong with it."""
    if calls is None:
        return ""
    if not isinstance(calls, list):
        return "tool_calls is not a list"
    for call in calls:
        if not isinstance(call, dict):
            return "a tool call is not an object"
        fn = call.get("function")
        if fn is not None and not isinstance(fn, dict):
            return "a tool call's function is not an object"
        if isinstance(fn, dict) and fn.get("name") is not None and not isinstance(fn["name"], str):
            return "a tool call's name is not text"
    return ""


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
            return LLMError("Error: no server address: set LLM_BASE_URL (or base_url=...) or "
                            "choose a provider (LLM_PROVIDER / provider=...)")
        if not self.request_model:
            return LLMError("Error: no model: set LLM_MODEL (or model=...)")
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

    def _failed(self, error) -> LLMError:
        """The error as an LLMError, with the key masked (a gateway may echo it back)."""
        return redact(error if isinstance(error, LLMError) else LLMError(error), self.api_key)

    def _post(self, body: dict):
        """(response data, "") or (None, "Error: ...")"""
        problem = self._problem()
        if problem:
            return None, problem
        try:
            with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
                # stream=True, then read: a timeout while reading a final error's text must
                # not turn that error into a retried timeout (retry.complete)
                resp = retry.send(lambda: retry.complete(client.send(client.build_request(
                    "POST", f"{self.base_url}/chat/completions", headers=self._headers(),
                    json=body), stream=True)), self.retry_waits, self._sleep)
        except httpx.HTTPError as e:
            return None, self._failed(f"Error: {type(e).__name__}: {e}")
        if resp.status_code >= 400:
            return None, self._failed(_error(resp.status_code, resp.text))
        try:
            data = resp.json()
        except ValueError:
            return None, self._failed(f"Error: the server did not send JSON: {resp.text[:200]}")
        return self._checked(data, resp.status_code)

    def _checked(self, data, status: int = 200):
        """(data, "") for a well-formed completion, else (None, "Error: ...")"""
        if isinstance(data, dict) and "error" in data:
            return None, self._failed(_error(status, json.dumps(data)))
        choices = data.get("choices") if isinstance(data, dict) else None
        choice = choices[0] if isinstance(choices, list) and choices else None
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            return None, self._failed(f"Error: unexpected reply: {str(data)[:200]}")
        bad = _bad_usage(data.get("usage")) or _bad_calls(choice["message"].get("tool_calls"))
        if bad:
            return None, self._failed(f"Error: unexpected reply ({bad}): {str(data)[:200]}")
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
        events = 0
        other: List[str] = []          # lines that are not server-sent events: an error page?

        def unexpected(data, why: str = "") -> str:
            return self._failed(f"Error: unexpected reply from the server"
                                f"{f' ({why})' if why else ''}: {str(data)[:200]}")

        def handle(data) -> str:
            """Apply one event; returns "" or the error that ends the reply."""
            nonlocal used
            if isinstance(data, _Corrupt):
                return self._failed(f"Error: the server sent an event that is not JSON: {data.text[:200]}")
            if not isinstance(data, dict):
                return unexpected(data)
            if "error" in data:
                return self._failed(_error(200, json.dumps(data)))
            bad = _bad_usage(data.get("usage"))
            if bad:
                return unexpected(data, bad)
            if data.get("usage"):
                used = self._count(data["usage"])
            choices = data.get("choices") or []
            if not isinstance(choices, list):
                return unexpected(data)
            for choice in choices:
                delta = (choice.get("delta") or {}) if isinstance(choice, dict) else None
                if not isinstance(delta, dict):
                    return unexpected(data)
                bad = _bad_calls(delta.get("tool_calls"))
                if bad:
                    return unexpected(data, bad)
                why = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(why, str) and why:
                    thinking.append(why)
                    if on_chunk:
                        _emit(on_chunk, why, "reasoning")
                text = delta.get("content")
                if isinstance(text, str) and text:
                    parts.append(text)
                    if on_chunk:
                        _emit(on_chunk, text, "content")
                for tc in delta.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        return unexpected(data)
                    slot = calls.setdefault(tc.get("index", len(calls)),
                                            {"id": "", "name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    name, new_id = fn.get("name") or "", tc.get("id") or ""
                    # The name arrives in pieces to be joined. A server that repeats the
                    # whole header (same id, same full name) must not double it.
                    if name and not (new_id and new_id == slot["id"] and name == slot["name"]):
                        slot["name"] += name
                    slot["id"] = new_id or slot["id"]
                    if fn.get("arguments"):
                        slot["arguments"] += (fn["arguments"] if isinstance(fn["arguments"], str)
                                              else json.dumps(fn["arguments"]))
                if choice.get("finish_reason"):
                    reply.finish_reason = choice["finish_reason"]
            return ""

        def payloads(lines: List[str]) -> list:
            """The events in one block of data: lines. One event may span several lines;
            servers that leave out the blank line between events give one per line."""
            joined = "\n".join(lines).strip()
            if not joined:
                return []
            if joined == "[DONE]":
                return ["[DONE]"]
            try:
                return [json.loads(joined)]
            except ValueError:
                pass
            out = []
            for line in lines:
                if line.strip() == "[DONE]":
                    out.append("[DONE]")
                    break
                if not line.strip():
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    out.append(_Corrupt(line))   # never skipped: the reply would be silently cut
                    break
            return out

        def read(resp) -> str:
            """Read the events of one response; "" or the error that ends the reply."""
            nonlocal events
            block: List[str] = []
            for line in itertools.chain(resp.iter_lines(), [""]):   # "" ends the last event
                if line.startswith("data:"):
                    block.append(line[5:])
                    continue
                if line and not line.startswith((":", "event:", "id:", "retry:")):
                    other.append(line)                  # not part of a server-sent event
                if line or not block:
                    continue
                for data in payloads(block):
                    if data == "[DONE]":
                        return ""
                    events += 1
                    error = handle(data)
                    if error:
                        return error
                block = []
            return ""

        try:
            with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
                for attempt in range(len(self.retry_waits) + 1):
                    last = attempt == len(self.retry_waits)
                    other.clear()
                    events = 0
                    try:
                        with client.stream("POST", f"{self.base_url}/chat/completions",
                                           headers=self._headers(), json=body) as resp:
                            if resp.status_code == 200:
                                error = read(resp)
                                if error:
                                    reply.error = error
                                    return reply, used
                                break
                            # a timeout while reading a final error's text is not retried
                            text = retry.complete(resp).content.decode("utf-8", "replace")
                            if last or not retry.retryable(resp.status_code):
                                reply.error = self._failed(_error(resp.status_code, text))
                                return reply, used
                            wait = retry.wait_before(attempt, resp, self.retry_waits)
                    except httpx.TimeoutException:
                        # part of the reply reached the caller: do not repeat it. Events that
                        # carried nothing (only the role, say) do not count.
                        if last or parts or thinking or calls:
                            raise
                        wait = retry.wait_before(attempt, None, self.retry_waits)
                    self._sleep(wait)
        except httpx.HTTPError as e:
            reply.error = self._failed(f"Error: {type(e).__name__}: {e}")
            return reply, used
        if not events:
            # no events at all: an error page, or a whole (non-streamed) reply
            text = "\n".join(other).strip()
            try:
                data = json.loads(text) if text else None
            except ValueError:
                data = None
            if data is None:
                reply.error = self._failed(
                    f"Error: the server did not send a stream: {text[:200]}" if text
                    else "Error: the server sent an empty reply")
                return reply, used
            data, error = self._checked(data)
            if error:
                reply.error = error
                return reply, used
            used = self._count(data.get("usage"))
            whole = self._reply(data)
            for kind, value in (("reasoning", whole.reasoning), ("content", whole.content)):
                if value and on_chunk:
                    _emit(on_chunk, value, kind)
            return whole, used
        reply.content = "".join(parts)
        reply.reasoning = "".join(thinking)
        reply.tool_calls = [self._tool_call(calls[i]["name"], calls[i]["arguments"], calls[i]["id"])
                            for i in sorted(calls, key=lambda i: (0, i) if isinstance(i, int) else (1, str(i)))]
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
            return LLMResponse(content=error)   # an LLMError
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
