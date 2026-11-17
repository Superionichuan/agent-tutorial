"""LLM base class"""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional

from . import retry


@dataclass
class LLMResponse:
    """LLM response (with token usage)"""
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""


class LLMError(str):
    """A failed model call: the text ("Error: ...") stands in for the reply.

    It is a str, so code that expects text keeps working; the agent loop sees
    the type and stops the run with reason "llm_error" instead of treating the
    error as a malformed reply and asking again.
    """


def mask(value: str) -> str:
    """A key shown safely: its start and last four characters."""
    if len(value) <= 12:
        return "****"
    head = value[:6] if not value.startswith("sk-or-") else value[:9]
    return f"{head}…{value[-4:]}"


def redact(text: str, *secrets: str) -> str:
    """text with every secret replaced by its masked form (a server may echo the key back).
    An LLMError stays an LLMError."""
    out = str(text)
    for secret in secrets:
        if secret and len(secret) >= 8 and secret in out:
            out = out.replace(secret, mask(secret))
    if out == str(text):
        return text
    return LLMError(out) if isinstance(text, LLMError) else out


@dataclass
class ToolCall:
    """One tool call requested by the model in native tool-calling mode.

    args is the parsed argument object; error is set (and args is {}) when the
    model's arguments were not a JSON object.
    """
    name: str
    args: dict = field(default_factory=dict)
    id: str = ""
    error: str = ""


@dataclass
class ChatReply:
    """A reply in native tool-calling mode: text, tool calls, or both.

    error: a transport or provider error ("Error: ..."); nothing else is set then.
    """
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    reasoning: str = ""
    finish_reason: str = ""
    error: str = ""


class BaseLLM(ABC):
    """LLM interface

    call(prompt) -> str is all the default (JSON-in-text) mode needs. Backends
    that support native tool calling also implement
    chat(prompt, tools, on_chunk=None) -> ChatReply and set supports_tools.
    """

    # cumulative token counters
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    # context limit (set by subclass)
    context_limit: int = 128_000

    # native tool calling (tool_mode="native") available
    supports_tools: bool = False

    # waits before trying again after a rate limit (429), a server error (5xx) or a
    # timeout; () = never retry (llm/retry.py)
    retry_waits = retry.WAITS
    _sleep = staticmethod(time.sleep)

    @abstractmethod
    def call(self, prompt: str) -> str:
        pass

    def chat(self, prompt: str, tools: List[Any], on_chunk=None) -> ChatReply:
        """Native tool calling: one user message, the tools offered, one reply."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support native tool calling; use tool_mode='json'")

    def call_with_usage(self, prompt: str) -> LLMResponse:
        """Call and return token usage (subclasses may override)"""
        content = self.call(prompt)
        return LLMResponse(content=content)

    def __call__(self, prompt: str) -> str:
        return self.call(prompt)

    def reset_usage(self):
        """Reset token counters"""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens
