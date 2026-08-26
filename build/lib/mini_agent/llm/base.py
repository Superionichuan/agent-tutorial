"""LLM base class"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """LLM response (with token usage)"""
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""


class BaseLLM(ABC):
    """LLM interface"""

    # cumulative token counters
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    # context limit (set by subclass)
    context_limit: int = 128_000

    @abstractmethod
    def call(self, prompt: str) -> str:
        pass

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
