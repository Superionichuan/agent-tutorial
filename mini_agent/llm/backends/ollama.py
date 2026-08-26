"""Ollama backend"""
import httpx
from ..base import BaseLLM, LLMResponse


class OllamaLLM(BaseLLM):
    """Local Ollama LLM"""

    def __init__(
        self,
        model: str = "qwen3:8b",
        host: str = "http://localhost:11434",
        timeout: int = 60,
        context_limit: int = 32_000  # locally configurable, default 32K
    ):
        self.model = model
        self.base_url = host
        self.timeout = timeout
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.context_limit = context_limit

    def _request(self, prompt: str) -> dict:
        resp = httpx.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=self.timeout
        )
        return resp.json()

    def call(self, prompt: str) -> str:
        data = self._request(prompt)
        # Ollama uses prompt_eval_count and eval_count
        self.total_prompt_tokens += data.get("prompt_eval_count", 0)
        self.total_completion_tokens += data.get("eval_count", 0)
        return data.get("response", "")

    def call_with_usage(self, prompt: str) -> LLMResponse:
        data = self._request(prompt)
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        return LLMResponse(
            content=data.get("response", ""),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=data.get("model", self.model)
        )
