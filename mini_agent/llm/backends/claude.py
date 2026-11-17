"""Claude backend"""
from typing import Optional

import httpx
from ..base import BaseLLM, LLMError, LLMResponse
from .claude_stream import stream_claude

# per-model max_output_tokens (measured)
MODEL_MAX_OUTPUT = {
    "claude-opus-4-6": 128000,
    "claude-opus-4": 32000,
    "claude-sonnet-4": 64000,
    "claude-3-7-sonnet": 64000,
    "claude-3-5-sonnet": 8192,
    "claude-3-5-haiku": 8192,
    "claude-3-opus": 4096,
    "claude-3-sonnet": 4096,
    "claude-3-haiku": 4096,
}


def _detect_max_tokens(model: str) -> int:
    for prefix, limit in MODEL_MAX_OUTPUT.items():
        if model.startswith(prefix):
            return limit
    return 16384


class ClaudeLLM(BaseLLM):
    """Anthropic Claude API"""

    BASE_URL = "https://api.anthropic.com/v1"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        timeout: int = 120,
        stream: bool = False,
        max_output_tokens: int = 0,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.stream = stream
        self.max_output_tokens = max_output_tokens or _detect_max_tokens(model)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.context_limit = 200_000
        self._base_url = base_url

    @property
    def base_url(self) -> str:
        """base_url=..., else the class attribute BASE_URL"""
        return (self._base_url or type(self).BASE_URL).rstrip("/")

    def _request(self, prompt: str) -> dict:
        try:
            return self._post(prompt)
        except httpx.HTTPError as e:
            return {"error": {"message": f"{type(e).__name__}: {e}"}}
        except ValueError as e:                       # not JSON
            return {"error": {"message": f"the server did not send JSON ({e})"}}

    def _post(self, prompt: str) -> dict:
        resp = httpx.post(
            f"{self.base_url}/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "max_tokens": self.max_output_tokens,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=self.timeout
        )
        return resp.json()

    def call(self, prompt: str) -> str:
        if self.stream:
            try:
                content, inp, out = stream_claude(
                    self.api_key, self.model, prompt, self.timeout, self.max_output_tokens,
                    url=f"{self.base_url}/messages"
                )
            except httpx.HTTPError as e:
                return LLMError(f"Error: {type(e).__name__}: {e}")
            self.total_prompt_tokens += inp
            self.total_completion_tokens += out
            return content

        data = self._request(prompt)
        if "content" in data:
            if "usage" in data:
                self.total_prompt_tokens += data["usage"].get("input_tokens", 0)
                self.total_completion_tokens += data["usage"].get("output_tokens", 0)
            return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        elif "error" in data:
            return LLMError(f"Error: {data['error'].get('message', data['error'])}")
        return LLMError(f"Error: unexpected reply: {str(data)[:200]}")

    def call_with_usage(self, prompt: str) -> LLMResponse:
        data = self._request(prompt)
        if "content" in data:
            content = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
            usage = data.get("usage", {})
            # Claude uses input_tokens/output_tokens
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            return LLMResponse(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                model=data.get("model", self.model)
            )
        elif "error" in data:
            return LLMResponse(content=LLMError(f"Error: {data['error'].get('message', data['error'])}"))
        return LLMResponse(content=str(data))
