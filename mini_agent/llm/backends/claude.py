"""Claude backend"""
from typing import Optional

import httpx
from .. import retry
from ..base import BaseLLM, LLMError, LLMResponse, redact
from .claude_stream import error_text, stream_claude

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
        resp = retry.send(lambda: httpx.post(
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
        ), self.retry_waits, self._sleep)
        if resp.status_code >= 400:     # "HTTP <status>: <message>", as in the streamed call
            return {"error": {"message": error_text(resp.status_code, resp.text).removeprefix("Error: ")}}
        data = resp.json()
        return data if isinstance(data, dict) else {"unexpected": data}

    def _failed(self, text: str) -> LLMError:
        """The error as an LLMError, with the key masked (a gateway may echo it back)."""
        return redact(LLMError(text), self.api_key)

    @staticmethod
    def _bad_count(*counts) -> bool:
        """True when a token count is not a whole number of zero or more.
        The counts feed the spending budget, so a malformed one is an error, not a zero."""
        return any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in counts)

    @staticmethod
    def _message(error) -> str:
        return str(error.get("message", error)) if isinstance(error, dict) else str(error)

    def call(self, prompt: str) -> str:
        if self.stream:
            try:
                content, inp, out, error = stream_claude(
                    self.api_key, self.model, prompt, self.timeout, self.max_output_tokens,
                    url=f"{self.base_url}/messages", waits=self.retry_waits, sleep=self._sleep
                )
            except httpx.HTTPError as e:
                return self._failed(f"Error: {type(e).__name__}: {e}")
            if self._bad_count(inp, out):
                return self._failed("Error: unexpected reply: the token counts are not whole numbers")
            self.total_prompt_tokens += inp
            self.total_completion_tokens += out
            return self._failed(error) if error else content

        data = self._request(prompt)
        if isinstance(data.get("content"), list):
            usage = data.get("usage") or {}
            if not isinstance(usage, dict):
                return self._failed("Error: unexpected reply: usage is not an object")
            inp, out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
            if self._bad_count(inp, out):
                return self._failed("Error: unexpected reply: the token counts are not whole numbers")
            self.total_prompt_tokens += inp
            self.total_completion_tokens += out
            return "".join(b.get("text", "") for b in data["content"]
                           if isinstance(b, dict) and b.get("type") == "text")
        elif "error" in data:
            return self._failed(f"Error: {self._message(data['error'])}")
        return self._failed(f"Error: unexpected reply: {str(data)[:200]}")

    def call_with_usage(self, prompt: str) -> LLMResponse:
        data = self._request(prompt)
        if isinstance(data.get("content"), list):
            content = "".join(b.get("text", "") for b in data["content"]
                              if isinstance(b, dict) and b.get("type") == "text")
            usage = data.get("usage") or {}
            # Claude uses input_tokens/output_tokens
            prompt_tokens = usage.get("input_tokens", 0) if isinstance(usage, dict) else None
            completion_tokens = usage.get("output_tokens", 0) if isinstance(usage, dict) else None
            if self._bad_count(prompt_tokens, completion_tokens):
                return LLMResponse(content=self._failed(
                    "Error: unexpected reply: the token counts are not whole numbers"))
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
            return LLMResponse(content=self._failed(f"Error: {self._message(data['error'])}"))
        return LLMResponse(content=self._failed(f"Error: unexpected reply: {str(data)[:200]}"))
