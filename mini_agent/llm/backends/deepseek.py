"""DeepSeek backend"""
import httpx
from ..base import BaseLLM, LLMResponse


class DeepSeekLLM(BaseLLM):
    """DeepSeek API (OpenAI-compatible)"""

    BASE_URL = "https://api.deepseek.com/v1"

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-flash",
        timeout: int = 60
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.context_limit = 128_000  # DeepSeek supports 128K

    def _request(self, prompt: str) -> dict:
        resp = httpx.post(
            f"{self.BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=self.timeout
        )
        return resp.json()

    def call_stream(self, prompt: str, on_chunk) -> str:
        """Streaming call: on_chunk(text) per delta; returns the full text.

        SSE protocol; usage arrives in the final chunk (stream_options.include_usage).
        """
        import json as _json
        parts = []
        with httpx.stream(
            "POST",
            f"{self.BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "stream_options": {"include_usage": True}
            },
            timeout=self.timeout
        ) as resp:
            if resp.status_code != 200:
                body = resp.read().decode("utf-8", "replace")
                return f"Error: HTTP {resp.status_code} {body[:200]}"
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                data = _json.loads(payload)
                usage = data.get("usage")
                if usage:
                    self.total_prompt_tokens += usage.get("prompt_tokens", 0)
                    self.total_completion_tokens += usage.get("completion_tokens", 0)
                for ch in data.get("choices", []):
                    delta = ch.get("delta", {})
                    reasoning = delta.get("reasoning_content")
                    if reasoning:
                        # deep-think reasoning chain: streamed for the UI, never enters the answer text
                        try:
                            on_chunk(reasoning, kind="reasoning")
                        except TypeError:
                            on_chunk(reasoning)
                    text = delta.get("content")
                    if text:
                        parts.append(text)
                        try:
                            on_chunk(text, kind="content")
                        except TypeError:
                            on_chunk(text)
        return "".join(parts)

    def call(self, prompt: str) -> str:
        data = self._request(prompt)
        if "choices" in data:
            if "usage" in data:
                self.total_prompt_tokens += data["usage"].get("prompt_tokens", 0)
                self.total_completion_tokens += data["usage"].get("completion_tokens", 0)
            return data["choices"][0]["message"]["content"]
        elif "error" in data:
            return f"Error: {data['error'].get('message', data['error'])}"
        return str(data)

    def call_with_usage(self, prompt: str) -> LLMResponse:
        data = self._request(prompt)
        if "choices" in data:
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
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
            return LLMResponse(content=f"Error: {data['error'].get('message', data['error'])}")
        return LLMResponse(content=str(data))
