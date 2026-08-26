"""Qwen backend"""
import httpx
from ..base import BaseLLM, LLMResponse


class QwenLLM(BaseLLM):
    """Qwen API (OpenAI-compatible)"""

    # regional API endpoints
    REGIONS = {
        "intl": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",  # Singapore
        "us": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",      # US
        "cn": "https://dashscope.aliyuncs.com/compatible-mode/v1",         # Beijing
    }

    def __init__(
        self,
        api_key: str,
        model: str = "qwen-plus",
        region: str = "intl",
        timeout: int = 60
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = self.REGIONS.get(region, self.REGIONS["intl"])
        self.timeout = timeout
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        # qwen-long supports 1M context, others 128K
        self.context_limit = 1_000_000 if "long" in model else 128_000

    def _request(self, prompt: str) -> dict:
        """Send the request, return the raw response"""
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
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

    def call(self, prompt: str) -> str:
        data = self._request(prompt)
        if "choices" in data:
            # accumulate tokens
            if "usage" in data:
                self.total_prompt_tokens += data["usage"].get("prompt_tokens", 0)
                self.total_completion_tokens += data["usage"].get("completion_tokens", 0)
            return data["choices"][0]["message"]["content"]
        elif "error" in data:
            return f"Error: {data['error'].get('message', data['error'])}"
        return str(data)

    def call_with_usage(self, prompt: str) -> LLMResponse:
        """Call and return detailed token usage"""
        data = self._request(prompt)

        if "choices" in data:
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            # accumulate
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
