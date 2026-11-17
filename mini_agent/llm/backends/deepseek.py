"""DeepSeek backend (an OpenAI-compatible server)"""
import os
from typing import Optional

from .openai_compat import PRESETS, OpenAICompatLLM

DEFAULT_URL = "https://api.deepseek.com/v1"


class DeepSeekLLM(OpenAICompatLLM):
    """DeepSeek API (OpenAI-compatible)

    The address is, in order: base_url=..., the DEEPSEEK_BASE_URL environment
    variable, then the class attribute BASE_URL (which older notebooks patch).
    An OpenRouter key (sk-or-...) with the default address goes to OpenRouter,
    and the model gets OpenRouter's "deepseek/" prefix.
    """

    BASE_URL = DEFAULT_URL

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-flash",
        timeout: int = 60,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(api_key=api_key, model=model, base_url=base_url, timeout=timeout, **kwargs)
        self.context_limit = 128_000  # DeepSeek supports 128K

    @property
    def base_url(self) -> str:
        if self._base_url:
            return self._base_url.rstrip("/")
        env = os.environ.get("DEEPSEEK_BASE_URL", "").strip()
        if env:
            return env.rstrip("/")
        url = type(self).BASE_URL.rstrip("/")
        if url == DEFAULT_URL and self.api_key.startswith("sk-or-"):
            return PRESETS["openrouter"].base_url
        return url

    @property
    def request_model(self) -> str:
        if "openrouter.ai" in self.base_url and "/" not in self.model:
            return f"deepseek/{self.model}"
        return self.model
