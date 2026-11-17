"""Qwen backend (DashScope's OpenAI-compatible server)"""
from .openai_compat import OpenAICompatLLM


class QwenLLM(OpenAICompatLLM):
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
        timeout: int = 60,
        **kwargs,
    ):
        super().__init__(api_key=api_key, model=model,
                         base_url=self.REGIONS.get(region, self.REGIONS["intl"]),
                         timeout=timeout, **kwargs)
        self.region = region
        # qwen-long supports 1M context, others 128K
        self.context_limit = 1_000_000 if "long" in model else 128_000
