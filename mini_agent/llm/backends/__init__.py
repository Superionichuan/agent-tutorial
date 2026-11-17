"""LLM backends"""
from .ollama import OllamaLLM
from .qwen import QwenLLM
from .deepseek import DeepSeekLLM
from .claude import ClaudeLLM
from .openai_compat import OpenAICompatLLM, PRESETS, detect_provider

__all__ = ["OllamaLLM", "QwenLLM", "DeepSeekLLM", "ClaudeLLM",
           "OpenAICompatLLM", "PRESETS", "detect_provider"]
