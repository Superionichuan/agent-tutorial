"""LLM module"""
from .base import BaseLLM, ChatReply, ToolCall
from .backends import OllamaLLM, QwenLLM, DeepSeekLLM, ClaudeLLM, OpenAICompatLLM, PRESETS, detect_provider
from .factory import create_llm, load_config, choose_backend
from .offline import OfflineLLM, RecordingLLM


__all__ = [
    "BaseLLM", "ChatReply", "ToolCall",
    "OllamaLLM", "QwenLLM", "DeepSeekLLM", "ClaudeLLM", "OpenAICompatLLM", "PRESETS", "detect_provider",
    "OfflineLLM", "RecordingLLM",
    "create_llm", "load_config", "choose_backend"
]
