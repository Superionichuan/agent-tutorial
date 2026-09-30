"""LLM factory: create an LLM instance from config"""
import os
from pathlib import Path
from typing import Optional

from .base import BaseLLM
from .backends import OllamaLLM, QwenLLM, DeepSeekLLM, ClaudeLLM
from .offline import OfflineLLM, RecordingLLM


def load_config(env_file: Optional[Path] = None) -> dict:
    """Load config from a .env file"""
    config = {}

    # locate the .env file
    if env_file is None:
        # search order: cwd -> package root
        for path in [Path.cwd() / ".env", Path(__file__).parent.parent.parent / ".env"]:
            if path.exists():
                env_file = path
                break

    # parse the .env file
    if env_file and env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    config[key.strip()] = value.strip().strip('"').strip("'")

    # environment variables take precedence
    for key in ["LLM_BACKEND", "LLM_MODEL", "LLM_STREAM", "LLM_RECORD", "LLM_RECORDING",
                "QWEN_API_KEY", "QWEN_REGION", "QWEN_MODEL",
                "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL",
                "CLAUDE_API_KEY", "CLAUDE_MODEL",
                "OLLAMA_MODEL", "OLLAMA_HOST"]:
        if key in os.environ:
            config[key] = os.environ[key]

    return config


def create_llm(config: Optional[dict] = None) -> BaseLLM:
    """Create an LLM instance from config.

    LLM_BACKEND=offline replays recorded responses (no key, no network).
    LLM_RECORD=<file> records every exchange of a live backend to that file.
    """
    if config is None:
        config = load_config()

    backend = config.get("LLM_BACKEND", "ollama").lower()
    if backend == "offline":
        return OfflineLLM(config.get("LLM_RECORDING") or None)

    llm = _create_live(backend, config)
    if config.get("LLM_RECORD"):
        return RecordingLLM(llm, config["LLM_RECORD"])
    return llm


def _create_live(backend: str, config: dict) -> BaseLLM:
    if backend == "qwen":
        return QwenLLM(
            api_key=config.get("QWEN_API_KEY", ""),
            model=config.get("QWEN_MODEL", config.get("LLM_MODEL", "qwen-plus")),
            region=config.get("QWEN_REGION", "intl")
        )

    elif backend == "deepseek":
        return DeepSeekLLM(
            api_key=config.get("DEEPSEEK_API_KEY", ""),
            model=config.get("DEEPSEEK_MODEL", config.get("LLM_MODEL", "deepseek-v4-flash"))
        )

    elif backend == "claude":
        return ClaudeLLM(
            api_key=config.get("CLAUDE_API_KEY", ""),
            model=config.get("CLAUDE_MODEL", config.get("LLM_MODEL", "claude-sonnet-4-20250514")),
            stream=config.get("LLM_STREAM", "").lower() == "true"
        )

    else:  # ollama (default)
        return OllamaLLM(
            model=config.get("OLLAMA_MODEL", config.get("LLM_MODEL", "qwen3:8b")),
            host=config.get("OLLAMA_HOST", "http://localhost:11434")
        )
