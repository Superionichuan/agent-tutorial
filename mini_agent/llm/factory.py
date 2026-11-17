"""LLM factory: create an LLM instance from config

Settings come from a .env file (the current directory, then the package root)
and from the environment, which wins. The backend is chosen by, in order:

1. LLM_BACKEND: offline | ollama | qwen | deepseek | claude (or anthropic) |
   openai_compat | a provider preset: openrouter, openai, gemini, groq, xai,
   together, mistral, fireworks, nvidia, lmstudio, vllm
2. LLM_BASE_URL set: openai_compat with that address
3. LLM_API_KEY set: the provider its prefix names (sk-or- OpenRouter,
   sk-ant- Anthropic, gsk_ Groq, ...); LLM_PROVIDER names it when the prefix
   does not
4. the first provider key found: OPENROUTER_API_KEY, ANTHROPIC_API_KEY,
   CLAUDE_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY,
   GROQ_API_KEY, XAI_API_KEY, MISTRAL_API_KEY, TOGETHER_API_KEY,
   FIREWORKS_API_KEY, NVIDIA_API_KEY, QWEN_API_KEY, DASHSCOPE_API_KEY
5. otherwise a local Ollama server (as before)

LLM_MODEL sets the model for any backend.
"""
import os
from pathlib import Path
from typing import Optional

from .base import BaseLLM
from .backends import OllamaLLM, QwenLLM, DeepSeekLLM, ClaudeLLM
from .backends.openai_compat import PRESETS, OpenAICompatLLM, detect_provider
from .offline import OfflineLLM, RecordingLLM

# provider key -> backend, in the order step 4 above tries them
KEY_ORDER = [
    ("OPENROUTER_API_KEY", "openrouter"), ("ANTHROPIC_API_KEY", "claude"),
    ("CLAUDE_API_KEY", "claude"), ("DEEPSEEK_API_KEY", "deepseek"),
    ("OPENAI_API_KEY", "openai"), ("GEMINI_API_KEY", "gemini"), ("GROQ_API_KEY", "groq"),
    ("XAI_API_KEY", "xai"), ("MISTRAL_API_KEY", "mistral"), ("TOGETHER_API_KEY", "together"),
    ("FIREWORKS_API_KEY", "fireworks"), ("NVIDIA_API_KEY", "nvidia"),
    ("QWEN_API_KEY", "qwen"), ("DASHSCOPE_API_KEY", "qwen"),
]

NATIVE_BACKENDS = ("offline", "ollama", "qwen", "deepseek", "claude", "anthropic")
COMPAT_NAMES = ("openai_compat", "openai-compatible", "compat")

CONFIG_KEYS = sorted({
    "LLM_BACKEND", "LLM_MODEL", "LLM_STREAM", "LLM_RECORD", "LLM_RECORDING",
    "LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY",
    "QWEN_REGION", "QWEN_MODEL", "DEEPSEEK_MODEL", "DEEPSEEK_BASE_URL",
    "CLAUDE_MODEL", "ANTHROPIC_BASE_URL", "OLLAMA_MODEL", "OLLAMA_HOST",
    *(k for k, _ in KEY_ORDER), "GOOGLE_API_KEY",
})


def load_config(env_file: Optional[Path] = None) -> dict:
    """Load config from a .env file, then the environment (which wins)"""
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
    for key in CONFIG_KEYS:
        if key in os.environ:
            config[key] = os.environ[key]

    return config


def choose_backend(config: dict) -> str:
    """The backend name the config selects (steps 1-5 in the module docstring)."""
    backend = (config.get("LLM_BACKEND") or "").strip().lower()
    if backend:
        return backend
    if config.get("LLM_BASE_URL"):
        return "openai_compat"
    if config.get("LLM_API_KEY"):
        return "claude" if detect_provider(config["LLM_API_KEY"]) == "anthropic" else "openai_compat"
    for key, name in KEY_ORDER:
        if config.get(key):
            return name
    return "ollama"


def create_llm(config: Optional[dict] = None) -> BaseLLM:
    """Create an LLM instance from config.

    LLM_BACKEND=offline replays recorded responses (no key, no network).
    LLM_RECORD=<file> records every exchange of a live backend to that file.
    """
    if config is None:
        config = load_config()

    backend = choose_backend(config)
    if backend == "offline":
        return OfflineLLM(config.get("LLM_RECORDING") or None)

    llm = _create_live(backend, config)
    if config.get("LLM_RECORD"):
        return RecordingLLM(llm, config["LLM_RECORD"])
    return llm


def _compat(provider: Optional[str], config: dict) -> OpenAICompatLLM:
    preset = PRESETS.get(provider) if provider else None
    key = config.get("LLM_API_KEY") or next(
        (config[k] for k in (preset.key_env if preset else ()) if config.get(k)), "")
    return OpenAICompatLLM(api_key=key, model=config.get("LLM_MODEL", ""),
                           base_url=config.get("LLM_BASE_URL") or None, provider=provider)


def _create_live(backend: str, config: dict) -> BaseLLM:
    model = config.get("LLM_MODEL")
    if backend == "qwen":
        return QwenLLM(
            api_key=config.get("QWEN_API_KEY") or config.get("DASHSCOPE_API_KEY", ""),
            model=config.get("QWEN_MODEL", model or "qwen-plus"),
            region=config.get("QWEN_REGION", "intl")
        )

    if backend == "deepseek":
        return DeepSeekLLM(
            api_key=config.get("DEEPSEEK_API_KEY") or config.get("LLM_API_KEY", ""),
            model=config.get("DEEPSEEK_MODEL", model or "deepseek-v4-flash"),
            base_url=config.get("DEEPSEEK_BASE_URL") or None
        )

    if backend in ("claude", "anthropic"):
        return ClaudeLLM(
            api_key=(config.get("ANTHROPIC_API_KEY") or config.get("CLAUDE_API_KEY")
                     or config.get("LLM_API_KEY", "")),
            model=config.get("CLAUDE_MODEL", model or "claude-sonnet-4-20250514"),
            stream=config.get("LLM_STREAM", "").lower() == "true",
            base_url=config.get("ANTHROPIC_BASE_URL") or None
        )

    if backend == "ollama":
        return OllamaLLM(
            model=config.get("OLLAMA_MODEL", model or "qwen3:8b"),
            host=config.get("OLLAMA_HOST", "http://localhost:11434")
        )

    if backend in COMPAT_NAMES:
        provider = (config.get("LLM_PROVIDER") or "").strip().lower() or None
        if provider is None and not config.get("LLM_BASE_URL"):
            provider = detect_provider(config.get("LLM_API_KEY", ""))
            if provider == "anthropic":
                return _create_live("claude", config)
        return _compat(provider, config)

    if backend in PRESETS:
        return _compat(backend, config)

    known = ", ".join([*NATIVE_BACKENDS, "openai_compat", *[p for p in PRESETS if p != "ollama"]])
    raise ValueError(f"unknown LLM_BACKEND {backend!r}; use one of: {known}")
