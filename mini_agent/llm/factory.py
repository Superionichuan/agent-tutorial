"""LLM factory: create an LLM instance from config

Each setting takes its value from the first of these that has it:
  1. command-line flags (--backend, --model)
  2. environment variables
  3. a .env file in the current directory
  4. ~/.mini_agent/config.env, written by `mini_agent setup` ($MINI_AGENT_HOME)
  5. a .env file next to the installed package. Older notebooks write one; it is
     ignored whenever 1-4 configure a model, so it never overrides your choice.

The backend is chosen by, in order:

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

from .. import paths
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


# the model setting of one backend; LLM_MODEL set in a higher layer overrides it
MODEL_KEYS = ("DEEPSEEK_MODEL", "QWEN_MODEL", "CLAUDE_MODEL", "OLLAMA_MODEL")

# settings that pick the model or the server: a layer that has any of them "configures a model"
SELECT_KEYS = frozenset({"LLM_BACKEND", "LLM_BASE_URL", "LLM_API_KEY", "LLM_PROVIDER", "LLM_MODEL",
                         "OLLAMA_HOST", "DEEPSEEK_BASE_URL", "ANTHROPIC_BASE_URL", "GOOGLE_API_KEY",
                         *MODEL_KEYS, *(k for k, _ in KEY_ORDER)})

ANTHROPIC_NAMES = ("anthropic", "claude")

PACKAGE_ENV = Path(__file__).parent.parent.parent / ".env"


def read_env_file(path: Path) -> dict:
    """KEY=value lines of a .env file (comments and blank lines skipped). Bytes that are
    not UTF-8 are replaced (U+FFFD), so a damaged file is reported, not fatal."""
    values = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def config_layers(env_file: Optional[Path] = None) -> list:
    """[(name, path or None, {key: value})], highest priority first; the package .env is
    left out when a higher layer already configures a model."""
    env = {k: os.environ[k] for k in CONFIG_KEYS if k in os.environ}
    layers = [("environment", None, env)]
    if env_file is not None:
        files = [("file", Path(env_file))]
    else:
        files = [("./.env", Path.cwd() / ".env"), ("user config", paths.config_file())]
    for name, path in files:
        try:
            if path.is_file():
                layers.append((name, path, read_env_file(path)))
        except OSError:             # unreadable: as if absent (doctor reports it)
            pass
    if env_file is None and PACKAGE_ENV.is_file() and PACKAGE_ENV.resolve() != (Path.cwd() / ".env").resolve():
        chosen = any(v for _, _, values in layers for k, v in values.items() if k in SELECT_KEYS)
        if not chosen:
            try:
                layers.append(("package .env", PACKAGE_ENV, read_env_file(PACKAGE_ENV)))
            except OSError:
                pass
    return layers


def load_config(env_file: Optional[Path] = None) -> dict:
    """The settings, each from the highest layer that has it (see the module docstring).

    LLM_MODEL from a higher layer than a backend's own model setting (DEEPSEEK_MODEL,
    ...) wins over it; in the same layer the backend's own setting wins.
    env_file: read only this file (plus the environment) instead of the usual files.
    """
    layers = config_layers(env_file)
    config = {}
    for _, _, values in reversed(layers):
        config.update(values)

    def rank(key):
        return next((i for i, (_, _, values) in enumerate(layers) if values.get(key)), None)

    generic = rank("LLM_MODEL")
    if generic is not None:
        for key in MODEL_KEYS:
            own = rank(key)
            if own is not None and generic < own:
                del config[key]
    return config


def choose_backend(config: dict) -> str:
    """The backend name the config selects (steps 1-5 in the module docstring)."""
    backend = (config.get("LLM_BACKEND") or "").strip().lower()
    if backend:
        return backend
    if config.get("LLM_BASE_URL"):
        return "openai_compat"
    if config.get("LLM_API_KEY"):
        return "claude" if _key_provider(config) == "anthropic" else "openai_compat"
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


def _key_provider(config: dict) -> Optional[str]:
    """Where LLM_API_KEY goes: the provider its prefix names; else LLM_PROVIDER
    ("anthropic" for anthropic / claude); else None. A recognised prefix wins over
    LLM_PROVIDER, so a key is never sent to a provider that did not issue it."""
    detected = detect_provider(config.get("LLM_API_KEY", ""))
    if detected:
        return detected
    hint = (config.get("LLM_PROVIDER") or "").strip().lower() or None
    return "anthropic" if hint in ANTHROPIC_NAMES else hint


# names in messages; the key variable to suggest for each target
LABELS = {"anthropic": "Anthropic", "openrouter": "OpenRouter", "openai": "OpenAI", "deepseek": "DeepSeek",
          "gemini": "Gemini", "groq": "Groq", "xai": "xAI", "nvidia": "NVIDIA", "qwen": "DashScope"}


def _generic_key(config: dict, target: str, address_given: bool) -> str:
    """LLM_API_KEY for the provider `target` ("anthropic" or a preset name).

    A key whose prefix names another provider is never sent to `target`: that would hand one
    provider's credential to another. Raises ValueError instead, unless the server address was
    set explicitly (LLM_BASE_URL, DEEPSEEK_BASE_URL, ANTHROPIC_BASE_URL): then the key goes
    where it was pointed (a gateway may accept other providers' keys).
    """
    key = config.get("LLM_API_KEY", "")
    issuer = detect_provider(key)
    if not issuer or issuer == target or address_given:
        return key
    if target != "anthropic" and PRESETS[target].local:
        return ""                     # a server on this machine needs no key: send none
    prefix ="sk-ant-" if issuer == "anthropic" else next(
        p for p in PRESETS[issuer].key_prefixes if key.startswith(p))
    variable = "ANTHROPIC_API_KEY" if target == "anthropic" else PRESETS[target].key_env[0]
    raise ValueError(
        f"LLM_API_KEY starts with {prefix} ({LABELS.get(issuer, issuer)}), so it is not sent to "
        f"{LABELS.get(target, target)}. Use LLM_BACKEND={'claude' if issuer == 'anthropic' else issuer}, "
        f"or set {variable} to your {LABELS.get(target, target)} key.")


def _compat(provider: Optional[str], config: dict) -> OpenAICompatLLM:
    preset = PRESETS.get(provider) if provider else None
    url = config.get("LLM_BASE_URL") or None
    specific = next((config[k] for k in (preset.key_env if preset else ()) if config.get(k)), "")
    generic = config.get("LLM_API_KEY", "")
    if generic and provider and not url and detect_provider(generic) not in (None, provider):
        key = specific or _generic_key(config, provider, False)     # raises without a specific key
    else:
        key = generic or specific
    return OpenAICompatLLM(api_key=key, model=config.get("LLM_MODEL", ""),
                           base_url=url, provider=provider)


def _claude(config: dict, base_url: Optional[str]) -> ClaudeLLM:
    return ClaudeLLM(
        api_key=(config.get("ANTHROPIC_API_KEY") or config.get("CLAUDE_API_KEY")
                 or _generic_key(config, "anthropic", bool(base_url))),
        model=config.get("CLAUDE_MODEL", config.get("LLM_MODEL") or "claude-sonnet-4-20250514"),
        stream=config.get("LLM_STREAM", "").lower() == "true",
        base_url=base_url,
    )


def _create_live(backend: str, config: dict) -> BaseLLM:
    model = config.get("LLM_MODEL")
    if backend == "qwen":
        return QwenLLM(
            api_key=config.get("QWEN_API_KEY") or config.get("DASHSCOPE_API_KEY", ""),
            model=config.get("QWEN_MODEL", model or "qwen-plus"),
            region=config.get("QWEN_REGION", "intl")
        )

    if backend == "deepseek":
        url = config.get("DEEPSEEK_BASE_URL") or None
        return DeepSeekLLM(
            api_key=config.get("DEEPSEEK_API_KEY") or _generic_key(config, "deepseek", bool(url)),
            model=config.get("DEEPSEEK_MODEL", model or "deepseek-v4-flash"),
            base_url=url
        )

    if backend in ("claude", "anthropic"):
        return _claude(config, config.get("ANTHROPIC_BASE_URL") or None)

    if backend == "ollama":
        return OllamaLLM(
            model=config.get("OLLAMA_MODEL", model or "qwen3:8b"),
            host=config.get("OLLAMA_HOST", "http://localhost:11434")
        )

    if backend in COMPAT_NAMES:
        if config.get("LLM_BASE_URL"):
            # the address is given: LLM_PROVIDER only fills in defaults (model, key variable);
            # anthropic means the server speaks Anthropic's Messages API at that address
            provider = (config.get("LLM_PROVIDER") or "").strip().lower() or None
            if provider in ANTHROPIC_NAMES:
                return _claude(config, config.get("ANTHROPIC_BASE_URL") or config["LLM_BASE_URL"])
            return _compat(provider, config)
        provider = _key_provider(config)
        if provider == "anthropic":
            return _create_live("claude", config)
        return _compat(provider, config)

    if backend in PRESETS:
        return _compat(backend, config)

    known = ", ".join([*NATIVE_BACKENDS, "openai_compat", *[p for p in PRESETS if p != "ollama"]])
    raise ValueError(f"unknown LLM_BACKEND {backend!r}; use one of: {known}")
