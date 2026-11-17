"""mini_agent setup: choose a model provider, paste a key, test it, save the settings.

The settings go to ~/.mini_agent/config.env ($MINI_AGENT_HOME), never into a
project directory. Environment variables and a ./.env file still override them.

  mini_agent setup                                   asks step by step
  mini_agent setup --provider offline                no key, no network (the recorded demo)
  mini_agent setup --provider openrouter --key-stdin < key.txt   (CI; the key never on the command line)
"""
import argparse
import getpass
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import httpx

from .. import paths
from ..llm.base import mask, redact
from ..llm.backends.openai_compat import PRESETS, detect_provider

# name, label, backend, key variable (None = no key), default model, needs a base URL
PROVIDERS: List[Tuple[str, str, str, Optional[str], str, bool]] = [
    ("openrouter", "OpenRouter: one key, many models (openrouter.ai)", "openrouter",
     "OPENROUTER_API_KEY", PRESETS["openrouter"].model, False),
    ("deepseek", "DeepSeek (platform.deepseek.com)", "deepseek", "DEEPSEEK_API_KEY", "deepseek-v4-flash", False),
    ("anthropic", "Anthropic Claude (console.anthropic.com)", "claude", "ANTHROPIC_API_KEY", "", False),
    ("openai", "OpenAI (platform.openai.com)", "openai", "OPENAI_API_KEY", "", False),
    ("gemini", "Google Gemini (aistudio.google.com)", "gemini", "GEMINI_API_KEY", "", False),
    ("qwen", "Alibaba Qwen (DashScope)", "qwen", "QWEN_API_KEY", "qwen-plus", False),
    ("compat", "another OpenAI-compatible server (vLLM, LM Studio, a company gateway, ...)",
     "openai_compat", "LLM_API_KEY", "", True),
    ("ollama", "a local Ollama server (no key)", "ollama", None, "qwen3:8b", False),
    ("offline", "the recorded demo (no key, no network; only the tutorial's own tasks)",
     "offline", None, "", False),
]
BY_NAME = {p[0]: p for p in PROVIDERS}
ALIASES = {"claude": "anthropic", "openai_compat": "compat", "openai-compatible": "compat",
           "dashscope": "qwen", "google": "gemini"}


def _ask(prompt: str, default: str = "") -> str:
    shown = f"{prompt} [{default}]: " if default else f"{prompt}: "
    answer = input(shown).strip()
    return answer or default


def _choose_provider() -> str:
    print("Which model provider?")
    for i, (name, label, *_rest) in enumerate(PROVIDERS, 1):
        print(f"  {i}. {name:<11} {label}")
    while True:
        choice = _ask("Number or name", "1").lower()
        if choice.isdigit() and 1 <= int(choice) <= len(PROVIDERS):
            return PROVIDERS[int(choice) - 1][0]
        choice = ALIASES.get(choice, choice)
        if choice in BY_NAME:
            return choice
        print("  Please type a number from the list, or a name such as openrouter.")


def list_models(provider: str, key: str, base_url: str = "") -> List[str]:
    """Model names the provider offers (empty when it cannot be asked)."""
    try:
        if provider == "anthropic":
            r = httpx.get("https://api.anthropic.com/v1/models", timeout=20,
                          headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
        else:
            url = base_url or PRESETS[BY_NAME[provider][2]].base_url
            r = httpx.get(f"{url.rstrip('/')}/models", timeout=20,
                          headers={"Authorization": f"Bearer {key}"} if key else {})
        r.raise_for_status()
        return [m.get("id", "") for m in r.json().get("data", []) if m.get("id")]
    except Exception:
        return []


def settings_for(provider: str, key: str, model: str, base_url: str) -> Dict[str, str]:
    name, _label, backend, key_var, _default, _needs_url = BY_NAME[provider]
    values = {"LLM_BACKEND": backend}
    if base_url:
        values["LLM_BASE_URL"] = base_url
    if key_var and key:
        values[key_var] = key
    if model:
        values["LLM_MODEL"] = model
    return values


def test_call(values: Dict[str, str]) -> Tuple[bool, str]:
    """One small call with these settings: (worked, reply or error)."""
    from ..llm.base import LLMError
    from ..llm.factory import create_llm
    secrets = [v for k, v in values.items() if k.endswith("_KEY")]
    llm = create_llm(dict(values))
    reply = llm("Reply with the single word: ready")
    if isinstance(reply, LLMError) or str(reply).startswith("Error:"):
        return False, redact(str(reply), *secrets)
    return True, redact(str(reply).strip()[:80], *secrets)


def write_config(values: Dict[str, str]) -> str:
    """Save to ~/.mini_agent/config.env, replacing the old file (no backup copy: it
    would keep an old key). The new file is written next to it and renamed over it,
    so a symlink at that name is replaced, never written through."""
    target = paths.config_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_dir() and not target.is_symlink():
        raise IsADirectoryError(f"{target} is a directory; move it away, then run setup again")
    lines = [f"# mini_agent settings, written by `mini_agent setup` on {time.strftime('%Y-%m-%d %H:%M')}",
             "# Environment variables and a .env file in the working directory override these."]
    lines += [f"{k}={v}" for k, v in values.items()]
    temp = target.with_name(f".config.env.{os.getpid()}.tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        if os.name != "nt":
            os.chmod(temp, 0o600)
        os.replace(temp, target)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise
    return str(target)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mini_agent setup", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", help=", ".join(BY_NAME))
    p.add_argument("--model", help="model name (default: the provider's suggestion)")
    p.add_argument("--base-url", help="server address, for --provider compat (e.g. http://localhost:8000/v1)")
    p.add_argument("--key-stdin", action="store_true",
                   help="read the API key from standard input (for scripts; never put a key on the command line)")
    p.add_argument("--no-test", action="store_true", help="save without the test call")
    return p


def main(argv: List[str]) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _setup(args)
    except (EOFError, KeyboardInterrupt):
        print("\nmini_agent setup: stopped before the end; nothing was saved.", file=sys.stderr)
        return 1


def _setup(args) -> int:
    interactive = not args.key_stdin and sys.stdin is not None and sys.stdin.isatty()
    provider = ALIASES.get((args.provider or "").lower(), (args.provider or "").lower())
    if provider and provider not in BY_NAME:
        print(f"mini_agent setup: unknown provider {args.provider!r}; use one of: {', '.join(BY_NAME)}",
              file=sys.stderr)
        return 2
    if not provider:
        if not interactive:
            print("mini_agent setup: no terminal to ask on; give --provider (and --key-stdin for a key)",
                  file=sys.stderr)
            return 2
        provider = _choose_provider()
    name, label, backend, key_var, default_model, needs_url = BY_NAME[provider]

    base_url = args.base_url or ""
    if needs_url and not base_url:
        if not interactive:
            print("mini_agent setup: --provider compat needs --base-url", file=sys.stderr)
            return 2
        base_url = _ask("Server address (ends in /v1)", "http://localhost:8000/v1")

    key = ""
    if key_var:
        if args.key_stdin:
            key = sys.stdin.readline().strip()
        elif interactive:
            key = getpass.getpass(f"Paste your {name} API key (typing is hidden): ").strip()
        if not key and provider != "compat":
            print("mini_agent setup: no key given", file=sys.stderr)
            return 2
        guess = detect_provider(key)
        if guess and ALIASES.get(guess, guess) not in (provider, "compat") and provider != "compat":
            print(f"Note: this key looks like a {guess} key, not {provider}.")
            if interactive and _ask(f"Use {guess} instead? (y/n)", "y").lower().startswith("y"):
                provider = ALIASES.get(guess, guess)
                name, label, backend, key_var, default_model, needs_url = BY_NAME[provider]

    model = args.model or default_model
    if not model and provider not in ("offline",):
        if interactive:
            offered = list_models(provider, key, base_url)
            if offered:
                print("Some models this key can use:", ", ".join(offered[:12]))
            model = _ask("Model name", offered[0] if offered else "")
        if not model:
            print("mini_agent setup: give a model name with --model", file=sys.stderr)
            return 2

    values = settings_for(provider, key, model, base_url)
    if not args.no_test and provider != "offline":
        print(f"Testing {provider}" + (f" · {model}" if model else "") + " ...")
        worked, reply = test_call(values)
        if worked:
            print(f"  it answered: {reply}")
        else:
            print(f"  the test call failed: {reply}")
            if not interactive or not _ask("Save these settings anyway? (y/n)", "n").lower().startswith("y"):
                print("Nothing was saved.")
                return 1
    try:
        where = write_config(values)
    except OSError as e:
        print(f"mini_agent setup: cannot save the settings: {e}", file=sys.stderr)
        return 1
    shown = ", ".join(f"{k}={mask(v) if k.endswith('_KEY') else v}" for k, v in values.items())
    print(f"Saved to {where}: {shown}")
    print('Try it:  mini_agent "What is 2**10?"      Check:  mini_agent doctor')
    return 0
