"""mini_agent doctor: what is installed, which model is set up and where that comes from, what is missing.

Exit code 0 when the installation works (also when no model is set up yet);
1 when the package itself is broken (missing files, Python too old).
"""
import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

from .. import __version__, paths


def _secret(key: str) -> bool:
    return key.endswith("_KEY") or key.endswith("_TOKEN")


def _shown(key: str, value: str) -> str:
    from .setup import mask
    return f"{key}={mask(value) if _secret(key) else value}"


def _version_of(module: str) -> str:
    try:
        from importlib.metadata import version
        return version(module)
    except Exception:
        return "?"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mini_agent doctor", description=__doc__.split("\n\n")[0])
    p.add_argument("--test", action="store_true", help="also make one small call to the configured model")
    return p


def main(argv: List[str]) -> int:
    args = build_parser().parse_args(argv)
    from ..llm.factory import PACKAGE_ENV, SELECT_KEYS, choose_backend, config_layers, load_config
    from ..tools.shell.bash import describe_system, shell

    broken = []
    pkg = Path(__file__).resolve().parent.parent
    print(f"mini_agent {__version__}")
    print(f"  Python {sys.version.split()[0]}  ({sys.executable})")
    print(f"  package: {pkg}")
    print(f"  system: {describe_system()}  [{shell()[1]}]")
    print(f"  home: {paths.home()}  (memory, skills, config.env)")
    if sys.version_info < (3, 10):
        broken.append("Python 3.10 or newer is needed")
    for rel in ("prompts/agent.md", "prompts/coding.md", "llm/recordings/tutorial.jsonl"):
        if not (pkg / rel).is_file():
            broken.append(f"missing package file {rel} (reinstall mini_agent)")

    print("\nSettings, highest priority first:")
    layers = config_layers()
    found = {name: values for name, _, values in layers}
    for name, path in (("environment", None), ("./.env", Path.cwd() / ".env"),
                       ("user config", paths.config_file())):
        label = f"{name} ({path})" if path else name
        if name in found:
            shown = ", ".join(_shown(k, v) for k, v in found[name].items()) or "(nothing relevant)"
            print(f"  {label}: {shown}")
        else:
            print(f"  {label}: not present")
    names = set(found)
    if PACKAGE_ENV.is_file():
        if "package .env" in names:
            supplied = ", ".join(_shown(k, v) for k, v in dict(next(v for n, _, v in layers
                                                                   if n == "package .env")).items())
            print(f"  Warning: the model setting comes from {PACKAGE_ENV} ({supplied}). A notebook wrote "
                  "that file. Run `mini_agent setup` to choose your own, or delete the file.")
        else:
            print(f"  note: {PACKAGE_ENV} (written by a notebook) is ignored: a model is set up above")

    config = load_config()
    configured = any(v for k, v in config.items() if k in SELECT_KEYS)
    backend = choose_backend(config)
    print("\nModel:")
    if not configured and backend == "ollama":
        print("  not set up yet: run `mini_agent setup` (or use a local Ollama server)")
    else:
        from ..llm.factory import create_llm
        try:
            llm = create_llm(config)
            where = getattr(llm, "base_url", "") or ""
            print(f"  {backend} · {getattr(llm, 'model', '?')}" + (f"  at {where}" if where else ""))
            if args.test:
                reply = llm("Reply with the single word: ready")
                print(f"  test call: {str(reply).strip()[:200]}")
        except Exception as e:
            print(f"  cannot use these settings: {e}")

    print("\nOptional:")
    print(f"  full-screen interface (mini_agent --tui): textual {_version_of('textual')}"
          if importlib.util.find_spec("textual") else
          "  full-screen interface (mini_agent --tui): textual is not installed (pip install textual)")
    docker = shutil.which("docker")
    if docker:
        try:
            out = subprocess.run([docker, "version", "--format", "{{.Server.Version}}"],
                                 capture_output=True, text=True, timeout=10)
            state = f"Docker {out.stdout.strip()}" if out.returncode == 0 else "Docker installed, not running"
        except Exception:
            state = "Docker installed, not answering"
    else:
        state = "Docker not found"
    print(f"  --sandbox docker: {state}")

    if broken:
        print("\nProblems:")
        for b in broken:
            print(f"  - {b}")
        return 1
    print("\nOK" + ("" if configured or backend != "ollama" else " (run `mini_agent setup` to choose a model)"))
    return 0
