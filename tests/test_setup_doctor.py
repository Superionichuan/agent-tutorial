"""mini_agent setup / doctor, the order of the settings sources, and the shell on each OS."""
import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_cli import FakeServer, cli, env_for

import importlib

from mini_agent.cli.setup import mask
from mini_agent.llm import factory

shell_mod = importlib.import_module("mini_agent.tools.shell.bash")   # the module, not the bash() function


def run_setup(*args, env, key=None):
    return subprocess.run([sys.executable, "-m", "mini_agent", "setup", *args], env=env,
                          input=key, capture_output=True, encoding="utf-8", errors="replace",
                          timeout=120,
                          cwd=tempfile.mkdtemp())


def config_of(env):
    return Path(env["MINI_AGENT_HOME"]) / "config.env"


# ---- doctor ----

def test_doctor_without_any_settings():
    r = cli("doctor", env=env_for())
    assert r.returncode == 0, r.stdout + r.stderr
    assert "not set up yet" in r.stdout and "mini_agent setup" in r.stdout
    assert "Traceback" not in r.stderr


# ---- setup ----

def test_setup_offline_then_a_task():
    env = env_for()
    r = run_setup("--provider", "offline", env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    path = config_of(env)
    assert "LLM_BACKEND=offline" in path.read_text(encoding="utf-8")
    if os.name != "nt":
        assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert "offline" in cli("doctor", env=env).stdout
    r = cli("Compute 3**7.", env=env)
    assert "2187" in r.stdout.split("Answer:")[-1]


def test_setup_with_a_key_on_stdin_tests_then_saves():
    server = FakeServer(["ready"])
    env = env_for()
    try:
        r = run_setup("--provider", "compat", "--base-url", server.url, "--model", "m1", "--key-stdin",
                      env=env, key="sk-test-0123456789abcdef\n")
    finally:
        server.close()
    assert r.returncode == 0, r.stdout + r.stderr
    saved = config_of(env).read_text(encoding="utf-8")
    assert "LLM_BACKEND=openai_compat" in saved and f"LLM_BASE_URL={server.url}" in saved
    assert "LLM_API_KEY=sk-test-0123456789abcdef" in saved and "LLM_MODEL=m1" in saved
    assert "sk-test-0123456789abcdef" not in r.stdout          # shown masked only
    assert "it answered: ready" in r.stdout
    doctor = cli("doctor", env=env).stdout
    assert "openai_compat · m1" in doctor and "sk-test-0123456789abcdef" not in doctor


def test_setup_does_not_save_when_the_test_call_fails():
    server = FakeServer([], status=401)
    env = env_for()
    try:
        r = run_setup("--provider", "compat", "--base-url", server.url, "--model", "m", "--key-stdin",
                      env=env, key="bad-key-0000000000\n")
    finally:
        server.close()
    assert r.returncode == 1 and "Nothing was saved" in r.stdout
    assert not config_of(env).exists()


def test_setup_needs_a_terminal_or_options():
    r = run_setup(env=env_for())
    assert r.returncode == 2 and "--provider" in r.stderr
    r = run_setup("--provider", "nonsense", env=env_for())
    assert r.returncode == 2


@pytest.mark.skipif(os.name == "nt", reason="pseudo-terminals are POSIX-only")
def test_interactive_setup_on_a_terminal():
    import pty
    import select
    import time

    server = FakeServer(["ready"])
    env = env_for(TERM="xterm")
    master, slave = pty.openpty()
    proc = subprocess.Popen([sys.executable, "-m", "mini_agent", "setup"], env=env, stdin=slave,
                            stdout=slave, stderr=slave, cwd=tempfile.mkdtemp())
    os.close(slave)
    seen = b""

    def wait_for(text: bytes, timeout=60):
        nonlocal seen
        end = time.time() + timeout
        while text not in seen:
            assert time.time() < end, f"did not see {text!r}: {seen[-1500:]!r}"
            if select.select([master], [], [], 0.2)[0]:
                try:
                    seen += os.read(master, 65536)
                except OSError:
                    break

    try:
        wait_for(b"Number or name")
        os.write(master, b"compat\r")
        wait_for(b"Server address")
        os.write(master, server.url.encode() + b"\r")
        wait_for(b"typing is hidden")
        os.write(master, b"sk-typed-0123456789abcdef\r")
        wait_for(b"Model name")
        os.write(master, b"m2\r")
        wait_for(b"Saved to")
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
        os.close(master)
        server.close()
    assert proc.returncode == 0
    assert b"sk-typed-0123456789abcdef" not in seen               # the key was never echoed
    saved = config_of(env).read_text(encoding="utf-8")
    assert "LLM_API_KEY=sk-typed-0123456789abcdef" in saved and "LLM_MODEL=m2" in saved


def test_mask():
    assert mask("sk-or-v1-abcdefghijklmnopf326") == "sk-or-v1-…f326"
    assert mask("sk-ant-api03-xyz123456789") == "sk-ant…6789"
    assert mask("short") == "****"


# ---- where settings come from ----

@pytest.fixture
def places(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key in factory.CONFIG_KEYS:
            monkeypatch.delenv(key)
    home = tmp_path / "home"
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("MINI_AGENT_HOME", str(home))
    monkeypatch.chdir(work)
    package_env = tmp_path / "package.env"
    monkeypatch.setattr(factory, "PACKAGE_ENV", package_env)
    home.mkdir()
    return {"user": home / "config.env", "cwd": work / ".env", "package": package_env}


def write(path, **values):
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")


def test_each_setting_from_the_highest_layer(places, monkeypatch):
    write(places["user"], LLM_BACKEND="openrouter", OPENROUTER_API_KEY="k-user", LLM_MODEL="user-model")
    write(places["cwd"], LLM_MODEL="project-model")
    config = factory.load_config()
    assert config["LLM_MODEL"] == "project-model" and config["OPENROUTER_API_KEY"] == "k-user"
    monkeypatch.setenv("LLM_MODEL", "env-model")
    assert factory.load_config()["LLM_MODEL"] == "env-model"


def test_the_package_env_never_overrides_a_choice(places, monkeypatch):
    write(places["package"], LLM_BACKEND="offline")
    assert factory.load_config()["LLM_BACKEND"] == "offline"          # nothing else: it is used
    names = [n for n, _, _ in factory.config_layers()]
    assert names[-1] == "package .env"

    monkeypatch.setenv("LLM_API_KEY", "sk-or-v1-x")                   # a key in the environment wins
    config = factory.load_config()
    assert "LLM_BACKEND" not in config and factory.choose_backend(config) == "openai_compat"
    monkeypatch.delenv("LLM_API_KEY")

    write(places["user"], LLM_BACKEND="deepseek", DEEPSEEK_API_KEY="k")  # so does setup's file
    assert factory.load_config()["LLM_BACKEND"] == "deepseek"
    assert "package .env" not in [n for n, _, _ in factory.config_layers()]


def test_a_given_env_file_replaces_the_usual_files(places, tmp_path):
    write(places["user"], LLM_BACKEND="deepseek")
    other = tmp_path / "other.env"
    write(other, LLM_BACKEND="offline", LLM_MODEL="x")
    assert factory.load_config(other)["LLM_BACKEND"] == "offline"


# ---- the shell ----

def test_system_line_names_the_shell():
    line = shell_mod.describe_system()
    assert "bash tool runs" in line


def test_powershell_command_is_encoded(monkeypatch):
    monkeypatch.setenv("MINI_AGENT_SHELL", "powershell")
    argv = shell_mod.command_line('Write-Output "你好 $((1))"')
    assert argv[1:4] == ["-NoProfile", "-NonInteractive", "-EncodedCommand"]
    script = base64.b64decode(argv[4]).decode("utf-16-le")
    assert script.endswith('Write-Output "你好 $((1))"') and "UTF8" in script


def test_shell_override_and_utf8_output(monkeypatch):
    if os.name == "nt":
        pytest.skip("POSIX shells")
    monkeypatch.setenv("MINI_AGENT_SHELL", "sh")
    assert shell_mod.shell()[0] == "sh"
    assert shell_mod.bash("printf '%s\\n' 中文") == "中文\n"


@pytest.mark.skipif(os.name != "nt", reason="Windows only")
def test_windows_shells_and_paths(monkeypatch, tmp_path):
    kind, exe = shell_mod.shell()
    assert kind in ("bash", "powershell") and "system32\\bash" not in exe.lower()
    monkeypatch.setenv("MINI_AGENT_SHELL", "powershell")
    assert shell_mod.bash('Write-Output "你好"').strip() == "你好"
    from mini_agent.tools import read_file, workspace
    ws = tmp_path / "ws"
    ws.mkdir()
    token, limit = workspace.current.set(ws), workspace.confined.set(True)
    try:
        assert "outside the workspace" in read_file("..\\secret.txt")
        assert "outside the workspace" in read_file("C:\\Windows\\win.ini")
        (ws / "中文.txt").write_text("内容", encoding="utf-8")
        assert read_file("中文.txt") == "内容"
    finally:
        workspace.confined.reset(limit)
        workspace.current.reset(token)
