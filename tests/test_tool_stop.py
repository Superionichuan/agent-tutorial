"""A tool's command stops with everything it started: on a stop signal, on Ctrl+C, at the
timeout. (A process the command started kept running after the agent had stopped.)"""
import importlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid

import pytest

from test_keepalive import Flaky, env_of, start, wait_for

bash_module = importlib.import_module("mini_agent.tools.shell.bash")     # the package exports
python_module = importlib.import_module("mini_agent.tools.shell.python")  # functions of these names

POSIX = pytest.mark.skipif(os.name == "nt", reason="POSIX process table and signals")


def marked_sleep() -> str:
    """A sleep whose command line is unique, so the process table shows whether it is left."""
    return f"sleep 30.{uuid.uuid4().int % 10**6:06d}"


def running(text: str) -> bool:
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return any(text in line for line in out.splitlines())


def bash_call(cmd: str) -> str:
    return json.dumps({"thought": "run it", "action": "bash", "args": {"cmd": cmd}})


@POSIX
@pytest.mark.parametrize("keep_alive", [False, True])
def test_a_stop_ends_what_the_tool_command_started(tmp_path, keep_alive):
    sleeper = marked_sleep()
    started, late = tmp_path / "started", tmp_path / "late"
    with Flaky([bash_call(f"touch '{started}' && {sleeper} && touch '{late}'")]) as server:
        env = env_of(server)
        proc = start((["--keep-alive"] if keep_alive else []) + ["--yes", "run it"], env, tmp_path)
        try:
            assert wait_for(lambda: started.exists() and running(sleeper), 60)
            proc.send_signal(signal.SIGTERM)
            out, err = proc.communicate(timeout=60)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
    assert proc.returncode == 128 + signal.SIGTERM, (out, err)
    time.sleep(1)
    assert not running(sleeper)
    assert not late.exists()


@POSIX
def test_the_timeout_ends_what_the_shell_command_started(tmp_path, monkeypatch):
    sleeper = marked_sleep()
    late = tmp_path / "late"
    monkeypatch.setattr(bash_module, "TIMEOUT", 1)
    monkeypatch.chdir(tmp_path)
    t0 = time.time()
    assert bash_module.bash(f"{sleeper}; touch '{late}'") == "Error: timeout"
    assert time.time() - t0 < 15
    time.sleep(1)
    assert not running(sleeper) and not late.exists()


def test_the_timeout_ends_what_the_python_code_started(tmp_path, monkeypatch):
    """Windows too: the call must not hang on a grandchild that holds the output pipes."""
    marker = f"30.{uuid.uuid4().int % 10**6:06d}"
    code = ("import subprocess, sys\n"
            f"subprocess.run([sys.executable, '-c', 'import time; time.sleep({marker})'])\n")
    monkeypatch.setattr(python_module, "TIMEOUT", 2)
    monkeypatch.chdir(tmp_path)
    t0 = time.time()
    assert python_module.run_python(code) == "Error: execution timeout"
    assert time.time() - t0 < 20
    if os.name != "nt":
        time.sleep(1)
        assert not running(f"time.sleep({marker})")


def test_tool_commands_get_no_terminal_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = python_module.run_python("import sys\nprint(repr(sys.stdin.read()))")
    assert out.strip() == "''"


@POSIX
def test_ctrl_c_in_the_interactive_mode_stops_the_tool_and_returns_to_the_prompt(tmp_path):
    import pty
    import select
    sleeper = marked_sleep()
    started = tmp_path / "started"
    server = Flaky([bash_call(f"touch '{started}' && {sleeper}")])
    master, slave = pty.openpty()
    env = env_of(server, TERM="xterm")

    def own_terminal():          # a new session whose controlling terminal is the pty, so
        os.setsid()              # that Ctrl+C typed there reaches the program as on a desk
        import fcntl
        import termios
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    proc = subprocess.Popen([sys.executable, "-m", "mini_agent", "--yes"], env=env,
                            stdin=slave, stdout=slave, stderr=slave, cwd=str(tmp_path),
                            preexec_fn=own_terminal)
    os.close(slave)
    seen = b""

    def read(timeout=0.2) -> None:      # keep reading: a full terminal buffer (small on
        nonlocal seen                      # macOS) blocks the program before it runs the tool
        if select.select([master], [], [], timeout)[0]:
            try:
                seen += os.read(master, 65536)
            except OSError:
                pass

    def wait_text(text: bytes, start=0, timeout=60):
        end = time.time() + timeout
        while text not in seen[start:]:
            assert time.time() < end, f"did not see {text!r}:\n{seen[-1500:]!r}"
            read()

    def wait_reading(condition, timeout=60) -> bool:
        end = time.time() + timeout
        while not condition():
            if time.time() > end:
                return False
            read()
        return True

    try:
        wait_text(b"\xe2\x9d\xaf")                       # the prompt
        os.write(master, b"run it\r")
        assert wait_reading(lambda: started.exists() and running(sleeper)), seen[-1500:]
        mark = len(seen)
        os.write(master, b"\x03")                        # Ctrl+C on the terminal
        wait_text(b"Interrupted", start=mark)
        wait_text(b"\xe2\x9d\xaf", start=seen.rindex(b"Interrupted"))
        assert wait_reading(lambda: not running(sleeper), 5)
        assert proc.poll() is None                      # still running, at the prompt
        for _ in range(10):
            try:
                os.write(master, b"/exit\r")
            except OSError:
                break
            if wait_reading(lambda: proc.poll() is not None, 3):
                break
    finally:
        if proc.poll() is None:
            proc.kill()
        os.close(master)
        server.__exit__()
