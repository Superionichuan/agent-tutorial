"""A stop that arrives while a timed-out tool command is being cleaned up must still stop the
task.

When the tool's timeout ends a call, the command's process group is killed and the tool then
waits up to 5 s for the command's output pipes to close. A process that left the group (one
that started a session of its own) can keep those pipes open for the whole 5 s. A Ctrl+C,
SIGTERM or SIGHUP that arrives in that wait has to stop the task like any other stop: exit
status 128 + signal, no further model call, no further tool call.

POSIX only (process groups and signals).
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from test_keepalive import Flaky, env_of, start, wait_for

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX process groups and signals")

STOPS = [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]


def call(action: str, **args) -> str:
    return json.dumps({"thought": "run it", "action": action, "args": args})


def with_tool_timeout(env: dict, folder: Path, seconds: float) -> dict:
    """The environment with the tool timeout set to `seconds` in every mini_agent process
    started with it (a sitecustomize module on PYTHONPATH)."""
    site = folder / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(      # import_module: the package also exports a
        "import importlib\n"                        # function named bash
        f"importlib.import_module('mini_agent.tools.shell.bash').TIMEOUT = {seconds!r}\n",
        encoding="utf-8")
    env = dict(env)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(site), env.get("PYTHONPATH")) if p)
    return env


def escaping_command() -> str:
    """A command that starts a process in a session of its own (it outlives the group kill and
    keeps the tool's pipes open), then runs until the timeout ends it."""
    escape = (f"{sys.executable} -c \"import os, time; os.setsid(); "
              "open('escaped', 'w').write(str(os.getpid())); time.sleep(30)\"")
    return f"{escape} & touch ready; while :; do sleep 0.1; done"


def end_escaped(folder: Path) -> None:
    path = folder / "escaped"
    if path.exists() and path.read_text().strip():
        try:
            os.kill(int(path.read_text()), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, ValueError):
            pass


def finish(proc: subprocess.Popen):
    try:
        return proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()


@pytest.mark.parametrize("sig", STOPS, ids=lambda s: signal.Signals(s).name)
def test_a_stop_while_a_timed_out_command_is_cleaned_up_stops_the_task(tmp_path, sig):
    with Flaky([call("bash", cmd=escaping_command()), call("bash", cmd="touch after")]) as server:
        proc = start(["--yes", "run it"], with_tool_timeout(env_of(server), tmp_path, 1), tmp_path)
        try:
            assert wait_for(lambda: (tmp_path / "ready").exists()
                            and (tmp_path / "escaped").exists(), 30)
            time.sleep(2.5)          # the timeout came at about 1 s; the wait lasts until about 6 s
            proc.send_signal(sig)
            out, err = finish(proc)
            requests = server.requests
        finally:
            end_escaped(tmp_path)
    assert proc.returncode == 128 + sig, (proc.returncode, out[-600:], err[-600:])
    assert requests == 1
    assert not (tmp_path / "after").exists()


def test_keep_alive_a_stop_passed_on_during_that_wait_leaves_nothing_running(tmp_path):
    """The supervisor passes its stop on to the run 3 s later. When that lands in the wait after
    a timeout, the run must stop there: a run that went on to its next command would be killed
    in it 10 s later, and that command (a process group of its own) would be left running."""
    second = ("import os, time\n"                    # Python: its timeout is not shortened
              "open('pgid2', 'w').write(str(os.getpgrp()))\n"
              "time.sleep(30)\n")
    groups = []
    with Flaky([call("bash", cmd=escaping_command()), call("run_python", code=second)]) as server:
        env = with_tool_timeout(env_of(server), tmp_path, 3)
        proc = start(["--keep-alive", "--yes", "run it"], env, tmp_path)
        try:
            assert wait_for(lambda: (tmp_path / "ready").exists()
                            and (tmp_path / "escaped").exists(), 30)
            time.sleep(1)            # timeout at about 3 s; the passed-on stop at about 4 s
            proc.send_signal(signal.SIGTERM)
            out, err = finish(proc)
            if (tmp_path / "pgid2").exists():
                groups.append(int((tmp_path / "pgid2").read_text()))
            left = []
            for g in groups:
                try:
                    os.killpg(g, 0)
                    left.append(g)
                except ProcessLookupError:
                    pass
        finally:
            for g in groups:
                try:
                    os.killpg(g, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            end_escaped(tmp_path)
    assert proc.returncode == 128 + signal.SIGTERM, (out[-600:], err[-600:])
    assert not (tmp_path / "pgid2").exists()          # no further command was started
    assert left == []
