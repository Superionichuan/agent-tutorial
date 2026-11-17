"""Tool paths: relative to the agent's workspace during a run, unchanged otherwise.

Run: python tests/test_workspace.py
"""
import importlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from mini_agent import Agent
from mini_agent.tools import TOOLS
from mini_agent.tools import workspace
from test_harness import J, ScriptedLLM

shell_mod = importlib.import_module("mini_agent.tools.shell.bash")   # the module, not bash()


def run(replies, ws, **options):
    ev = []
    Agent(llm=ScriptedLLM(replies), use_memory=False, skills=[], workspace=ws, **options).run(
        "goal", on_event=lambda t, **d: ev.append((t, d)))
    return [d["content"] for t, d in ev if t == "tool/result"]


def test_relative_write_lands_in_workspace():
    ws = tempfile.mkdtemp()
    run([J("write_file", {"path": "sub/note.txt", "content": "x"}), J("done", {"answer": "ok"})], ws)
    assert (Path(ws) / "sub" / "note.txt").read_text() == "x"
    assert not Path("sub/note.txt").exists()


def test_relative_read_and_list():
    ws = tempfile.mkdtemp()
    (Path(ws) / "data.txt").write_text("42")
    res = run([J("read_file", {"path": "data.txt"}), J("list_dir", {"path": "."}),
               J("done", {"answer": "ok"})], ws)
    assert res[0] == "42" and "data.txt" in res[1]


def native(path):
    return os.path.normcase(os.path.realpath(path.strip()))


def test_shell_runs_in_workspace():
    ws = os.path.realpath(tempfile.mkdtemp())
    kind = shell_mod.shell()[0]                      # Git Bash: pwd -W gives C:/Users/...
    where = ("(Get-Location).Path" if kind == "powershell"
             else "pwd -W" if os.name == "nt" else "pwd")
    res = run([J("bash", {"cmd": where}),
               J("run_python", {"code": "import os; print(os.getcwd())"}),
               J("done", {"answer": "ok"})], ws)
    assert native(res[0]) == os.path.normcase(ws) and native(res[1]) == os.path.normcase(ws)


def test_absolute_path_outside_is_refused_unless_confine_is_off():
    ws, other = tempfile.mkdtemp(), tempfile.mkdtemp()
    target = os.path.join(other, "abs.txt")
    res = run([J("write_file", {"path": target, "content": "y"}), J("done", {"answer": "ok"})], ws)
    assert res[0].startswith("Error:") and "outside the workspace" in res[0]
    assert not Path(target).exists()
    run([J("write_file", {"path": target, "content": "y"}), J("done", {"answer": "ok"})], ws,
        confine=False)
    assert Path(target).read_text() == "y"
    inside = os.path.join(ws, "in.txt")             # an absolute path inside is fine
    run([J("write_file", {"path": inside, "content": "z"}), J("done", {"answer": "ok"})], ws)
    assert Path(inside).read_text() == "z"


def test_default_is_current_directory_and_reset_after_run():
    assert workspace.current.get() is None
    cwd = tempfile.mkdtemp()
    old = os.getcwd()
    os.chdir(cwd)
    try:
        run([J("write_file", {"path": "here.txt", "content": "z"}), J("done", {"answer": "ok"})], None)
        assert (Path(cwd) / "here.txt").exists()
    finally:
        os.chdir(old)
    assert workspace.current.get() is None


def test_missing_workspace_is_created():
    ws = os.path.join(tempfile.mkdtemp(), "new", "ws")
    res = run([J("list_dir", {"path": "."}), J("done", {"answer": "ok"})], ws)
    assert os.path.isdir(ws) and res[0] == "(empty directory)"


def test_uncreatable_workspace_does_not_crash():
    fd, blocker = tempfile.mkstemp()                 # a file where a folder would have to be
    os.close(fd)
    res = run([J("list_dir", {"path": "."}), J("done", {"answer": "ok"})],
              os.path.join(blocker, "no", "ws"))
    assert res[0].startswith("Error")


def test_tool_outside_a_run_uses_cwd():
    assert TOOLS["read_file"]("/nonexistent/x").startswith("Error")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
