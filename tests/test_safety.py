"""Safety defaults: approval policies, file tools confined to the workspace, failed model calls."""
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from mini_agent import Agent, tool
from mini_agent.core.agent.run import run_agent
from mini_agent.core import policy
from mini_agent.core.policy import (Approval, NeedsApproval, console_ask, resolve_approval,
                                    summarize_call)
from mini_agent.llm import OpenAICompatLLM
from mini_agent.llm.base import LLMError
from mini_agent.tools import TOOLS, ToolSpec, BUILTIN_SPECS, read_file, workspace
from test_harness import J, ScriptedLLM


def run(replies, **options):
    events = []
    llm = ScriptedLLM(replies)
    options.setdefault("workspace", tempfile.mkdtemp())
    answer = Agent(llm=llm, use_memory=False, skills=[], **options).run(
        "goal", on_event=lambda t, **d: events.append((t, d)))
    results = [d["content"] for t, d in events if t == "tool/result"]
    end = [d for t, d in events if t == "turn/end"][-1]
    for t, d in events:
        json.dumps({"type": t, **d})              # every event is plain JSON data
    return answer, results, end, llm


# ---- approval ----

def test_library_default_runs_everything():
    _, res, _, _ = run([J("bash", {"cmd": "echo hi"}), J("done", {"answer": "ok"})])
    assert res == ["hi\n"]


def test_deny_blocks_risky_tools_only():
    _, res, _, _ = run([J("bash", {"cmd": "echo hi"}), J("calc", {"expr": "2+2"}),
                        J("done", {"answer": "ok"})], approve="deny")
    assert res[0] == "Error: 'bash' was not approved and did not run" and res[1] == "4"


def test_ask_policy_yes_no_always():
    asked = []

    def answers(*replies):
        it = iter(replies)
        return lambda action, args: (asked.append((action, dict(args))), next(it))[1]

    _, res, _, _ = run([J("write_file", {"path": "a.txt", "content": "x"}), J("done", {"answer": "ok"})],
                       approve=Approval("ask", ask=answers("no")))
    assert res[0].startswith("Error:") and asked == [("write_file", {"path": "a.txt", "content": "x"})]

    asked.clear()
    ws = tempfile.mkdtemp()
    approval = Approval("ask", ask=answers("always"))
    _, res, _, _ = run([J("bash", {"cmd": "echo 1"}), J("bash", {"cmd": "echo 2"}), J("list_dir", {}),
                        J("done", {"answer": "ok"})], approve=approval, workspace=ws)
    assert res[:2] == ["1\n", "2\n"] and len(asked) == 1        # asked once; list_dir never asked
    assert approval.always == {"bash"}


def test_policy_name_on_agent_keeps_always_between_runs():
    asked = []
    agent = Agent(llm=ScriptedLLM([J("bash", {"cmd": "echo a"}), J("done", {"answer": "1"}),
                                   J("bash", {"cmd": "echo b"}), J("done", {"answer": "2"})]),
                  use_memory=False, skills=[], workspace=tempfile.mkdtemp(), approve="ask")
    assert isinstance(agent.approve, Approval)
    agent.approve.ask = lambda action, args: (asked.append(action), "always")[1]
    agent.run("one", verbose=False)
    agent.run("two", verbose=False)
    assert asked == ["bash"]


def test_risky_custom_tool_is_asked_and_plain_one_is_not():
    @tool(risky=True)
    def launch(target: str) -> str:
        """Launch something."""
        return f"launched {target}"

    @tool
    def look(target: str) -> str:
        """Look at something."""
        return f"saw {target}"

    asked = []
    _, res, _, _ = run([J("launch", {"target": "x"}), J("look", {"target": "y"}), J("done", {"answer": "ok"})],
                       tools=[launch, look],
                       approve=Approval("ask", ask=lambda a, k: (asked.append(a), "no")[1]))
    assert asked == ["launch"]
    assert res[0].startswith("Error:") and res[1] == "saw y"


def test_builtin_shell_keeps_its_risk_under_any_name():
    from mini_agent.tools import bash as builtin_bash
    for tools in ({"shell": builtin_bash}, [builtin_bash], {"bash": lambda cmd: "own"},
                  [ToolSpec("bash", "d", BUILTIN_SPECS["bash"].parameters, builtin_bash, risky=False)]):
        name = "shell" if isinstance(tools, dict) and "shell" in tools else "bash"
        asked = []
        _, res, _, _ = run([J(name, {"cmd": "echo hi"}), J("done", {"answer": "ok"})], tools=tools,
                           approve=Approval("ask", ask=lambda a, k: (asked.append(a), "no")[1]))
        assert asked == [name], tools
        assert res[0].startswith("Error:")


def test_entries_added_to_the_builtin_table_still_run():
    def lookup(q: str) -> str:
        """Look up."""
        return "found " + q
    TOOLS["lookup"] = lookup
    try:
        _, res, _, llm = run([J("lookup", {"q": "x"}), J("done", {"answer": "ok"})])
    finally:
        del TOOLS["lookup"]
    assert res == ["found x"]


def test_custom_function_and_bad_values():
    assert resolve_approval(None, ["bash"]) is None
    assert resolve_approval("auto", ["bash"]) is None
    check = lambda action, args: action != "calc"
    assert resolve_approval(check, ["bash"]) is check
    with pytest.raises(ValueError):
        Approval("sometimes")
    with pytest.raises(TypeError):
        resolve_approval(42, [])


def test_console_ask_without_a_terminal_stops(monkeypatch):
    monkeypatch.setattr(policy, "stdin_is_terminal", lambda: False)
    with pytest.raises(NeedsApproval):
        console_ask("bash", {"cmd": "rm -rf /"})


def test_no_terminal_stops_the_run_at_the_first_risky_call(monkeypatch, tmp_path):
    monkeypatch.setattr(policy, "stdin_is_terminal", lambda: False)
    replies = iter(['{"thought": "", "action": "bash", "args": {"cmd": "echo hi"}}'] * 5)
    calls, events = [], []

    def model(prompt):
        calls.append(prompt)
        return next(replies)
    answer = run_agent(model, TOOLS, "", "say hi", approve="ask", verbose=False,
                       on_event=lambda kind, **data: events.append((kind, data)))
    assert len(calls) == 1 and answer.startswith("Stopped: bash needs approval")
    end = [d for k, d in events if k == "turn/end"][-1]
    assert end["reason"] == "needs_approval" and end["tool"] == "bash"
    result = [d for k, d in events if k == "tool/result"][-1]
    assert "did not run: it needs approval" in result["content"] and result["is_error"]


def test_a_person_saying_no_lets_the_task_go_on():
    replies = iter(['{"thought": "", "action": "bash", "args": {"cmd": "echo hi"}}',
                    '{"thought": "", "action": "done", "args": {"answer": "skipped"}}'])
    answer = run_agent(lambda prompt: next(replies), TOOLS, "", "say hi", verbose=False,
                       approve=Approval("ask", ask=lambda action, args: "no"))
    assert answer == "skipped"


def test_console_ask_reads_the_answer(monkeypatch):
    monkeypatch.setattr(policy, "stdin_is_terminal", lambda: True)
    for typed, expected in (("y", "yes"), ("YES", "yes"), ("a", "always"), ("", "no"), ("n", "no")):
        monkeypatch.setattr("builtins.input", lambda prompt, typed=typed: typed)
        assert console_ask("bash", {"cmd": "ls"}) == expected


def test_summary_shows_what_would_run():
    assert summarize_call("bash", {"cmd": "ls -la"}) == "ls -la"
    assert summarize_call("write_file", {"path": "a.txt", "content": "abc"}).startswith("a.txt  (3 characters)")
    assert len(summarize_call("run_python", {"code": "x" * 5000})) < 700


def test_run_agent_accepts_a_policy_name():
    llm = ScriptedLLM([J("bash", {"cmd": "echo hi"}), J("done", {"answer": "ok"})])
    events = []
    run_agent(llm, TOOLS, None, "goal", approve="deny", verbose=False,
              on_event=lambda t, **d: events.append((t, d)))
    assert [d["content"] for t, d in events if t == "tool/result"][0].startswith("Error:")


# ---- file tools stay inside the workspace ----

def test_paths_outside_the_workspace_are_refused():
    ws = Path(tempfile.mkdtemp())
    outside = Path(tempfile.mkdtemp()) / "secret.txt"
    outside.write_text("secret")
    _, res, _, _ = run([J("read_file", {"path": str(outside)}), J("read_file", {"path": "../x"}),
                        J("list_dir", {"path": ".."}), J("read_file", {"path": "~/.bashrc"}),
                        J("write_file", {"path": "ok/inside.txt", "content": "fine"}),
                        J("done", {"answer": "ok"})], workspace=str(ws))
    assert all(r.startswith("Error:") and "outside the workspace" in r for r in res[:4])
    assert res[4].startswith("OK:") and (ws / "ok" / "inside.txt").read_text() == "fine"


@pytest.mark.skipif(os.name == "nt", reason="symlinks need extra rights on Windows")
def test_symlinks_that_leave_the_workspace_are_refused():
    ws = Path(tempfile.mkdtemp())
    elsewhere = Path(tempfile.mkdtemp())
    (elsewhere / "secret.txt").write_text("secret")
    (ws / "link").symlink_to(elsewhere, target_is_directory=True)
    (ws / "file-link").symlink_to(elsewhere / "secret.txt")
    (ws / "real.txt").write_text("ok")
    (ws / "inner-link").symlink_to(ws / "real.txt")
    _, res, _, _ = run([J("read_file", {"path": "link/secret.txt"}), J("read_file", {"path": "file-link"}),
                        J("write_file", {"path": "link/new.txt", "content": "x"}),
                        J("read_file", {"path": "inner-link"}), J("done", {"answer": "ok"})],
                       workspace=str(ws))
    assert all("outside the workspace" in r for r in res[:3])
    assert not (elsewhere / "new.txt").exists()
    assert res[3] == "ok"                       # a symlink that stays inside is fine


def test_confinement_only_during_a_run():
    assert workspace.confined.get() is False
    outside = Path(tempfile.mkdtemp()) / "f.txt"
    outside.write_text("free")
    assert read_file(str(outside)) == "free"        # direct calls are not limited
    run([J("done", {"answer": "ok"})])
    assert workspace.confined.get() is False


# ---- a failed model call stops the run at once ----

def test_llm_error_stops_without_a_second_call():
    answer, res, end, llm = run([LLMError("Error: HTTP 401: invalid api key"), J("done", {"answer": "x"})])
    assert end["reason"] == "llm_error" and answer == "Error: HTTP 401: invalid api key"
    assert llm.calls == 1 and res == []


def test_provider_error_through_a_real_backend():
    requests = []

    def server(request):
        requests.append(request)
        return httpx.Response(401, json={"error": {"message": "No auth credentials found"}})

    llm = OpenAICompatLLM(base_url="https://x.test/v1", model="m", transport=httpx.MockTransport(server))
    events = []
    answer = Agent(llm=llm, use_memory=False, skills=[], workspace=tempfile.mkdtemp()).run(
        "goal", on_event=lambda t, **d: events.append((t, d)))
    assert answer == "Error: HTTP 401: No auth credentials found"
    assert events[-1][1]["reason"] == "llm_error" and len(requests) == 1


def test_a_plain_reply_starting_with_error_is_still_a_reply():
    # text that merely starts with "Error:" (not an LLMError) is an invalid reply, as before
    answer, _, end, llm = run(["Error: I cannot do that", "Error: still no"])
    assert end["reason"] == "invalid_reply" and llm.calls == 2
