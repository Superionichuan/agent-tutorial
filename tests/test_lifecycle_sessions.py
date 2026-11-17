"""Lifecycle: a new task is shown the user messages and the open handoff of its own session.

Several one-shot tasks in one folder are separate sessions: the words of an earlier task are
kept in user.md but are not shown to a new task, so that it does not take them as requests.
A task continued with -c sees its own words; the messages of one interactive session (one
session file, or one Lifecycle object) stay in view as before.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from mini_agent import Agent  # noqa: E402
from mini_agent.core.agent.run import run_agent  # noqa: E402
from mini_agent.lifecycle import Lifecycle, restore_steps  # noqa: E402
from mini_agent.lifecycle import notes as N  # noqa: E402
from test_cli import FakeServer, cli, env_for  # noqa: E402

DONE = json.dumps({"thought": "", "action": "done", "args": {"answer": "ok"}})
WORK = json.dumps({"thought": "working", "action": "work", "args": {}})


def first_prompt(request):
    return request["messages"][0]["content"]


def test_a_second_one_shot_task_is_not_shown_the_first_as_a_request():
    server = FakeServer([DONE, DONE])
    try:
        env, work = env_for(server), tempfile.mkdtemp()
        assert cli("--yes", "Compute 3**7 and report it.", env=env, cwd=work).returncode == 0
        assert cli("--yes", "Write hello into hi.txt.", env=env, cwd=work).returncode == 0
        second = first_prompt(server.requests[1])
        assert "Write hello into hi.txt." in second
        assert "Compute 3**7" not in second
        recorded = [t for _, _, t in N.user_messages(Path(work) / ".mini_agent" / "user.md")]
        assert recorded == ["Compute 3**7 and report it.", "Write hello into hi.txt."]
    finally:
        server.close()


def test_continue_after_a_kill_restores_the_task_and_not_an_earlier_one():
    step = lambda cmd: json.dumps({"thought": "t", "action": "bash", "args": {"cmd": cmd}})  # noqa: E731
    server = FakeServer([
        DONE,                                                  # an earlier task, finished
        step("echo one >> count.txt"),
        step("echo two >> count.txt; kill -9 $PPID"),          # the agent process dies here
        json.dumps({"thought": "t", "action": "done", "args": {"answer": "finished"}}),
    ])
    try:
        env, work = env_for(server), tempfile.mkdtemp()
        assert cli("--yes", "EARLIER: compute 3**7.", env=env, cwd=work).returncode == 0
        assert cli("--yes", "COUNT to two in count.txt.", env=env, cwd=work).returncode != 0
        r = cli("-c", "--yes", env=env, cwd=work)
        assert "finished" in r.stdout, r.stdout + r.stderr
        for request in server.requests[1:]:                    # the counting task, before and after
            assert "COUNT to two in count.txt." in first_prompt(request)
            assert "EARLIER: compute" not in first_prompt(request)
        assert (Path(work) / "count.txt").read_text() == "one\ntwo\n"
    finally:
        server.close()


def test_the_messages_of_one_session_stay_in_view(tmp_path):
    prompts = []

    def model(prompt):
        prompts.append(prompt)
        return DONE

    agent = Agent(llm=model, workspace=tmp_path, lifecycle=True, use_memory=False, skills=[])
    agent.run("Answer in French from now on.", on_event=lambda *a, **k: None)
    agent.run("What is 2+2?", on_event=lambda *a, **k: None)
    assert "Answer in French from now on." in prompts[-1]      # as before the change


def test_a_session_named_by_its_file_keeps_its_messages_across_objects(tmp_path):
    prompts = []

    def model(prompt):
        prompts.append(prompt)
        return DONE

    for message_id, goal in (("s1.jsonl#1", "Use metric units."), ("s2.jsonl#1", "Unrelated task."),
                             ("s1.jsonl#4", "How tall is it?")):
        lc = Lifecycle(tmp_path)                    # a new process each time, as the CLI is
        lc.message_id = message_id
        run_agent(model, {}, "", goal, lifecycle=lc, verbose=False)
    assert "Use metric units." not in prompts[1]                 # another session
    assert "Use metric units." in prompts[2] and "Unrelated task." not in prompts[2]


def test_an_open_handoff_is_shown_to_its_own_task_only(tmp_path):
    events, prompts = [], []

    def model(prompt):
        prompts.append(prompt)
        return WORK if len(prompts) == 1 else DONE

    task_a = Lifecycle(tmp_path)
    task_a.message_id = "a.jsonl#1"
    run_agent(model, {"work": lambda: "half done"}, "", "Task A: migrate the data.",
              lifecycle=task_a, max_steps=1, verbose=False,
              on_event=lambda t, **d: events.append({"type": t, "data": d}))
    handoff = tmp_path / ".mini_agent" / "handoff.md"
    assert "stopped: max_steps" in N.open_handoff(handoff)

    task_b = Lifecycle(tmp_path)                    # a new one-shot task in the same folder
    task_b.message_id = "b.jsonl#1"
    run_agent(model, {}, "", "Task B: say hello.", lifecycle=task_b, verbose=False)
    assert "### Handoff" not in prompts[1] and "Task A" not in prompts[1]
    assert "stopped: max_steps" in N.open_handoff(handoff)       # B's done leaves A's handoff open

    continued = Lifecycle(tmp_path)                 # mini_agent -c on task A
    run_agent(model, {"work": lambda: "half done"}, "", "Task A: migrate the data.",
              lifecycle=continued, state=restore_steps(events), verbose=False)
    assert "### Handoff" in prompts[2] and "migrate the data" in prompts[2]
    assert "Task B" not in prompts[2]
    assert N.open_handoff(handoff) == ""                         # A finished: its handoff closed
