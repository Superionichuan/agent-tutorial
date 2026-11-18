"""Lifecycle: failures found in two further independent code reviews, kept as regression tests.

Each test failed on the code before the fixes. The prompt: a result or arguments too large
for the window among the last steps (dropped unseen, or sent over the window), a prompt
still over the window after compaction, a model summary requested at every step once the
prompt passed 80% of the window, a summary accepted although it was larger in tokens than
its source. The files: a .mini_agent link to a folder inside the workspace, a link swapped
in while a file opens, a user.md entry with CRLF line ends, an entry cut short by a crash,
a message recorded twice when its task was continued. The rest: a started tool call lost
when the turn ended with an error, and the earlier-session text over its budget.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mini_agent.core.agent.run import run_agent  # noqa: E402
from mini_agent.core.state import State  # noqa: E402
from mini_agent.lifecycle import INTERRUPTED, Lifecycle, restore_steps  # noqa: E402
from mini_agent.lifecycle import compact as C  # noqa: E402
from mini_agent.lifecycle import notes as N  # noqa: E402
from mini_agent.session import SessionLog  # noqa: E402

DONE = json.dumps({"thought": "", "action": "done", "args": {"answer": "ok"}})


class Model:
    """Answers done; a summary request gets a short summary. Records the agent prompts."""
    context_limit = 2000

    def __init__(self):
        self.prompts = []

    def __call__(self, prompt):
        if prompt.startswith(C.SUMMARY_PROMPT):
            return "read earlier data"
        self.prompts.append(prompt)
        return DONE


def step(result, args=None):
    return {"thought": "", "action": "read", "args": args or {}, "result": result}


def event(kind, **data):
    return {"type": kind, "data": data}


# ---------- the prompt fits the window ----------

def test_a_result_over_the_history_cap_keeps_its_head_and_tail(tmp_path):
    prompts = []
    result = "RESULT_HEAD" + "x" * 20000 + "RESULT_TAIL"

    def model(prompt):
        prompts.append(prompt)
        return json.dumps({"action": "large" if len(prompts) == 1 else "done", "args": {}})

    model.context_limit = 4000
    run_agent(model, {"large": lambda: result}, "", "inspect result",
              lifecycle=Lifecycle(tmp_path, llm=model, keep_steps=1), verbose=False)
    assert "RESULT_HEAD" in prompts[-1] and "RESULT_TAIL" in prompts[-1]
    assert C.count_tokens(prompts[-1]) <= model.context_limit


def test_large_arguments_of_a_recent_step_are_cut_to_fit(tmp_path):
    sizes = []

    def model(prompt):
        sizes.append(C.count_tokens(prompt))
        return json.dumps({"action": "work" if len(sizes) == 1 else "done",
                           "args": {"payload": "x" * 13000} if len(sizes) == 1 else {}})

    model.context_limit = 4000
    run_agent(model, {"work": lambda payload: "ok"}, "", "process payload",
              lifecycle=Lifecycle(tmp_path, llm=model, keep_steps=1), verbose=False)
    assert len(sizes) == 2 and max(sizes) <= model.context_limit


def test_the_prompt_after_compaction_fits_the_window(tmp_path):
    llm = Model()
    lc = Lifecycle(workspace=tmp_path, llm=llm, compact_at=1000, compact_to=800, keep_steps=1)
    state = State("goal", [step("old " * 1000), step("recent " * 1500)])
    events = []
    run_agent(llm, {}, "", "goal", state=state, lifecycle=lc,
              on_event=lambda t, **d: events.append((t, d)))
    assert any(t == "context/compact" for t, d in events)
    assert C.count_tokens(llm.prompts[-1]) <= llm.context_limit


def test_the_newest_result_is_in_the_history_the_model_sees(tmp_path):
    llm = Model()
    lc = Lifecycle(workspace=tmp_path, llm=llm, compact_at=1000, compact_to=800, keep_steps=1)
    state = State("goal", [step("old"), step("NEWEST_RESULT_SENTINEL" + "x" * 170000)])
    run_agent(llm, {}, "", "goal", state=state, lifecycle=lc, verbose=False)
    history = llm.prompts[-1].split("[Steps taken so far]\n", 1)[1]
    assert state.steps[-1]["result"] in history and "NEWEST_RESULT_SENTINEL" in history


def test_a_prompt_that_cannot_fit_stops_the_turn_with_a_handoff(tmp_path):
    calls, events = [], []

    def model(prompt):
        calls.append(prompt)
        return DONE

    model.context_limit = 4000
    answer = run_agent(model, {}, "", "g" * 13000, lifecycle=Lifecycle(tmp_path, llm=model),
                       on_event=lambda t, **d: events.append((t, d)))
    assert calls == []                                  # nothing over the window is sent
    assert [d["reason"] for t, d in events if t == "turn/end"] == ["context_full"]
    assert "Context full" in answer
    assert "stopped: context_full" in N.open_handoff(tmp_path / ".mini_agent" / "handoff.md")


def test_a_summary_waits_for_growth_also_above_80_percent(tmp_path):
    calls, compactions = [], []

    def model(prompt):
        if prompt.startswith(C.SUMMARY_PROMPT):
            calls.append("summary")
            return "prior work"
        calls.append("agent")
        return json.dumps({"thought": "continue", "action": "work", "args": {}})

    model.context_limit = 4000
    lc = Lifecycle(tmp_path, llm=model, keep_steps=1)
    run_agent(model, {"work": lambda: "r" * 600}, "", "g" * 8100, lifecycle=lc, max_steps=12,
              on_event=lambda t, **d: compactions.append(d) if t == "context/compact" else None)
    summaries = [c for c in compactions if c["method"] == "llm"]
    assert summaries and len(compactions) > len(summaries)          # mechanical ones in between
    assert all(c["before_tokens"] > lc.hard for c in compactions[1:])
    assert calls.count("summary") == len(summaries)
    early = [(a["after_tokens"], b["before_tokens"]) for a, b in zip(summaries, summaries[1:])
             if b["before_tokens"] < a["after_tokens"] + lc.trigger - lc.target]
    assert not early, f"summary repeated before the prompt grew: {early}"


def test_nothing_to_slim_above_80_percent_is_no_compaction(tmp_path):
    events, calls = [], []

    def model(prompt):
        calls.append(prompt)
        return json.dumps({"action": "work" if len(calls) < 4 else "done", "args": {"answer": "ok"}})

    model.context_limit = 4000
    lc = Lifecycle(tmp_path, llm=model, keep_steps=10)
    answer = run_agent(model, {"work": lambda: "ok"}, "", "g" * 9600, lifecycle=lc,
                       on_event=lambda t, **d: events.append(t))
    assert answer == "ok" and C.count_tokens(calls[0]) > lc.hard
    assert "context/compact" not in events
    assert N.open_handoff(tmp_path / ".mini_agent" / "handoff.md") == ""


def test_the_summary_guard_counts_tokens_with_the_framing():
    old = [step("a" * 450)]
    source = C.cut(f"read({{}}) → {'a' * 450}", C.SUMMARY_LINE)
    assert C.summarize(lambda prompt: "€" * 400, old) is None     # fewer characters, more tokens
    short = C.summarize(lambda prompt: "a read of 450 a", old)
    assert short and C.count_tokens(C.render(C.compacted(short, 1))) < C.count_tokens(source)


# ---------- the files ----------

def test_a_link_to_a_folder_inside_the_workspace_is_refused(tmp_path):
    target = tmp_path / "other"
    target.mkdir()
    try:
        (tmp_path / ".mini_agent").symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    events = []
    run_agent(lambda _: '{"action":"done","args":{}}', {}, "", "private goal",
              lifecycle=Lifecycle(tmp_path), verbose=False,
              on_event=lambda kind, **data: events.append(kind))
    assert not list(target.iterdir())
    assert "lifecycle/warning" in events


@pytest.mark.skipif(not N._AT, reason="files opened through a directory handle: POSIX only")
def test_a_link_swapped_in_while_a_file_opens_is_not_followed(tmp_path, monkeypatch):
    workspace, outside = tmp_path / "workspace", tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    real_open, swapped = os.open, []

    def open_after_swap(path, flags, *args, **kwargs):
        if path == "user.md" and flags & os.O_APPEND and not swapped:
            # another process swaps .mini_agent for a link after the checks, before the open
            (workspace / ".mini_agent").rename(workspace / "original")
            (workspace / ".mini_agent").symlink_to(outside, target_is_directory=True)
            swapped.append(True)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_after_swap)
    run_agent(lambda _: '{"action":"done","args":{}}', {}, "", "private goal",
              lifecycle=Lifecycle(workspace), verbose=False)
    assert swapped and not list(outside.iterdir())
    assert "private goal" in (workspace / "original" / "user.md").read_text(encoding="utf-8")


def test_an_entry_with_crlf_line_ends_survives_an_append(tmp_path):
    path = tmp_path / "user.md"
    path.write_bytes((N.USER_TITLE + "\n## 1 · 2026-11-17 12:00:00\nkeep this instruction\n")
                     .replace("\n", "\r\n").encode())
    N.append_user(path, "new request")
    assert [text for _, _, text in N.user_messages(path)] == ["keep this instruction", "new request"]


def test_an_append_after_a_cut_short_entry_keeps_the_new_message(tmp_path):
    path = tmp_path / "user.md"
    N.append_user(path, "x" * 300)
    path.write_bytes(path.read_bytes()[:-299])     # a crash left the header and two characters
    N.append_user(path, "fresh request")
    assert [(n, text) for n, _, text in N.user_messages(path)] == [(1, "xx"), (2, "fresh request")]


def test_a_message_sent_again_with_its_id_is_recorded_once(tmp_path):
    lc = Lifecycle(tmp_path)
    ids = ["s.jsonl#3", "s.jsonl#3", "my session.jsonl#5", "my session.jsonl#5", None, None]
    for message_id in ids:
        lc.message_id = message_id
        run_agent(lambda _: DONE, {}, "", "task", lifecycle=lc, verbose=False)
        assert lc.message_id is None                     # for one run only
    entries = N.entries(tmp_path / ".mini_agent" / "user.md")
    assert [(n, text, mid) for n, _, text, mid in entries[:2]] == [
        (1, "task", "s.jsonl#3"), (2, "task", "my_session.jsonl#5")]
    unnamed = [mid for _, _, _, mid in entries[2:]]        # runs not named: this object's session
    assert len(unnamed) == 2 and unnamed[0] != unnamed[1]
    assert N.session_of(unnamed[0]) == N.session_of(unnamed[1]) and unnamed[0].startswith("agent-")


# ---------- continuing a turn; between turns ----------

@pytest.mark.parametrize("reason", ["error", "interrupted"])
def test_a_started_call_before_a_turn_end_comes_back_interrupted(reason):
    events = [event("turn/start", goal="write release"), event("step/start", step=1),
              event("step/think", step=1, thought="write", action="write_file",
                    args={"path": "release.txt", "content": "release"}),
              event("tool/call", step=1, call_id="s1", name="write_file",
                    args={"path": "release.txt", "content": "release"}),
              event("turn/end", reason=reason, answer="stopped"),
              event("turn/start", goal="write release", resumed=True), event("step/start", step=2),
              event("step/think", step=2, thought="check", action="read_file", args={}),
              event("turn/end", reason="token_limit", answer="stopped")]
    state = restore_steps(events)
    assert [(s["action"], s["result"]) for s in state.steps] == [("write_file", INTERRUPTED)]
    assert state.steps_done == 2
    assert state.progress.files == {"release.txt": "written"}


def test_the_earlier_session_text_stays_within_its_budget(tmp_path):
    log = SessionLog(tmp_path / "session.jsonl")
    log.append("assistant/message", content="x" * 3000)
    text = Lifecycle(workspace=tmp_path, session_budget=100).conversation(log)
    assert 0 < C.count_tokens(text) <= 100 and text.startswith("Assistant: ")


def test_the_earlier_session_text_until_a_seq_is_the_one_the_turn_started_with(tmp_path):
    class NoSummary:
        context_limit = 1000

        def __call__(self, prompt):
            raise AssertionError("a past view is only read, never compacted")

    log = SessionLog(tmp_path / "session.jsonl")
    log.append("user/message", content="first question")
    before = log.append("assistant/message", content="first answer")["seq"]
    expected = Lifecycle(tmp_path, session_budget=10_000).conversation(log)
    log.append("user/message", content="second question")
    log.append("assistant/message", content="second answer")
    assert Lifecycle(tmp_path, session_budget=10_000).conversation(log, until_seq=before) == expected
    small = Lifecycle(tmp_path, llm=NoSummary(), session_budget=5)
    assert C.count_tokens(small.conversation(log, until_seq=before)) <= 5
