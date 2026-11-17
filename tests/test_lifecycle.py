"""Lifecycle: written memory, compaction inside a turn, handoff, continuing a killed turn.

A scripted model drives long tasks with a small context window, so that the steps are
compacted several times in one turn; every prompt it receives is checked.
Run: python -m pytest tests/test_lifecycle.py
"""
import json
import os
import subprocess

import pytest
import sys
import textwrap
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from mini_agent import Agent, SessionLog  # noqa: E402
from mini_agent.core.agent import run_agent  # noqa: E402
from mini_agent.lifecycle import INTERRUPTED, Lifecycle, count_tokens, restore_steps  # noqa: E402
from mini_agent.lifecycle import compact as C  # noqa: E402
from mini_agent.lifecycle import notes as N  # noqa: E402
from mini_agent.llm.base import BaseLLM, LLMError  # noqa: E402
from mini_agent.tools import default_tools, remember  # noqa: E402
from mini_agent.mem import memory as _memory  # noqa: E402


def blob(n: int) -> str:
    """A long block of text (stands for a big file or a long command output).

    Args:
        n: which block
    """
    return f"block {n} begins\n" + ("lorem ipsum dolor sit amet " * 60) + f"\nblock {n} ends"


class Scripted(BaseLLM):
    """Replies from a script, one per agent step; answers summary requests itself."""

    def __init__(self, script, window=6000, check=None, summary=None):
        self.script = list(script)
        self.context_limit = window
        self.check = check
        self.summary = summary          # None: a short summary; a function: its reply
        self.prompts, self.summaries = [], 0
        self.model = "scripted"
        self.total_prompt_tokens = self.total_completion_tokens = 0

    def call(self, prompt):
        self.total_prompt_tokens += count_tokens(prompt)      # usage as a provider reports it
        self.total_completion_tokens += 10
        if prompt.startswith(C.SUMMARY_PROMPT):
            self.summaries += 1
            if self.summary is not None:
                return self.summary(prompt)
            return f"- summary {self.summaries}: blocks were read; the key is in the notes"
        self.prompts.append(prompt)
        if self.check is not None:
            self.check(prompt)
        action, args = self.script.pop(0)
        return json.dumps({"thought": f"step {len(self.prompts)}", "action": action, "args": args})


def agent_for(tmp_path, llm, **lc):
    lifecycle = Lifecycle(**{"keep_steps": 2, **lc})
    # max_tokens: the turn's spending budget (Agent's default is the window, too small here)
    return Agent(llm=llm, workspace=tmp_path / "ws", tools=default_tools() + [blob], skills=[],
                 lifecycle=lifecycle, max_steps=200, max_tokens=10**9)


def home(monkeypatch, tmp_path):
    monkeypatch.setenv("MINI_AGENT_HOME", str(tmp_path / "home"))


def kinds(events, kind):
    return [d for t, d in events if t == kind]


# ---------- the long task (the acceptance test) ----------

def test_long_task_compacts_several_times_and_keeps_messages_notes_and_handoff(tmp_path, monkeypatch):
    home(monkeypatch, tmp_path)
    goals = ["Read blocks 1 to 40 and report the key; keep a note of the key with its evidence.",
             "Which block held the key? Answer in one sentence."]
    note = ("The key is 4217", "block 3, line 2: key=4217")
    events, seen = [], {"goals": [], "note": False, "turn": 0}

    def on_event(t, **d):
        json.dumps(d)                                   # every event can be logged
        events.append((t, d))
        if t == "tool/result" and "with evidence" in d["content"]:
            seen["note"] = True

    def check(prompt):
        for goal in seen["goals"]:                      # every user message, verbatim
            assert goal in prompt
        if seen["note"]:                                # the note and its evidence
            assert note[0] in prompt and note[1] in prompt
        compacted = [d for t, d in events if t == "context/compact"]
        if seen["turn"] == 1 and compacted:             # after a compaction: the handoff
            assert "### Handoff" in prompt and "Status: open" in prompt
        if seen["turn"] == 2:                           # turn 1 finished: its handoff is closed
            assert "### Handoff" not in prompt

    script = [("todo_write", {"todos": [{"content": "read the blocks", "status": "in_progress"},
                                        {"content": "report the key", "status": "pending"}]})]
    script += [("blob", {"n": i}) for i in range(1, 4)]
    script += [("remember", {"content": note[0], "evidence": note[1]})]
    script += [("blob", {"n": i}) for i in range(4, 41)]
    script += [("done", {"answer": "The key is 4217."})]
    script += [("done", {"answer": "Block 3 held the key."})]
    llm = Scripted(script, window=6000, check=check)
    agent = agent_for(tmp_path, llm)

    seen["goals"].append(goals[0]); seen["turn"] = 1
    assert agent.run(goals[0], on_event=on_event) == "The key is 4217."
    compactions = kinds(events, "context/compact")
    assert len(compactions) >= 3, len(compactions)
    lc = agent.lifecycle
    for c in compactions:                               # each one brought the prompt under the trigger
        assert c["after_tokens"] <= lc.trigger < c["before_tokens"]
    assert kinds(events, "turn/end")[-1]["reason"] == "done"

    seen["goals"].append(goals[1]); seen["turn"] = 2
    assert agent.run(goals[1], on_event=on_event) == "Block 3 held the key."
    assert llm.script == []

    folder = tmp_path / "ws" / ".mini_agent"
    assert [text for _, _, text in N.user_messages(folder / "user.md")] == goals
    assert (folder / ".gitignore").read_text() == "*\n"
    handoff = (folder / "handoff.md").read_text(encoding="utf-8")
    assert handoff.startswith("# Handoff\nStatus: closed")


def test_mechanical_passes_first_the_model_only_when_they_are_not_enough(tmp_path, monkeypatch):
    home(monkeypatch, tmp_path)
    # Roomy window: trimming and folding the old steps are enough, no summary call.
    script = [("blob", {"n": i}) for i in range(1, 41)] + [("done", {"answer": "ok"})]
    events = []
    llm = Scripted(script, window=12000)
    agent_for(tmp_path, llm).run("read the blocks", on_event=lambda t, **d: events.append((t, d)))
    compactions = kinds(events, "context/compact")
    assert len(compactions) >= 3 and llm.summaries == 0
    assert {d["method"] for d in compactions} == {"mechanical"}
    assert all(d["at"] > 0 for d in compactions[1:])   # steps folded before are not sent again

    # Tight window: folding is not enough, so the model summarises; the summary is kept.
    script = [("blob", {"n": i}) for i in range(1, 60)] + [("done", {"answer": "ok"})]
    events = []
    llm = Scripted(script, window=3000)
    agent_for(tmp_path / "b", llm).run("read the blocks", on_event=lambda t, **d: events.append((t, d)))
    llm_compactions = [d for d in kinds(events, "context/compact") if d["method"] == "llm"]
    assert llm.summaries >= 1 and llm_compactions
    assert any(s.get("compacted") and s["result"].startswith("- summary")
               for d in llm_compactions for s in d["steps"])


def test_a_summary_that_is_not_shorter_is_refused(tmp_path, monkeypatch):
    home(monkeypatch, tmp_path)
    script = [("blob", {"n": i}) for i in range(1, 60)] + [("done", {"answer": "ok"})]
    events = []
    llm = Scripted(script, window=3000, summary=lambda prompt: prompt * 2)   # longer than its source
    answer = agent_for(tmp_path, llm).run("read", on_event=lambda t, **d: events.append((t, d)))
    assert answer == "ok" and llm.summaries >= 1
    dropped = [s for d in kinds(events, "context/compact") for s in d["steps"] if s.get("compacted")]
    assert dropped and all("folded away" in s["result"] for s in dropped)


def test_the_last_steps_are_never_cut(tmp_path, monkeypatch):
    home(monkeypatch, tmp_path)
    keep = 3
    script = [("blob", {"n": i}) for i in range(1, 30)] + [("done", {"answer": "ok"})]

    def check(prompt):                                   # the newest blocks always in full
        done = len(llm.prompts) - 1
        for n in range(max(1, done - keep + 1), done + 1):
            assert blob(n) in prompt

    llm = Scripted(script, window=6000, check=check)
    agent_for(tmp_path, llm, keep_steps=keep).run("read")
    assert llm.script == []


def test_token_limit_is_a_spending_budget_compaction_does_not_reset_it(tmp_path, monkeypatch):
    home(monkeypatch, tmp_path)
    script = [("blob", {"n": i}) for i in range(1, 80)] + [("done", {"answer": "ok"})]
    events = []
    llm = Scripted(script, window=6000)
    agent = agent_for(tmp_path, llm)
    agent.config.max_tokens = 40_000                   # the turn's budget, summed over all calls
    agent.run("read", on_event=lambda t, **d: events.append((t, d)))
    assert kinds(events, "context/compact")            # compaction happened, the budget still stopped it
    assert kinds(events, "turn/end")[-1]["reason"] == "token_limit"
    handoff = (tmp_path / "ws" / ".mini_agent" / "handoff.md").read_text(encoding="utf-8")
    assert "Status: open · stopped: token_limit" in handoff and "## Done" in handoff


# ---------- written memory ----------

def test_user_messages_are_appended_verbatim_and_not_twice_on_resume(tmp_path):
    lc = Lifecycle(tmp_path)
    lc.attach(lambda *a, **k: None, None)
    messages = ["first message\nwith two lines", "## 3 · not a header, just text", "第三条：中文"]
    for text in messages:
        lc.attach(lambda *a, **k: None, None)
        memory = lc.start(text, "")
    assert [t for _, _, t in N.user_messages(tmp_path / ".mini_agent" / "user.md")] == messages
    assert messages[0] in memory and messages[1] in memory and "第三条" not in memory   # current = [Goal]
    lc.attach(lambda *a, **k: None, None, resumed=True)
    lc.start(messages[-1], "")
    assert len(N.user_messages(tmp_path / ".mini_agent" / "user.md")) == 3


def test_remember_without_evidence_is_unchanged_and_with_evidence_is_a_fact(tmp_path, monkeypatch):
    home(monkeypatch, tmp_path)
    mem = _memory.Memory()
    token = _memory.current.set(mem)
    try:
        assert remember("plain note") == "Remembered: plain note"
        assert remember("checked", evidence="pytest -q\n3 passed\n\nexit 0") == \
            "Remembered: checked (with evidence)"
        assert remember("also plain", evidence="") == "Remembered: also plain"
        text = mem.long_term_file.read_text(encoding="utf-8")
        assert "] plain note\n" in text
        facts = N.evidence_notes(text)
        assert len(facts) == 1 and facts[0].endswith(
            "] checked\n  evidence: pytest -q\n    3 passed\n    \n    exit 0")
        lc = Lifecycle(tmp_path)
        lc.attach(lambda *a, **k: None, None)
        shown = lc.start("goal", "(none)")
        assert facts[0] in shown and "plain note" not in shown      # facts only, in full
        both = lc.start("goal", "### long-term memory\n" + text)
        assert both.count(facts[0]) == 1                            # not twice when memory has it
    finally:
        _memory.current.reset(token)


def test_handoff_is_written_at_every_stop_but_done_and_shown_until_closed(tmp_path, monkeypatch):
    home(monkeypatch, tmp_path)
    events = []
    llm = Scripted([("write_file", {"path": "a.txt", "content": "x"}),
                    ("bash", {"cmd": "echo hi"})], window=60_000)
    agent = agent_for(tmp_path, llm)
    agent.config.max_steps = 2
    agent.approve = None
    agent.run("make a.txt then say hi", on_event=lambda t, **d: events.append((t, d)))
    assert kinds(events, "turn/end")[-1]["reason"] == "max_steps"
    path = tmp_path / "ws" / ".mini_agent" / "handoff.md"
    handoff = path.read_text(encoding="utf-8")
    assert "Status: open · stopped: max_steps" in handoff
    assert "written: a.txt" in handoff and "commands: echo hi" in handoff
    assert "make a.txt then say hi" in handoff

    llm.script = [("done", {"answer": "done now"})]
    llm.check = lambda prompt: (_ for _ in ()).throw(AssertionError("no handoff")) \
        if "Status: open · stopped: max_steps" not in prompt else None
    agent.run("continue", on_event=lambda t, **d: events.append((t, d)))
    assert path.read_text(encoding="utf-8").startswith("# Handoff\nStatus: closed")

    llm.script, llm.check = [("done", {"answer": "x"})], None
    agent.run("next", on_event=lambda t, **d: events.append((t, d)))
    assert "### Handoff" not in llm.prompts[-1]


def test_a_crash_inside_the_run_still_leaves_a_handoff(tmp_path, monkeypatch):
    home(monkeypatch, tmp_path)

    class Broken(Scripted):
        def call(self, prompt):
            if len(self.prompts) == 2:
                raise RuntimeError("network fell over")
            return super().call(prompt)

    llm = Broken([("blob", {"n": 1}), ("blob", {"n": 2})], window=60_000)
    try:
        agent_for(tmp_path, llm).run("read", on_event=lambda t, **d: None)
    except RuntimeError:
        pass
    handoff = (tmp_path / "ws" / ".mini_agent" / "handoff.md").read_text(encoding="utf-8")
    assert "Status: open · stopped: error" in handoff and "network fell over" in handoff


# ---------- the mechanical passes ----------

def test_trim_and_fold_change_a_step_once():
    step = {"thought": "t" * 1000, "action": "read_file", "args": {"path": "a", "x": "y" * 2000},
            "result": "line one\n" + "z" * 5000}
    once = C.trim(step)
    assert C.trim(once) == once and len(once["result"]) < 900 and len(once["args"]["x"]) < 450
    folded = C.fold(step)
    assert C.fold(folded) == folded and C.trim(folded) == folded
    assert folded["result"].startswith("line one") and "\n" not in folded["result"]
    summary = C.compacted("text", 5)
    assert C.covered([summary, folded, step]) == 7


def test_streamed_replies_pass_through(tmp_path, monkeypatch):
    """A streaming backend emits assistant/chunk events with a field named kind."""
    home(monkeypatch, tmp_path)

    class Streaming(Scripted):
        def call_stream(self, prompt, on_chunk):
            text = self.call(prompt)
            on_chunk("thinking about it", kind="reasoning")
            on_chunk(text, kind="content")
            return text

    events = []
    script = [("blob", {"n": i}) for i in range(1, 20)] + [("done", {"answer": "ok"})]
    llm = Streaming(script, window=6000)
    assert agent_for(tmp_path, llm).run("read", on_event=lambda t, **d: events.append((t, d))) == "ok"
    chunks = kinds(events, "assistant/chunk")
    assert {d["kind"] for d in chunks} == {"reasoning", "content"}
    assert kinds(events, "context/compact")


def test_lifecycle_none_sends_the_same_prompts(tmp_path, monkeypatch):
    home(monkeypatch, tmp_path)
    script = [("blob", {"n": 1}), ("calc", {"expr": "1+1"}), ("done", {"answer": "2"})]
    plain = Scripted(script)
    run_agent(plain, default_tools() + [blob], None, "goal", verbose=False, memory_prompt="(none)")
    again = Scripted(script)
    run_agent(again, default_tools() + [blob], None, "goal", verbose=False, memory_prompt="(none)",
              lifecycle=None, state=None)
    assert plain.prompts == again.prompts and len(plain.prompts) == 3


# ---------- continuing a turn from the session log ----------

def test_restore_steps_follows_the_rules_of_the_loop():
    def ev(kind, **data):
        return {"type": kind, "data": data}

    events = [
        ev("turn/start", goal="g", name="agent"),
        ev("step/start", step=1),
        ev("step/think", step=1, thought="a", action="calc", args={"expr": "1+1"}),
        ev("tool/call", step=1, call_id="s1", name="calc", args={"expr": "1+1"}),
        ev("tool/result", step=1, call_id="s1", content="2", is_error=False),
        ev("step/start", step=2),
        ev("step/think", step=2, thought="b", action=None, args={"error": "no JSON"}),
        ev("tool/result", step=2, call_id="s2", content="Error: no JSON.", is_error=True),
        ev("step/start", step=3),
        ev("step/think", step=3, thought="c", action="bash", args={"cmd": "sleep 9"}),
        ev("tool/call", step=3, call_id="s3", name="bash", args={"cmd": "sleep 9"}),
        # killed here; continued:
        ev("turn/start", goal="g", name="agent", resumed=True),
        ev("step/start", step=4),
        ev("context/compact", step=4, before_tokens=9, after_tokens=5, method="mechanical",
           at=0, replaced=2, steps=[C.compacted("two steps", 2)]),
        ev("step/think", step=4, thought="d", action="calc", args={"expr": "2+2"}),
        ev("turn/end", answer="Token limit", reason="token_limit"),      # d never ran
        ev("turn/start", goal="g", name="agent", resumed=True),
    ]
    state = restore_steps(events)
    assert state.goal == "g" and state.steps_done == 4
    assert [s["action"] for s in state.steps] == ["compacted", "bash"]
    assert state.steps[0]["result"] == "two steps"
    assert state.steps[1] == {"thought": "c", "action": "bash", "args": {"cmd": "sleep 9"},
                              "result": INTERRUPTED}
    before = restore_steps(events[:8])
    assert before.steps[1] == {"thought": "b", "action": "(invalid reply)", "args": {},
                               "result": "Error: no JSON."}


KILLED_RUN = textwrap.dedent("""
    import os, sys
    from pathlib import Path
    sys.path.insert(0, {repo!r}); sys.path.insert(0, {tests!r})
    import test_lifecycle as T
    from mini_agent import SessionLog

    log = SessionLog({log!r})
    compactions, calls = [], [0]       # agent calls since the last compaction

    class KilledAfterCompaction(T.Scripted):
        def call(self, prompt):
            if not prompt.startswith(T.C.SUMMARY_PROMPT):
                if len(compactions) >= {after} and calls[0] >= {calls}:
                    Path({out!r}).write_text(prompt, encoding="utf-8")
                    os._exit(137)      # like kill -9: no handler, no handoff, nothing flushed after
                calls[0] += 1
            return super().call(prompt)

    def on_event(t, **d):
        log.append(t, **d)
        if t == "context/compact":
            compactions.append(d)
            if len(compactions) <= {after}:
                calls[0] = 0           # counted from the compaction the kill waits for

    llm = KilledAfterCompaction(T.KILL_SCRIPT, window=T.KILL_WINDOW)
    agent = T.agent_for(Path({tmp!r}), llm)
    log.append("user/message", content=T.KILL_GOAL)
    agent.run(T.KILL_GOAL, on_event=on_event)
    """)
KILL_GOAL = "Read blocks 1 to 30, keep a note of the key, then report it."
KILL_SCRIPT = ([("blob", {"n": i}) for i in range(1, 4)]
               + [("remember", {"content": "The key is 4217", "evidence": "block 3: key=4217"})]
               + [("blob", {"n": i}) for i in range(4, 31)] + [("done", {"answer": "4217"})])
KILL_WINDOW = 6000


@pytest.mark.parametrize("after,calls", [(0, 3), (1, 0), (2, 0), (2, 3)])
def test_compact_kill_continue_sends_the_prompt_it_would_have_sent(tmp_path, monkeypatch, after, calls):
    """Killed during a model call (no compaction yet / right after one / some calls after one):
    rebuilt from the log, the turn sends the prompt it was sending, byte for byte."""
    home(monkeypatch, tmp_path)
    log_path, out = tmp_path / "session.jsonl", tmp_path / "would_have_sent.txt"
    script = tmp_path / "killed_run.py"
    script.write_text(KILLED_RUN.format(repo=os.path.join(HERE, ".."), tests=HERE, log=str(log_path),
                                        out=str(out), tmp=str(tmp_path), after=after, calls=calls),
                      encoding="utf-8")
    env = {**os.environ, "MINI_AGENT_HOME": str(tmp_path / "home")}
    r = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 137, r.stderr[-2000:]
    would_have_sent = out.read_text(encoding="utf-8")

    log = SessionLog(log_path)
    unfinished = log.unfinished_turn()
    assert unfinished is not None and unfinished["reason"] == "interrupted"
    compactions = [e["data"] for e in unfinished["events"] if e["type"] == "context/compact"]
    assert len(compactions) >= after and (after or not compactions)
    state = restore_steps(unfinished["events"])
    steps_before = len([e for e in unfinished["events"] if e["type"] == "tool/result"])
    assert state.steps_done == max(e["data"]["step"] for e in unfinished["events"]
                                   if isinstance(e["data"].get("step"), int))
    if after:
        assert len(state.steps) < steps_before                # the compacted context, not the raw steps
    else:
        assert len(state.steps) == steps_before

    llm = Scripted([("done", {"answer": "4217"})], window=KILL_WINDOW)
    agent = agent_for(tmp_path, llm)
    answer = agent.run(unfinished["goal"], on_event=lambda t, **d: log.append(t, **d), state=state)
    assert answer == "4217"
    assert llm.prompts[0] == would_have_sent                  # the same context, byte for byte
    assert log.unfinished_turn() is None
    resumed = [e for e in log.events() if e["type"] == "turn/start"][-1]
    assert resumed["data"].get("resumed") is True
    first = [e for e in log.events() if e["type"] == "step/start"][-1]
    assert first["data"]["step"] == state.steps_done + 1      # numbering goes on
    user_md = tmp_path / "ws" / ".mini_agent" / "user.md"
    assert [t for _, _, t in N.user_messages(user_md)] == [KILL_GOAL]


# ---------- between turns ----------

def test_conversation_is_token_based_and_summarises_only_when_over_budget(tmp_path):
    class Summ(BaseLLM):
        context_limit = 10_000
        calls = 0

        def call(self, prompt):
            Summ.calls += 1
            return "- they talked about blocks"

    log = SessionLog(tmp_path / "s.jsonl")
    for i in range(5):
        log.append("user/message", content=f"short message {i}")
    lc = Lifecycle(tmp_path, llm=Summ(), session_budget=0.1)        # 1,000 tokens
    text = lc.conversation(log)
    assert Summ.calls == 0 and text.count("User: short message") == 5
    for i in range(40):
        log.append("user/message", content=f"long message {i} " + "words " * 60)
    text = lc.conversation(log)
    assert Summ.calls == 1 and count_tokens(text) <= 1000
    assert "they talked about blocks" in text or "long message 39" in text


def test_llm_errors_in_a_summary_fall_back_to_the_pointer(tmp_path, monkeypatch):
    home(monkeypatch, tmp_path)
    script = [("blob", {"n": i}) for i in range(1, 60)] + [("done", {"answer": "ok"})]
    events = []
    llm = Scripted(script, window=3000, summary=lambda prompt: LLMError("Error: 503"))
    assert agent_for(tmp_path, llm).run("read", on_event=lambda t, **d: events.append((t, d))) == "ok"
    assert any("folded away" in s["result"] for d in kinds(events, "context/compact")
               for s in d["steps"] if s.get("compacted"))
