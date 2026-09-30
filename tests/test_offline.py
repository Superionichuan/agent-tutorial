"""Offline mode: record a run with a scripted model, replay it without one.

Run: python tests/test_offline.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mini_agent import Agent
from mini_agent.llm import OfflineLLM, RecordingLLM
from mini_agent.llm.offline import NO_RECORDING, describe, normalize

sys.path.insert(0, os.path.dirname(__file__))
from test_harness import J, ScriptedLLM

GOAL = "Compute 947*863 and report it."
REPLIES = [J("calc", {"expr": "947*863"}), J("done", {"answer": "817261"})]


def record(path, workspace):
    rec = RecordingLLM(ScriptedLLM(REPLIES), path)
    ev = []
    ans = Agent(llm=rec, use_memory=False, skills=[], workspace=workspace).run(
        GOAL, on_event=lambda t, **d: ev.append((t, d)))
    return ans, ev, rec


def replay(path, workspace):
    OfflineLLM._cursor.clear()
    off = OfflineLLM(path)
    ev = []
    ans = Agent(llm=off, use_memory=False, skills=[], workspace=workspace).run(
        GOAL, on_event=lambda t, **d: ev.append((t, d)))
    return ans, ev, off


def strip(ev):
    # streaming chunks depend on the backend's chunking, not on the run itself
    return [(t, d) for t, d in ev if t != "assistant/chunk"]


def test_record_then_replay_identical():
    path = os.path.join(tempfile.mkdtemp(), "rec.jsonl")
    a1, e1, rec = record(path, "/tmp/ws_record_machine")
    a2, e2, off = replay(path, "/content/other_machine_ws")   # different workspace path
    assert a1 == a2 == "817261"
    assert strip(e1) == strip(e2)
    assert (off.total_prompt_tokens, off.total_completion_tokens) == (rec.total_prompt_tokens, rec.total_completion_tokens)


def test_recording_stores_no_full_prompt():
    path = os.path.join(tempfile.mkdtemp(), "rec.jsonl")
    record(path, "/tmp/ws_private_path")
    raw = open(path).read()
    assert "ws_private_path" not in raw and "[Tools]" not in raw
    rows = [json.loads(x) for x in raw.splitlines()]
    assert set(rows[0]) >= {"key", "goal", "steps", "history", "response"}


def test_normalize_masks_machine_and_day():
    p1 = ("[Goal]\ng\n\n[Tools]\nWorkspace: /home/a/x\n\n[Memory]\n### Today's log\n- 09:12 note\n\n"
          "[Steps taken so far]\n(none)\n\n[Instructions]\nx")
    p2 = p1.replace("/home/a/x", "/content").replace("- 09:12 note", "(none)")
    assert normalize(p1) == normalize(p2)
    assert describe(p1)["key"] == describe(p2)["key"]


def test_fallback_on_small_history_difference():
    path = os.path.join(tempfile.mkdtemp(), "rec.jsonl")
    record(path, "/tmp/ws")
    rows = [json.loads(x) for x in open(path)]
    for r in rows:                      # simulate a slightly different tool output
        r["key"] = "0" * 24
    open(path, "w").write("".join(json.dumps(r) + "\n" for r in rows))
    ans, _, _ = replay(path, "/tmp/ws")
    assert ans == "817261"


def test_unrecorded_prompt_is_reported():
    path = os.path.join(tempfile.mkdtemp(), "rec.jsonl")
    record(path, "/tmp/ws")
    off = OfflineLLM(path)
    assert off("A question that was never recorded.") == NO_RECORDING
    ev = []
    ans = Agent(llm=off, use_memory=False, skills=[]).run(
        "A different goal", on_event=lambda t, **d: ev.append((t, d)))
    assert "offline mode" in ans and [d for t, d in ev if t == "turn/end"][0]["reason"] == "invalid_reply"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
