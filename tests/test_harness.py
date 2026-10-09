"""Harness behavior under malformed model output, approval policy, and budgets.

A scripted model returns fixed replies, so every path is deterministic.
Run: python tests/test_harness.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mini_agent import Agent, SessionLog
from mini_agent.core.parser import parse_response
from mini_agent.llm.base import BaseLLM


class ScriptedLLM(BaseLLM):
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.model = "scripted"
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def call(self, prompt):
        self.calls += 1
        self.total_prompt_tokens += 10
        self.total_completion_tokens += 5
        return self.replies.pop(0)


def J(action, args=None, thought="t"):
    return json.dumps({"thought": thought, "action": action, "args": args or {}})


def run(replies, **kw):
    llm = ScriptedLLM(replies)
    events = []
    agent = Agent(llm=llm, use_memory=False, skills=[], **kw)
    answer = agent.run("goal", on_event=lambda t, **d: events.append((t, d)))
    return answer, events, llm


def results(events):
    return [d["content"] for t, d in events if t == "tool/result"]


def end(events):
    return [d for t, d in events if t == "turn/end"][-1]


def test_parser_cases():
    assert parse_response(J("calc", {"expr": "1+1"}))[1] == "calc"
    assert parse_response("```json\n" + J("calc") + "\n```")[1] == "calc"
    two = J("calc", {"expr": "1+1"}) + "\n" + J("done", {"answer": "2"})
    assert parse_response(two)[1:] == ("calc", {"expr": "1+1"})
    fabricated = J("calc", {"expr": "947*863"}) + "\nObservation: 1\n" + J("done", {"answer": "1"})
    assert parse_response(fabricated)[1:] == ("calc", {"expr": "947*863"})
    assert parse_response("The answer is 42.")[1] is None
    assert parse_response('I will call {calc}. {"thought": "x", "action": "calc", "args": {"expr": "2"}}')[1] == "calc"
    assert parse_response('{"thought": "x", "action": ["calc"]}')[1] is None
    assert parse_response('{"thought": "x", "action": "done", "args": null}')[2] == {}


def test_two_steps_executes_only_first():
    two = J("calc", {"expr": "947*863"}) + "\n" + J("done", {"answer": "wrong"})
    answer, ev, _ = run([two, J("done", {"answer": "817261"})])
    assert results(ev) == ["817261"], results(ev)
    assert answer == "817261"


def test_self_written_observation_is_discarded():
    fab = J("calc", {"expr": "947*863"}) + "\nObservation: 999999\n" + J("done", {"answer": "999999"})
    answer, ev, _ = run([fab, J("done", {"answer": "817261"})])
    assert results(ev) == ["817261"]
    assert "999999" not in answer


def test_invalid_reply_then_recovery():
    answer, ev, llm = run(["The answer is 42.", J("done", {"answer": "42"})])
    r = results(ev)
    assert len(r) == 1 and r[0].startswith("Error: the reply contains no JSON object")
    assert answer == "42" and end(ev)["reason"] == "done" and llm.calls == 2


def test_two_invalid_replies_stop():
    answer, ev, llm = run(["Error: Authentication Fails (invalid key)"] * 5)
    assert llm.calls == 2
    assert end(ev)["reason"] == "invalid_reply"
    assert "Authentication Fails" in answer


def test_invalid_counter_resets():
    replies = ["bad", J("calc", {"expr": "1+1"}), "bad", J("done", {"answer": "ok"})]
    answer, ev, _ = run(replies)
    assert answer == "ok" and end(ev)["reason"] == "done"


def test_done_with_string_args():
    answer, _, _ = run([J("done", "42")])
    assert answer == "42"


def test_unknown_tool_is_error():
    _, ev, _ = run([J("nonexistent_tool"), J("done", {"answer": "x"})])
    res = [d for t, d in ev if t == "tool/result"]
    assert res[0]["content"].startswith("Error: unknown tool") and res[0]["is_error"]


def test_approval_denies_execution():
    ws = tempfile.mkdtemp()
    target = os.path.join(ws, "out.txt")
    deny_writes = lambda action, args: action not in {"write_file", "bash"}
    _, ev, _ = run([J("write_file", {"path": target, "content": "x"}), J("done", {"answer": "x"})],
                   approve=deny_writes, workspace=ws)
    assert "not approved" in results(ev)[0]
    assert not os.path.exists(target)


def test_max_steps_reason():
    answer, ev, _ = run([J("calc", {"expr": "1+1"})] * 3, max_steps=2)
    assert end(ev)["reason"] == "max_steps" and answer.startswith("Unfinished")


def test_log_projection_includes_parse_error():
    path = os.path.join(tempfile.mkdtemp(), "s.jsonl")
    log = SessionLog(path)
    llm = ScriptedLLM(["no json here", J("done", {"answer": "ok"})])
    Agent(llm=llm, use_memory=False, skills=[]).run("goal", on_event=lambda t, **d: log.append(t, **d))
    msgs = SessionLog(path).derive_messages()
    assert any(m["role"] == "tool" and "no JSON object" in m["content"] for m in msgs)


def test_calc_scientific_notation():
    from mini_agent.tools.math import calc
    assert abs(float(calc("78.4e9 * 11.9e-30 / 1.602176634e-19")) - 5.823078306) < 1e-6
    assert calc("2 * x").startswith("Error: use only numbers")
    assert calc("e").startswith("Error: use only numbers")
    assert calc("...").startswith("Error: use only numbers")
    assert calc("9**9**9").startswith("Error: integer exponents")
    assert abs(float(calc("27**(1/3)")) - 3.0) < 1e-12
    assert calc("__import__('os')").startswith("Error: use only numbers")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
