"""Skills: names and descriptions in context, instructions loaded on request.

Run: python tests/test_skills.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from mini_agent import Agent
from mini_agent.skills import format_skills_prompt, loader, reload_skills
from mini_agent.tools import TOOLS, TOOL_DESC
from test_harness import J, ScriptedLLM

SKILL = """---
name: result-format
description: Required format for reporting any numerical result
---
Report every numerical result as three lines: Value, Unit, Formula.
"""


def with_skill_dir():
    d = Path(tempfile.mkdtemp())
    (d / "result-format").mkdir()
    (d / "result-format" / "SKILL.md").write_text(SKILL)
    loader.SKILL_DIRS[:] = [d]
    return reload_skills()


def test_context_has_description_not_instructions():
    assert with_skill_dir() == ["result-format"]
    section = format_skills_prompt(["result-format"])
    assert "result-format: Required format" in section
    assert "Value, Unit, Formula" not in section
    assert "read_skill" in section


def test_read_skill_tool():
    with_skill_dir()
    assert "Value, Unit, Formula" in TOOLS["read_skill"]("result-format")
    assert TOOLS["read_skill"]("missing").startswith("Error: no skill named")
    assert "read_skill" not in TOOL_DESC


def test_agent_loads_skill_on_request():
    with_skill_dir()
    llm = ScriptedLLM([J("read_skill", {"name": "result-format"}), J("done", {"answer": "ok"})])
    ev = []
    Agent(llm=llm, use_memory=False, skills=["result-format"]).run(
        "goal", on_event=lambda t, **d: ev.append((t, d)))
    res = [d["content"] for t, d in ev if t == "tool/result"]
    assert "Value, Unit, Formula" in res[0]


def test_no_skills_no_section():
    loader.SKILL_DIRS[:] = [Path(tempfile.mkdtemp())]
    reload_skills()
    assert format_skills_prompt(None) == ""


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
