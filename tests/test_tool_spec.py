"""Tools from plain functions: @tool, Agent(tools=[...]), and the old TOOLS / TOOL_DESC form."""
import json
import os
import sys
from typing import List, Literal, Optional

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mini_agent import Agent, tool
from mini_agent.core.agent.run import run_agent
from mini_agent.llm.base import BaseLLM
from mini_agent.tools import (BUILTIN_SPECS, RISKY, TOOL_DESC, TOOLS, ToolSet, ToolSpec,
                              default_tools, render_tool_desc, spec_from_function)


class Scripted(BaseLLM):
    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []
        self.model = "scripted"

    def call(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def J(action, args=None, thought="t"):
    return json.dumps({"thought": thought, "action": action, "args": args or {}})


@tool
def density(mass_g: float, volume_cm3: float) -> float:
    """Density in g/cm^3.

    Args:
        mass_g: mass in grams
        volume_cm3: volume in cubic
            centimetres
    """
    return mass_g / volume_cm3


def test_builtin_specs_render_exactly_tool_desc():
    # TOOL_DESC is the tutorial's text and stays as it is; remember's optional evidence
    # argument (mini_agent.lifecycle) appears only in the spec (native tool calling)
    assert render_tool_desc(BUILTIN_SPECS.values()) == TOOL_DESC.replace(
        "remember(content)", "remember(content, evidence)")
    assert set(BUILTIN_SPECS) == set(TOOLS)
    assert all(BUILTIN_SPECS[n].fn is TOOLS[n] for n in TOOLS)
    assert {n for n, s in BUILTIN_SPECS.items() if s.risky} == RISKY == {"bash", "run_python", "write_file"}
    assert not BUILTIN_SPECS["read_skill"].listed
    assert [s.name for s in default_tools()] == list(BUILTIN_SPECS)


def test_decorator_keeps_the_function_and_derives_the_schema():
    assert density(7.874, 1.0) == pytest.approx(7.874)
    spec = density.__tool__
    assert spec.name == "density" and spec.description == "Density in g/cm^3."
    assert spec.describe() == "- density(mass_g, volume_cm3): Density in g/cm^3."
    props = spec.parameters["properties"]
    assert props["mass_g"] == {"type": "number", "description": "mass in grams"}
    assert props["volume_cm3"]["description"] == "volume in cubic centimetres"
    assert spec.parameters["required"] == ["mass_g", "volume_cm3"]
    assert spec.parameters["additionalProperties"] is False
    assert spec.openai_schema()["function"]["parameters"] is spec.parameters
    assert spec.anthropic_schema()["input_schema"] is spec.parameters
    json.dumps(spec.openai_schema()), json.dumps(spec.anthropic_schema())


def test_type_hints_map_to_json_schema():
    def f(a: str, b: int, c: bool, d: Optional[int] = None, e: List[str] = (),
          g: Literal["x", "y"] = "x", h=3, i=None, *rest, **more):
        """Summary line
        continues here.

        Longer text.
        """
    spec = spec_from_function(f)
    p = spec.parameters
    assert spec.description == "Summary line continues here."
    assert p["properties"]["a"]["type"] == "string"
    assert p["properties"]["b"]["type"] == "integer"
    assert p["properties"]["c"]["type"] == "boolean"
    assert p["properties"]["d"] == {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None}
    assert p["properties"]["e"] == {"type": "array", "items": {"type": "string"}}
    assert p["properties"]["g"] == {"enum": ["x", "y"], "type": "string", "default": "x"}
    assert p["properties"]["h"] == {"type": "integer", "default": 3}
    assert p["properties"]["i"] == {"type": "string"}           # a None default the type rejects is left out
    assert p["required"] == ["a", "b", "c"]
    assert "rest" not in p["properties"] and "additionalProperties" not in p   # **more allows extras


def test_sphinx_and_numpy_docstrings():
    def s(x, y=2):
        """Sphinx style.

        :param x: the x
        :param int y: the y
        """
    def n(x, y):
        """NumPy style.

        Parameters
        ----------
        x : float
            first value
        y : float
            second value
        """
    ps = spec_from_function(s).parameters["properties"]
    assert ps["x"]["description"] == "the x" and ps["y"]["description"] == "the y"
    pn = spec_from_function(n).parameters["properties"]
    assert pn["x"]["description"] == "first value" and pn["y"]["description"] == "second value"


def test_decorator_options_and_bad_names():
    @tool(name="lookup-v2", description="Look something up", risky=True)
    def lookup(q: str) -> str:
        return q.upper()
    spec = lookup.__tool__
    assert (spec.name, spec.description, spec.risky) == ("lookup-v2", "Look something up", True)
    assert lookup("fe") == "FE"
    with pytest.raises(ValueError):
        spec_from_function(lambda x: x)            # "<lambda>" is not a usable name
    assert spec_from_function(lambda x: x, name="ok").name == "ok"


def test_agent_accepts_a_list_of_functions():
    def word_count(text: str) -> int:
        """Count the words in a text."""
        return len(text.split())
    llm = Scripted(J("density", {"mass_g": 7.874, "volume_cm3": 1.0}),
                   J("word_count", {"text": "a b c"}),
                   J("done", {"answer": "ok"}))
    events = []
    agent = Agent(llm=llm, tools=[density, word_count], use_memory=False, skills=[])
    assert set(agent.tools) == {"density", "word_count"}
    assert all(isinstance(v, ToolSpec) for v in agent.tools.values())
    answer = agent.run("go", verbose=False, on_event=lambda t, **d: events.append((t, d)))
    assert answer == "ok"
    assert "- density(mass_g, volume_cm3): Density in g/cm^3." in llm.prompts[0]
    assert "- word_count(text): Count the words in a text." in llm.prompts[0]
    assert "- calc(" not in llm.prompts[0]
    results = [d["content"] for t, d in events if t == "tool/result"]
    assert results == ["7.874", "3"]
    for t, d in events:                       # every event is plain JSON data
        json.dumps({"type": t, **d})


def test_old_dict_form_keeps_working():
    llm = Scripted(J("calc", {"expr": "2**10"}), J("done", {"answer": "1024"}))
    agent = Agent(llm=llm, use_memory=False, skills=[])
    assert agent.tools is TOOLS and agent.tool_desc == TOOL_DESC
    assert agent.run("go", verbose=False) == "1024"
    assert TOOL_DESC in llm.prompts[0]

    from mini_agent.tools import calc
    custom = {"calc": calc}
    llm = Scripted(J("calc", {"expr": "1+1"}), J("done", {"answer": "2"}))
    agent = Agent(llm=llm, tools=custom, tool_desc="- calc(expr): add", use_memory=False, skills=[])
    assert agent.tools is custom
    agent.run("go", verbose=False)
    assert "- calc(expr): add" in llm.prompts[0]

    # a dict without a description now gets one rendered from its functions
    agent = Agent(llm=Scripted(), tools={"density": density}, use_memory=False, skills=[])
    assert agent.tool_desc.strip() == "- density(mass_g, volume_cm3): Density in g/cm^3."


def test_run_agent_with_a_list_and_no_description():
    llm = Scripted(J("density", {"mass_g": 2, "volume_cm3": 4}), J("done", {"answer": "0.5"}))
    answer = run_agent(llm, [density], None, "go", verbose=False)
    assert answer == "0.5"
    assert "- density(mass_g, volume_cm3)" in llm.prompts[0]


def test_toolset_errors():
    with pytest.raises(ValueError):
        ToolSet.build([density, density])
    with pytest.raises(TypeError):
        ToolSet.build("calc")
    assert len(ToolSet.build([])) == 0
