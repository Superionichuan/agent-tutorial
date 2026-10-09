"""Local deterministic fixtures; no network or supplied-file changes."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # demo_build: tests/referee/ -> package root
sys.path.insert(0, str(ROOT))

from mini_agent.core.agent.run import run_agent
from mini_agent.llm.base import BaseLLM
from mini_agent.llm.offline import describe


class Scripted(BaseLLM):
    def __init__(self, *replies):
        self.replies = iter(replies)
        self.calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def call(self, prompt):
        self.calls += 1
        self.total_prompt_tokens += 1
        return next(self.replies)


def reply(action, args, thought="thinking"):
    return json.dumps(dict(action=action, args=args, thought=thought))


def run(llm, **kwargs):
    events = []
    options = dict(llm=llm, tools={}, tool_desc="", goal="fixture goal",
                   verbose=False, max_steps=4,
                   on_event=lambda kind, **data: events.append((kind, data)))
    options.update(kwargs)
    answer = run_agent(**options)
    return answer, events


def recording(path, prompt, *responses):
    rows = [dict(describe(prompt), response=response) for response in responses]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def skill(root, body="old instructions", name="demo"):
    directory = root / name
    directory.mkdir()
    path = directory / "SKILL.md"
    path.write_text(f"---\nname: {name}\ndescription: demonstration\n---\n{body}\n")
    return path
