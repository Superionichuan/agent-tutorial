"""
Mini Agent — a minimal real agent

Layout (one concern per package):
    core/        the ReAct loop: think -> act -> observe
    llm/         pluggable LLM backends (.env configured)
    tools/       tool registry (math / file / shell / memory / todo)
    session.py   append-only event log: projections, fork, replay, compaction
    mem/         durable memory (long-term + daily log)
    skills/      skill plugins (SKILL.md)
    ui/          plain REPL + textual TUI
    cli/         entry points

Usage:
    from mini_agent import Agent
    agent = Agent()
    result = agent.run("calculate 1+1")
"""
__version__ = "1.2.1"

from .core import Agent, State
from .session import SessionLog, new_call_id
from .tools import TOOLS, TOOL_DESC, calc, read_file, write_file, list_dir, bash, run_python
from .llm import BaseLLM, create_llm
from .mem import Memory, get_long_term, get_today_log, append_log, append_memory
from .skills import Skill, load_skills, get_skill, list_skills

__all__ = [
    "Agent", "State",
    "SessionLog", "new_call_id",
    "TOOLS", "TOOL_DESC",
    "calc", "read_file", "write_file", "list_dir", "bash", "run_python",
    "BaseLLM", "create_llm",
    "Memory", "get_long_term", "get_today_log", "append_log", "append_memory",
    "Skill", "load_skills", "get_skill", "list_skills",
]
