"""Where mini_agent keeps its own files: ~/.mini_agent, or $MINI_AGENT_HOME when set.

  config.env   settings written by `mini_agent setup` (keys, backend, model)
  memory/      long-term memory and daily logs
  skills/      skills you add (folders with a SKILL.md)
"""
import os
from pathlib import Path


def home() -> Path:
    """$MINI_AGENT_HOME, else ~/.mini_agent (on Windows: C:\\Users\\<name>\\.mini_agent)"""
    return Path(os.environ.get("MINI_AGENT_HOME") or Path.home() / ".mini_agent").expanduser()


def config_file() -> Path:
    """The settings file written by `mini_agent setup`"""
    return home() / "config.env"
