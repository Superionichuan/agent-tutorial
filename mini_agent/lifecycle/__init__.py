"""Lifecycle: what lets one agent keep working on a long task.

Written memory in <workspace>/.mini_agent/ (user.md: every user message verbatim;
handoff.md: where the work stands), notes with evidence (remember(content, evidence)),
and compaction of the steps before each model call: mechanical first, the model only
when that is not enough. The core loop calls it through small hooks; see README.md.

    from mini_agent import Agent
    from mini_agent.lifecycle import Lifecycle

    agent = Agent(lifecycle=Lifecycle(compact_at=0.5, keep_steps=6))   # or lifecycle=True

restore_steps(events) rebuilds an unfinished turn from the session log (mini_agent -c).
"""
from .compact import count_tokens
from .lifecycle import INTERRUPTED, Lifecycle, Progress, Restored, restore_steps

__all__ = ["Lifecycle", "restore_steps", "Restored", "Progress", "count_tokens", "INTERRUPTED"]
