"""The Agent class"""
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..config import AgentConfig, get_workspace_str
from .run import run_agent
from ...llm import create_llm
from ...tools import TOOLS, TOOL_DESC
from ...mem import Memory
from ...skills import format_skills_prompt, list_skills
from ...tools import workspace as _workspace


class Agent:
    """Minimal real agent"""

    def __init__(
        self,
        name: str = "agent",
        workspace: str | Path = None,
        llm: Callable[[str], str] = None,
        tools: Dict[str, Callable] = None,
        tool_desc: str = None,
        model: str = None,
        max_tokens: int = None,  # None = derive from the LLM
        max_steps: int = 100,
        skills: Optional[List[str]] = None,  # None = all, [] = none
        use_memory: bool = True,
        prompt_name: str = "agent",  # prompts/<name>.md (e.g. "coding")
        approve: Optional[Callable[[str, dict], bool]] = None,  # policy check per tool call
    ):
        self.llm = llm or create_llm()  # auto-load config from .env

        # max_tokens: user override first, else from the LLM
        if max_tokens is None:
            max_tokens = getattr(self.llm, 'context_limit', 128_000)

        self.config = AgentConfig(name, workspace, model, max_tokens, max_steps)
        self.tools = tools or TOOLS
        self.tool_desc = tool_desc or TOOL_DESC

        # Skills
        self._skills = skills  # None=all, []=none, ['a','b']=specific
        self._skills_prompt = ""
        self._load_skills()

        # Memory
        self.prompt_name = prompt_name
        self.use_memory = use_memory
        self.memory = Memory() if use_memory else None
        self.approve = approve

    def _load_skills(self):
        """Load skills"""
        if self._skills is not None and len(self._skills) == 0:
            # explicitly disabled
            self._skills_prompt = ""
        else:
            self._skills_prompt = format_skills_prompt(self._skills)

    def _get_memory_prompt(self) -> str:
        """Assemble memory content"""
        if not self.memory:
            return "(none)"

        parts = []

        # long-term memory
        long_term = self.memory.get_long_term()
        if long_term:
            parts.append(f"### long-term memory\n{long_term}")

        # today's log
        today = self.memory.get_today_log()
        if today:
            parts.append(f"### Today's log\n{today}")

        return "\n\n".join(parts) if parts else "(none)"

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def workspace(self) -> Path:
        return self.config.workspace

    def run(self, goal: str, verbose: bool = True, on_event=None, conversation: str = "") -> str:
        """Run a goal. on_event(type, **data) receives the event stream (None = console output)."""
        workspace_str = get_workspace_str(self.config.workspace)
        context = f"Workspace: {workspace_str}\n\n{self.tool_desc}"
        try:
            self.config.workspace.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # a tool that uses the path reports the problem as data
        token = _workspace.current.set(self.config.workspace)  # relative tool paths start here
        try:
            return self._run(goal, context, verbose, on_event, conversation)
        finally:
            _workspace.current.reset(token)

    def _run(self, goal, context, verbose, on_event, conversation) -> str:
        return run_agent(
            llm=self.llm,
            tools=self.tools,
            tool_desc=context,
            goal=goal,
            max_steps=self.config.max_steps,
            max_tokens=self.config.max_tokens,
            name=self.name,
            verbose=verbose,
            skills_prompt=self._skills_prompt,
            memory_prompt=self._get_memory_prompt(),
            on_event=on_event,
            conversation=conversation,
            prompt_name=self.prompt_name,
            approve=self.approve
        )

    def log(self, content: str):
        """Append a note to today's log"""
        if self.memory:
            self.memory.append_log(content)

    def remember(self, content: str):
        """Add to long-term memory"""
        if self.memory:
            self.memory.append_memory(content)
