"""The Agent class"""
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..config import AgentConfig, get_workspace_str
from .run import run_agent
from ...llm import create_llm
from ...tools import TOOLS, TOOL_DESC
from ...tools.spec import ToolSet
from ..policy import Approval
from ...mem import Memory
from ...mem import memory as _memory
from ...skills import format_skills_prompt, list_skills
from ...tools import workspace as _workspace
from ...tools.shell.bash import describe_system


class Agent:
    """Minimal real agent"""

    def __init__(
        self,
        name: str = "agent",
        workspace: str | Path = None,
        llm: Callable[[str], str] = None,
        tools=None,  # list of functions / ToolSpecs, or dict name -> function; None = built-ins
        tool_desc: str = None,  # None = rendered from the tools
        model: str = None,
        max_tokens: int = None,  # None = derive from the LLM
        max_steps: int = 100,
        skills: Optional[List[str]] = None,  # None = all, [] = none
        use_memory: bool = True,
        prompt_name: str = "agent",  # prompts/<name>.md (e.g. "coding")
        approve=None,  # None / "auto": all run; "ask": confirm risky tools; "deny"; or a function
        confine: bool = True,  # file tools only reach paths inside the workspace
        lifecycle=None,  # a mini_agent.lifecycle.Lifecycle (True = defaults): memory, compaction
    ):
        self.llm = llm or create_llm()  # auto-load config from .env

        # max_tokens: user override first, else from the LLM
        if max_tokens is None:
            max_tokens = getattr(self.llm, 'context_limit', 128_000)

        self.config = AgentConfig(name, workspace, model, max_tokens, max_steps)
        if tools is None or (isinstance(tools, dict) and not tools):
            tools = TOOLS                      # as before: no tools given = the built-ins
        toolset = ToolSet.build(tools, tool_desc or None)
        # a dict given (or TOOLS) stays the tool table, as before;
        # a list becomes name -> ToolSpec (callable like the function)
        self.tools = tools if isinstance(tools, dict) else dict(toolset.specs)
        self.tool_desc = toolset.desc
        self.confine = confine

        # Skills
        self._skills = skills  # None=all, []=none, ['a','b']=specific
        self._skills_prompt = ""
        self._load_skills()

        # Memory
        self.prompt_name = prompt_name
        self.use_memory = use_memory
        self.memory = Memory() if use_memory else None
        # a policy name becomes one Approval for the agent's lifetime ("always" answers persist)
        self.approve = Approval(approve) if isinstance(approve, str) else approve

        # Lifecycle: written memory in <workspace>/.mini_agent, compaction, handoff
        if lifecycle is True:
            from ...lifecycle import Lifecycle
            lifecycle = Lifecycle()
        if lifecycle is not None:
            if lifecycle.workspace is None:
                lifecycle.workspace = self.config.workspace
            if lifecycle.llm is None:
                lifecycle.llm = self.llm
            if lifecycle.base_memory is None:
                lifecycle.base_memory = self._get_memory_prompt
        self.lifecycle = lifecycle

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

    def run(self, goal: str, verbose: bool = True, on_event=None, conversation: str = "",
            state=None) -> str:
        """Run a goal. on_event(type, **data) receives the event stream (None = console output).
        state: an unfinished turn to continue (mini_agent.lifecycle.restore_steps)."""
        workspace_str = get_workspace_str(self.config.workspace)
        context = f"Workspace: {workspace_str}\nSystem: {describe_system()}\n\n{self.tool_desc}"
        try:
            self.config.workspace.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # a tool that uses the path reports the problem as data
        token = _workspace.current.set(self.config.workspace)  # relative tool paths start here
        limit = _workspace.confined.set(self.confine)        # file tools stay inside it
        mem = _memory.current.set(self.memory)                # remember / log write here
        try:
            return self._run(goal, context, verbose, on_event, conversation, state)
        except BaseException as e:
            if self.lifecycle is not None:   # a crash or Ctrl-C still leaves a handoff
                self.lifecycle.end("interrupted" if isinstance(e, KeyboardInterrupt) else "error",
                                   f"{type(e).__name__}: {e}")
            raise
        finally:
            _memory.current.reset(mem)
            _workspace.confined.reset(limit)
            _workspace.current.reset(token)

    def _run(self, goal, context, verbose, on_event, conversation, state=None) -> str:
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
            approve=self.approve,
            lifecycle=self.lifecycle,
            state=state,
        )

    def log(self, content: str):
        """Append a note to today's log"""
        if self.memory:
            self.memory.append_log(content)

    def remember(self, content: str):
        """Add to long-term memory"""
        if self.memory:
            self.memory.append_memory(content)
