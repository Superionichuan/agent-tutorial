"""Approval: which tool calls run without asking

Three policies:
  auto  every call runs. The library default, so notebooks and scripts never block.
  ask   risky tools (bash, run_python, write_file, and any @tool(risky=True)) need a
        yes from the person at the keyboard; other tools run without asking.
        The command-line default. With no terminal to ask on (input redirected, a
        scheduled job), the first risky call stops the task (turn/end reason
        needs_approval) instead of being refused again and again; `mini_agent -c --yes`
        continues it.
  deny  risky tools never run; other tools do.
A custom check is any function approve(action, args) -> bool.

Separately, the file tools (read_file, write_file, list_dir) are confined to
the agent's workspace (Agent(confine=True), the default): a path that resolves
outside it, also through a symlink, is refused. This limit is a convenience
boundary, not isolation: bash and run_python are not confined by it, and the
path is checked just before it is used, so a program changing symlinks at the
same moment can get past it. For real isolation, run mini_agent inside a
virtual machine or container of your choice.
"""
import json
from typing import Callable, Iterable, Optional

from ..terminal import stdin_is_terminal

POLICIES = ("ask", "auto", "deny")


class NeedsApproval(Exception):
    """A risky call needs a yes and there is nobody to ask: the loop stops the task."""

    def __init__(self, action: str):
        super().__init__(f"{action} needs approval, and there is no terminal to ask")
        self.action = action


def summarize_call(action: str, args: dict, width: int = 600) -> str:
    """What the person is asked to approve: the command, the code, or the file to write."""
    if action == "bash":
        text = str(args.get("cmd", ""))
    elif action == "run_python":
        text = str(args.get("code", ""))
    elif action == "write_file":
        content = str(args.get("content", ""))
        text = f"{args.get('path', '')}  ({len(content)} characters)\n{content[:300]}"
    else:
        text = json.dumps(args, ensure_ascii=False, default=str)
    return text if len(text) <= width else text[:width] + " ..."


def console_ask(action: str, args: dict) -> str:
    """Ask on the terminal. Returns "yes", "no" or "always". Without a terminal, raises
    NeedsApproval: nobody can answer, so the task stops instead of going on without the tool."""
    if not stdin_is_terminal():
        raise NeedsApproval(action)
    body = summarize_call(action, args).replace("\n", "\n    ")
    print(f"\n  Allow {action}?\n    {body}")
    try:
        answer = input("  [y]es / [n]o / [a]lways for this tool: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "no"
    return {"y": "yes", "yes": "yes", "a": "always", "always": "always"}.get(answer, "no")


class Approval:
    """An approval policy. bind(risky) gives the approve(action, args) check for a run.

    ask(action, args) -> "yes" | "no" | "always" asks the person (default: the
    terminal). "always" allows that tool for the rest of the session.
    """

    def __init__(self, policy: str = "ask", ask: Optional[Callable[[str, dict], str]] = None):
        if policy not in POLICIES:
            raise ValueError(f"unknown approval policy {policy!r}; use one of: {', '.join(POLICIES)}")
        self.policy = policy
        self.ask = ask or console_ask
        self.always = set()

    def bind(self, risky: Iterable[str]) -> Optional[Callable[[str, dict], bool]]:
        risky = frozenset(risky)
        if self.policy == "auto":
            return None
        if self.policy == "deny":
            return lambda action, args: action not in risky

        def approve(action: str, args: dict) -> bool:
            if action not in risky or action in self.always:
                return True
            answer = self.ask(action, args)
            if answer == "always":
                self.always.add(action)
                return True
            return answer in ("yes", True)
        return approve

    def __repr__(self):
        return f"Approval({self.policy!r})"


def resolve_approval(approve, risky: Iterable[str]) -> Optional[Callable[[str, dict], bool]]:
    """What Agent / run_agent were given (None, a policy name, an Approval, or a
    function) -> the approve(action, args) check for this run (None = allow all)."""
    if approve is None:
        return None
    if isinstance(approve, str):
        approve = Approval(approve)
    if isinstance(approve, Approval):
        return approve.bind(risky)
    if callable(approve):
        return approve
    raise TypeError("approve must be None, 'ask' | 'auto' | 'deny', an Approval, "
                    "or a function approve(action, args) -> bool")
