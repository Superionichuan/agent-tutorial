"""Command line

  mini_agent setup                 choose a model provider and paste a key (once)
  mini_agent doctor                what is installed and which model is set up
  mini_agent                       interactive session in the terminal
  mini_agent "task"                run one task and exit
  mini_agent --tui                 full-screen interface

Options go before the task; everything after the first task word is the task.
"""
import argparse
import os
import sys
from typing import Optional

from .. import __version__
from ..core.policy import POLICIES

DEFAULT_BUDGET = 1_000_000      # tokens per task on the command line (with the lifecycle)
KEEP_ALIVE_TRIES = 5            # --keep-alive: continues in a row without progress

SETUP_HELP = """\
First time: `mini_agent setup` asks for a provider and a key and saves them in
~/.mini_agent/config.env; `mini_agent doctor` shows what is set up.

Or choose a model with environment variables (or a .env file in the current directory):
  OPENROUTER_API_KEY=sk-or-...            many models through one key (openrouter.ai)
  DEEPSEEK_API_KEY=sk-...                 DeepSeek
  ANTHROPIC_API_KEY=sk-ant-...            Claude
  OPENAI_API_KEY=sk-...                   OpenAI
  LLM_BASE_URL=http://host:port/v1        any OpenAI-compatible server (vLLM, LM Studio, ...)
  LLM_BACKEND=offline                     a recorded demo; no key, no network
  LLM_MODEL=...                           the model, for any of the above
Without any of these, a local Ollama server (ollama serve) is used.

Safety: the agent can read and write files and run shell commands and Python.
By default it asks before each risky call (bash, run_python, write_file), and
its file tools only reach paths inside the workspace. That limit is a
convenience, not isolation: for real isolation, run mini_agent inside a
virtual machine or container of your choice.

Exit status of `mini_agent "task"` (and of `mini_agent -c` with no terminal): 0 the
agent finished (its answer says whether it managed); 1 stopped before the end (token
budget, step limit, a model error, a task offline mode has no recording of, ...);
3 a risky call needs approval and there was no terminal to ask (`mini_agent -c --yes`
continues); 2 a usage or set-up error, or nothing for -c to continue in this folder;
4 the session is in use by another mini_agent process; 130 interrupted. After 1, 3 or
130, `mini_agent -c` in the same folder continues the task.

--keep-alive [N] keeps a task going after a passing failure: the provider down or
unreachable (no HTTP status, 408, 429, 5xx), or the process killed. It waits (30 s, 1, 2,
4, 8 min, then 10 min) and continues with -c; all runs share the task's token budget. A
stop that waiting will not change ends it at once, with that stop's status. After N
continues in a row (default 5) that complete no step it gives up: exit 75 ("try again
later"). `mini_agent -c --keep-alive` with nothing unfinished here exits 0.
"""

NO_MODEL = """\
mini_agent: no model is configured, and no local Ollama server is answering.
Run `mini_agent setup` to choose one, or set one of these (environment or a .env file here):
  OPENROUTER_API_KEY=sk-or-...     DEEPSEEK_API_KEY=sk-...     ANTHROPIC_API_KEY=sk-ant-...
  LLM_BASE_URL=http://host:port/v1 and LLM_MODEL=...   (any OpenAI-compatible server)
  LLM_BACKEND=offline              (a recorded demo; no key needed)
More: mini_agent --help"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mini_agent",
        description="A small, readable agent: a language model in a think-act-observe loop "
                    "with tools, memory and skills.",
        epilog=SETUP_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", nargs=argparse.REMAINDER,
                   help="run this task and exit; without a task, start an interactive session")
    p.add_argument("-c", "--continue", dest="resume", action="store_true",
                   help="continue the latest session of the current folder (with no terminal: "
                        "its latest unfinished task, then exit)")
    p.add_argument("--tui", action="store_true", help="full-screen interface")
    p.add_argument("--coding", action="store_true",
                   help="coding prompt: read before writing, run and check the code")
    p.add_argument("--approve", choices=POLICIES, default="ask",
                   help="risky tool calls (bash, run_python, write_file): ask before each "
                        "(default), run without asking (auto), or never run them (deny)")
    p.add_argument("-y", "--yes", action="store_true", help="same as --approve auto")
    p.add_argument("--workspace", metavar="DIR",
                   help="the directory the agent works in (default: the current directory)")
    p.add_argument("--no-confine", dest="confine", action="store_false",
                   help="let the file tools reach paths outside the workspace")
    p.add_argument("--budget", type=int, metavar="TOKENS",
                   help=f"stop a task after this many tokens (all its model calls together); "
                        f"`mini_agent -c` then continues it (default {DEFAULT_BUDGET:,}, or "
                        f"$MINI_AGENT_BUDGET)")
    p.add_argument("--no-lifecycle", dest="lifecycle", action="store_false",
                   help="no written memory, compaction or handoff (in <workspace>/.mini_agent): "
                        "the plain loop, which stops when the context window is full")
    p.add_argument("--keep-alive", dest="keep_alive", nargs="?", const=KEEP_ALIVE_TRIES,
                   metavar="N",
                   help="keep the task going: after a passing failure (the provider down, the "
                        "process killed) continue it with -c, up to N times in a row without "
                        f"progress (default {KEEP_ALIVE_TRIES}); with a task, or with -c")
    p.add_argument("--keepalive-session", help=argparse.SUPPRESS)   # a run of --keep-alive
    p.add_argument("--backend", metavar="NAME",
                   help="LLM backend or provider, e.g. openrouter, deepseek, claude, offline "
                        "(sets LLM_BACKEND)")
    p.add_argument("--model", metavar="NAME", help="model name (sets LLM_MODEL)")
    p.add_argument("--version", action="version", version=f"mini_agent {__version__}")
    return p


def keep_alive_count(args, argv) -> str:
    """--keep-alive takes an optional count: `--keep-alive 3 "task"`, `--keep-alive=3`. A task
    word right after it (`--keep-alive "task"`) is the task, not a count: put it back.
    Returns "" or what is wrong."""
    value = args.keep_alive
    if value is None or isinstance(value, int):
        return ""
    if value.isdigit():
        if int(value) < 1:
            return "--keep-alive N must be 1 or more"
        args.keep_alive = int(value)
        return ""
    if f"--keep-alive={value}" in argv:
        return f"--keep-alive N must be a whole number, not {value!r}"
    args.task = [value, *args.task]
    args.keep_alive = KEEP_ALIVE_TRIES
    return ""


def setup_problem() -> str:
    """NO_MODEL when nothing is configured and no local Ollama server answers; else ""."""
    from ..llm.factory import choose_backend, load_config
    config = load_config()
    if choose_backend(config) != "ollama" or config.get("LLM_BACKEND") \
            or config.get("OLLAMA_HOST") or config.get("OLLAMA_MODEL"):
        return ""
    import httpx
    try:
        httpx.get("http://localhost:11434/api/tags", timeout=1.0)
        return ""
    except httpx.HTTPError:
        return NO_MODEL


def budget(args) -> Optional[int]:
    """Tokens a task may use: --budget, else $MINI_AGENT_BUDGET, else DEFAULT_BUDGET with the
    lifecycle (None without it: the agent's own limit, the context window)."""
    if args.budget is not None:
        value, where = args.budget, "--budget"
    elif os.environ.get("MINI_AGENT_BUDGET", "").strip():
        where = "MINI_AGENT_BUDGET"
        try:
            value = int(os.environ["MINI_AGENT_BUDGET"].replace(",", "").replace("_", ""))
        except ValueError:
            raise ValueError(f"MINI_AGENT_BUDGET must be a whole number of tokens, "
                             f"not {os.environ['MINI_AGENT_BUDGET']!r}") from None
    else:
        return DEFAULT_BUDGET if args.lifecycle else None
    if value <= 0:
        raise ValueError(f"{where} must be a positive number of tokens")
    return value


def agent_options(args) -> dict:
    """Agent(...) keyword arguments from the command-line options. The command line
    runs with a lifecycle (written memory, compaction, handoff); the library does not."""
    options = dict(approve="auto" if args.yes else args.approve, workspace=args.workspace,
                   confine=args.confine, prompt_name="coding" if args.coding else "agent",
                   lifecycle=True if args.lifecycle else None)
    tokens = budget(args)
    if tokens is not None:
        options["max_tokens"] = tokens
    return options


def utf8_console():
    """Windows consoles and pipes default to a legacy code page: write UTF-8 instead."""
    if os.name == "nt":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main(argv=None):
    """Main entry"""
    argv = sys.argv[1:] if argv is None else list(argv)
    utf8_console()
    if argv and argv[0] in ("setup", "doctor"):
        from . import doctor, setup
        return (setup if argv[0] == "setup" else doctor).main(argv[1:])
    args = build_parser().parse_args(argv)
    if args.task and args.task[0] == "--":
        args.task = args.task[1:]
    wrong = keep_alive_count(args, argv)
    if not wrong and args.keep_alive is not None:
        if args.tui:
            wrong = "--keep-alive keeps a task going in the terminal; it does not work with --tui"
        elif not args.task and not args.resume:
            wrong = ('--keep-alive needs a task (mini_agent --keep-alive "task") or -c '
                     "(mini_agent -c --keep-alive)")
    if wrong:
        print(f"mini_agent: {wrong}", file=sys.stderr)
        return 2
    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    problem = setup_problem()
    if problem:
        print(problem, file=sys.stderr)
        return 2
    try:
        options = agent_options(args)
    except ValueError as e:                      # a budget that is not a positive number
        print(f"mini_agent: {e}", file=sys.stderr)
        return 2
    if args.resume and args.task:
        print("mini_agent: -c continues the latest session; use it without a task "
              "(mini_agent -c), then type the next message", file=sys.stderr)
        return 2
    if args.keep_alive is not None:
        # this process supervises; the task runs in child processes (cli/keepalive.py)
        from .keepalive import supervise
        return supervise(args, options.get("max_tokens") or DEFAULT_BUDGET)

    if not args.task:
        # Settings that cannot work (an unknown LLM_BACKEND, a key meant for another
        # provider) are a message here, as in one-shot mode, not a traceback later.
        from ..llm import create_llm
        try:
            create_llm()
        except (RuntimeError, ValueError) as e:
            print(f"mini_agent: {e}", file=sys.stderr)
            return 2
        if args.resume and args.keepalive_session:
            # a run of --keep-alive: continue its pinned session once (its supervisor holds the lock)
            from ..ui.repl import run_repl
            return run_repl(resume=True, agent_options=options, session=args.keepalive_session,
                            once=True)
        if args.resume:
            # -c works only on the current folder: it never runs another folder's task here
            from ..terminal import stdin_is_terminal
            from ..ui.common import nothing_to_continue
            why = nothing_to_continue(args.workspace or os.getcwd(),
                                      terminal=args.tui or stdin_is_terminal())
            if why:
                print(why, file=sys.stderr)
                return 2
        if args.tui:
            # full textual UI (--tui)
            from ..ui.common import EXIT_BUSY
            from ..ui.tui import MiniAgentApp
            app = MiniAgentApp(resume=args.resume, agent_options=options)
            app.run(inline=True, inline_no_clear=True)
            if app.busy:                         # another process runs this session
                print(app.busy, file=sys.stderr)
                return EXIT_BUSY
            return 0
        # default: plain CLI mode (no screen takeover; output goes to scrollback)
        from ..ui.repl import run_repl
        return run_repl(resume=args.resume, agent_options=options)

    # one-shot mode: logged like an interactive session, so `mini_agent -c` can
    # continue it if it is cut off or stopped
    from ..core import Agent
    from ..core.agent.run import console_renderer
    from ..session import SessionLog
    from ..ui.common import (SESS_DIR, exit_status, folder_of, hold_session, mark_message,
                             new_session_path, stop_notice)
    goal = " ".join(args.task)
    print(f"Query: {goal}")
    print("-" * 40)
    try:
        agent = Agent(**options)
    except (RuntimeError, ValueError) as e:      # e.g. an unknown LLM_BACKEND
        print(f"mini_agent: {e}", file=sys.stderr)
        return 2
    if args.keepalive_session:                   # the first run of --keep-alive: its session
        log = SessionLog(args.keepalive_session)
    else:
        log = SessionLog(new_session_path(SESS_DIR))
        hold_session(log.path)                   # a new session: nobody else can hold it
    mark_message(agent, log, log.append("user/message", content=goal,
                                        workspace=folder_of(agent))["seq"])

    ended = {}

    def on_event(type, /, **data):
        log.append(type, **data)
        if type == "turn/end":
            ended["reason"] = data.get("reason")
            ended["notice"] = stop_notice(data)
            if ended["notice"]:        # the instruction says it all: no "-> stopped", no "Answer:"
                if data.get("reason") == "llm_error":
                    print(f"  {data.get('answer')}")     # the provider's message, then what to do
                print(f"  {ended['notice']}")
                return
        console_renderer(type, **data)
    try:
        result = agent.run(goal, on_event=on_event)
    except KeyboardInterrupt:
        print("\nInterrupted. `mini_agent -c` continues this task.", file=sys.stderr)
        return 130
    print("-" * 40)
    if not ended.get("notice"):
        print(f"Answer: {result}")
    return exit_status(ended.get("reason"))


if __name__ == "__main__":
    sys.exit(main())
