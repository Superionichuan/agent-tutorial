"""CLI entry point

- no args: interactive mode (multi-turn)
- with args: one-shot mode (backward compatible)
"""
import sys

from ..core import Agent


def main():
    """Main entry"""
    args = sys.argv[1:]
    resume = False
    use_tui = False
    profile = "agent"
    while args and args[0] in ("--continue", "-c", "--tui", "--coding"):
        if args[0] == "--tui":
            use_tui = True
        elif args[0] == "--coding":
            profile = "coding"
        else:
            resume = True
        args = args[1:]
    if not args:
        if use_tui:
            # full textual UI (--tui)
            from ..ui.tui import MiniAgentApp
            MiniAgentApp(resume=resume).run(inline=True, inline_no_clear=True)
        else:
            # default: plain CLI mode (no screen takeover; output goes to scrollback)
            from ..ui.repl import run_repl
            run_repl(resume=resume)
        return
    sys.argv = [sys.argv[0]] + args

    # one-shot mode
    goal = " ".join(sys.argv[1:])
    print(f"Query: {goal}")
    print("-" * 40)

    agent = Agent(prompt_name=profile)
    result = agent.run(goal)

    print("-" * 40)
    print(f"Answer: {result}")


if __name__ == "__main__":
    main()
