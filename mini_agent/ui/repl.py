"""repl.py — plain CLI mode (no screen takeover, no TUI)

The rendering model:
- finalized content is printed straight into terminal scrollback (native scrolling; everything survives exit)
- only the "live area" (thinking stream / running tool) redraws in place via rich Live; it finalizes on completion
- animation is time-driven: the renderable picks its frame from time.time() on each repaint
"""
from __future__ import annotations

import time

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from ..core import Agent
from ..core.policy import Approval, summarize_call
from ..session import SessionLog
from . import pets, theme
from .common import (PLAN_GUIDANCE, SESS_DIR, compact_notice, latest_session, new_session_path,
                     resume_state, session_conversation, stop_notice)

SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class LiveStatus:
    """Live-area renderable: time-driven animation (spinner/shimmer/stream text)."""

    def __init__(self):
        self.mode = "think"          # think | tool
        self.verb = theme.pick_verb()
        self.started = time.time()
        self.reasoning = ""
        self.text = ""
        self.tool_line = ""

    def __rich_console__(self, console, options):
        frame = int((time.time() - self.started) * 12)
        spin = SPIN[frame % len(SPIN)]
        secs = time.time() - self.started
        head = Text()
        if self.mode == "tool":
            head.append(f"{spin} ", style=f"bold {theme.WARNING}")
            head.append(theme.shimmer_text("Running…", frame, base=theme.WARNING,
                                           light=theme.WARNING_SHIMMER,
                                           wavelength=5.0, speed=1.4, bold=False))
            head.append(f" {self.tool_line}", style=theme.INACTIVE)
            yield head
            return
        verb = "Reasoning deeply" if (self.reasoning and not self.text) else self.verb
        head.append(f"{spin} ", style=f"bold {theme.SPIN}")
        head.append(theme.shimmer_text(f"{verb}…", frame))
        head.append(f" ({secs:.0f}s)", style=theme.SUBTLE)
        yield head
        if self.reasoning:
            yield Text(self.reasoning, style=f"italic {theme.SUBTLE}")
        if self.text:
            body = Text()
            glow = 100
            if len(self.text) > glow:
                body.append(self.text[:-glow], style=theme.INACTIVE)
            body.append(theme.shimmer_text(self.text[-glow:], frame,
                                           base=theme.INACTIVE,
                                           light=theme.INACTIVE_SHIMMER,
                                           wavelength=14.0, speed=1.6, bold=False))
            yield body


def _print_think(console, d):
    t = Text()
    t.append("✻ ", style=theme.CLAUDE)
    t.append(d["thought"], style=theme.INACTIVE)
    t.append(f"  (↑{d.get('tokens_in', 0):,} ↓{d.get('tokens_out', 0):,})",
             style=theme.SUBTLE)
    console.print(t)


def _print_tool(console, call, d):
    t = Text()
    err = d.get("is_error", False)
    t.append("● ", style=f"bold {theme.ERROR if err else theme.SUCCESS}")
    t.append(call.get("name", "(invalid reply)"), style=f"bold {theme.TEXT}")
    t.append(f"({call.get('args', '')})", style=theme.INACTIVE)
    content = str(d.get("content", ""))
    lines = content.split("\n")
    for j, ln in enumerate(lines[:30]):
        t.append("\n  ⎿ " if j == 0 else "\n    ", style=theme.SUBTLE)
        t.append(ln, style=theme.ERROR if err else theme.INACTIVE)
    if len(lines) > 30:
        t.append(f"\n    … +{len(lines) - 30} lines", style=theme.SUBTLE)
    console.print(t)


def _print_todos(console, todos):
    t = Text()
    t.append("⚑ Plan", style=f"bold {theme.CLAUDE}")
    for item in todos:
        st = item.get("status")
        if st == "completed":
            t.append("\n  ☑ ", style=theme.SUCCESS)
            t.append(item["content"], style=f"strike {theme.SUBTLE}")
        elif st == "in_progress":
            t.append("\n  ▸ ", style=f"bold {theme.WARNING}")
            t.append(item["content"], style=f"bold {theme.TEXT}")
        else:
            t.append("\n  ☐ ", style=theme.SUBTLE)
            t.append(item["content"], style=theme.INACTIVE)
    console.print(t)


def _rebuild(console, log):
    """Resume: print history straight into scrollback."""
    calls = {}
    for ev in log.events():
        t, d = ev["type"], ev["data"]
        if t == "user/message":
            u = Text()
            u.append("> ", style=f"bold {theme.SUBTLE}")
            u.append(d["content"], style=f"bold {theme.TEXT}")
            console.print(u)
        elif t == "step/think":
            _print_think(console, d)
        elif t == "tool/call":
            calls[d["call_id"]] = d
        elif t == "tool/result":
            _print_tool(console, calls.pop(d["call_id"], {}), d)
        elif t == "todo/write":
            _print_todos(console, d["todos"])
        elif t == "context/compact":
            console.print(Text(compact_notice(d), style=theme.INACTIVE))
        elif t == "turn/end":
            console.print(Text("● ", style=f"bold {theme.CLAUDE}")
                          + Text(str(d["answer"]), style=theme.TEXT))


ANSWERS = {"y": "yes", "yes": "yes", "a": "always", "always": "always"}


def run_repl(resume: bool = False, agent_options: dict | None = None):
    console = Console()
    live_now = []          # the Live display of the running turn, paused while asking

    def ask(action: str, args: dict) -> str:
        live = live_now[-1] if live_now else None
        if live is not None:
            live.stop()
        try:
            console.print(Text(f"Allow {action}?", style=f"bold {theme.CLAUDE}"))
            console.print(Text(summarize_call(action, args), style=theme.TEXT))
            typed = console.input(Text("[y]es / [n]o / [a]lways for this tool: ",
                                       style=theme.SUBTLE))
        except (EOFError, KeyboardInterrupt):
            typed = "n"
        finally:
            if live is not None:
                live.start()
        return ANSWERS.get(typed.strip().lower(), "no")

    options = dict(agent_options or {})
    if options.get("approve") == "ask":
        options["approve"] = Approval("ask", ask=ask)
    latest = latest_session(SESS_DIR) if resume else None
    log = SessionLog(latest or new_session_path(SESS_DIR))
    agent = Agent(use_memory=True, **options)
    pet = pets.choose_pet()
    plan_active = False

    if latest is not None:
        _rebuild(console, log)
        for ev in log.events():
            if ev["type"] == "plan/mode":
                plan_active = ev["data"]["active"]
    else:
        # intro animation: rain reveal + wiggle cycles (Live plays frames, last frame finalizes)
        frames = list(pets.intro_frames(pet))
        with Live(frames[0][0], console=console, refresh_per_second=24,
                  transient=True) as live:
            for art, delay in frames:
                live.update(art)
                if delay:
                    time.sleep(delay)
        console.print(pets.render_pet(pet, 0))
        banner = Text(pets.PETS[pet]["banner"], style=theme.INACTIVE)
        banner.append("  — mini_agent (/plan to plan, /exit to quit)", style=theme.SUBTLE)
        console.print(Text("NTU · Schmidt Science · @shichuan", style=theme.SUBTLE))
        console.print(banner)

    backend = type(agent.llm).__name__.replace("LLM", "").lower()
    model = getattr(agent.llm, "model", "?")
    console.print(Text(f"{backend}·{model}", style=theme.SUBTLE))

    def conversation_text(limit=1600, until_seq=None):
        msgs = log.derive_messages(until_seq=until_seq)
        lines = []
        for m in msgs:
            if m["role"] == "user":
                lines.append("User: " + str(m["content"]))
            elif m["role"] == "assistant":
                lines.append("Assistant: " + str(m["content"]))
        return "\n".join(lines)[-limit:]

    def run_turn(goal, conversation, state=None):
        """Run (or continue, with state) one turn, showing it live and logging every event."""
        status = LiveStatus()
        calls = {}
        with Live(status, console=console, refresh_per_second=12,
                  transient=True) as live:
            live_now.append(live)

            def on_event(t, /, **d):
                log.append(t, **d)
                if t == "step/start":
                    status.mode = "think"
                    status.verb = theme.pick_verb()
                    status.started = time.time()
                    status.reasoning = ""
                    status.text = ""
                elif t == "assistant/chunk":
                    if d.get("kind") == "reasoning":
                        status.reasoning += d["text"]
                    else:
                        status.text += d["text"]
                elif t == "step/think":
                    _print_think(live.console, d)
                elif t == "tool/call":
                    calls[d["call_id"]] = d
                    status.mode = "tool"
                    status.tool_line = f"{d['name']}({d['args']})"
                elif t == "tool/result":
                    _print_tool(live.console, calls.pop(d["call_id"], {}), d)
                    status.mode = "think"
                elif t == "todo/write":
                    _print_todos(live.console, d["todos"])
                elif t == "context/compact":
                    live.console.print(Text(compact_notice(d), style=theme.INACTIVE))
                elif t == "turn/end":
                    live.console.print(Text("● Answer:", style=f"bold {theme.CLAUDE}"))
                    live.console.print(Markdown(str(d["answer"])))
                    if stop_notice(d):
                        live.console.print(Text(stop_notice(d), style=theme.WARNING))

            try:
                options = {"state": state} if state is not None else {}
                answer = agent.run(goal, on_event=on_event, conversation=conversation, **options)
                agent.log(f"Q: {goal} -> A: {str(answer)[:120]}")
            except KeyboardInterrupt:
                live.console.print(Text("Interrupted. Type the next message, or later run "
                                        "`mini_agent -c` to continue this task.", style=theme.WARNING))
            except Exception as e:
                live.console.print(Text(f"[internal error] {e}", style=theme.ERROR))
            finally:
                live_now.pop()

    if latest is not None:
        unfinished = log.unfinished_turn()
        if unfinished is not None:
            # -c: continue the task that did not finish, from its last completed step
            state = resume_state(unfinished)
            if state is None:                 # sent, but the program stopped before it began
                note = f"Running the task sent before the program stopped: {unfinished['goal']}"
            else:
                kept = len(state.steps)
                note = (f"Continuing the unfinished task ({unfinished['reason']}, "
                        f"{kept} step{'' if kept == 1 else 's'} kept): {unfinished['goal']}")
            console.print(Text(note, style=theme.INACTIVE))
            conversation = conversation_text(until_seq=unfinished["before_seq"])
            if plan_active:
                conversation = PLAN_GUIDANCE + conversation
            run_turn(unfinished["goal"], conversation, state=state)

    while True:
        try:
            goal = console.input(Text("❯ ", style=f"bold {theme.CLAUDE}"))
        except (EOFError, KeyboardInterrupt):
            break
        goal = goal.strip()
        if not goal:
            continue
        if goal in ("/exit", "/quit"):
            break
        if goal.startswith("/plan"):
            arg = goal[5:].strip()
            plan_active = arg != "off"
            log.append("plan/mode", active=plan_active)
            console.print(Text(f"✻ plan mode {'on' if plan_active else 'off'}",
                               style=theme.INACTIVE))
            if not plan_active or not arg:
                continue
            goal = arg

        conversation = session_conversation(agent, log, conversation_text)
        if plan_active:
            conversation = PLAN_GUIDANCE + conversation
        log.append("user/message", content=goal)
        run_turn(goal, conversation)


def main(resume: bool = False, agent_options: dict | None = None):
    run_repl(resume=resume, agent_options=agent_options)


if __name__ == "__main__":
    main()
