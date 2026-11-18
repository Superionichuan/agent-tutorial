"""repl.py — plain CLI mode (no screen takeover, no TUI)

The rendering model:
- finalized content is printed straight into terminal scrollback (native scrolling; everything survives exit)
- only the "live area" (thinking stream / running tool) redraws in place via rich Live; it finalizes on completion
- animation is time-driven: the renderable picks its frame from time.time() on each repaint
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from ..core import Agent
from ..core.agent.run import failed_call
from ..core.policy import Approval, NeedsApproval, summarize_call
from ..terminal import stdin_is_terminal
from ..session import SessionLog
from . import pets, theme
from .common import (EXIT_BUSY, PLAN_GUIDANCE, SESS_DIR, compact_notice, exit_status, folder_of,
                     interrupted_line, interrupted_status, stop_signals_raise,
                     hold_session, latest_session, mark_message, new_session_path,
                     note_approval_now, provider_label, resume_state, session_conversation,
                     shows_answer, stop_notice)

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
    t.append("✻ ", style=theme.ACCENT)
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
    t.append("⚑ Plan", style=f"bold {theme.ACCENT}")
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
        elif t == "step/think" and not failed_call(d):
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
            if shows_answer(d):
                console.print(Text("● ", style=f"bold {theme.ACCENT}")
                              + Text(str(d["answer"]), style=theme.TEXT))
            else:
                console.print(Text(stop_notice(d), style=theme.WARNING))


ANSWERS = {"y": "yes", "yes": "yes", "a": "always", "always": "always"}


def run_repl(resume: bool = False, agent_options: dict | None = None,
             session: str | Path | None = None, once: bool = False) -> int:
    """The plain terminal mode. resume (-c): continue the latest session of the agent's
    folder (with no terminal: its latest unfinished task, then exit with that task's
    status, like a one-shot task). Returns the exit status.
    session, once (a run of `--keep-alive`, whose supervisor holds the session's lock):
    continue the unfinished task of this session, then exit with its status."""
    console = Console()
    terminal = stdin_is_terminal()
    once = once or (resume and not terminal)       # continue one task, then exit: no prompt
    if once:
        stop_signals_raise()       # SIGTERM/SIGHUP stop the running tool too, like Ctrl+C
    stopped = {}           # the exit status of a run stopped by Ctrl+C or a signal
    live_now = []          # the Live display of the running turn, paused while asking

    def ask(action: str, args: dict) -> str:
        if not stdin_is_terminal():        # nobody to ask: the task stops (needs_approval)
            raise NeedsApproval(action)
        live = live_now[-1] if live_now else None
        if live is not None:
            live.stop()
        try:
            console.print(Text(f"Allow {action}?", style=f"bold {theme.ACCENT}"))
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
    agent = Agent(use_memory=True, **options)
    folder = folder_of(agent)
    if session is not None:
        latest = Path(session)
    else:
        latest = latest_session(SESS_DIR, workspace=folder, unfinished=once) if resume else None
        if resume and latest is None and once:
            return 2                      # the command line says why (nothing_to_continue)
    path = latest or new_session_path(SESS_DIR)
    if session is None:
        busy = hold_session(path)         # one process per session
        if busy:
            print(busy, file=sys.stderr)
            return EXIT_BUSY
    log = SessionLog(path)
    pet = pets.choose_pet()
    plan_active = False

    if latest is not None:
        if terminal and not once:         # the history; continuing once, just the task
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
        console.print(Text("NTU · Schmidt Sciences · @shichuan", style=theme.SUBTLE))
        console.print(banner)

    model = getattr(agent.llm, "model", "?")
    console.print(Text(f"{provider_label(agent.llm)}·{model}", style=theme.SUBTLE))

    def conversation_text(limit=1600, until_seq=None):
        msgs = log.derive_messages(until_seq=until_seq)
        lines = []
        for m in msgs:
            if m["role"] == "user":
                lines.append("User: " + str(m["content"]))
            elif m["role"] == "assistant":
                lines.append("Assistant: " + str(m["content"]))
        return "\n".join(lines)[-limit:]

    def run_turn(goal, conversation, state=None) -> str:
        """Run (or continue, with state) one turn, showing it live and logging every event.
        Returns how it ended: the turn/end reason, or "interrupted" / "error"."""
        status = LiveStatus()
        calls = {}
        ended = {"reason": "error"}
        with Live(status, console=console, refresh_per_second=12,
                  transient=True) as live:
            live_now.append(live)

            def on_event(t, /, **d):
                log.append(t, **d)
                if t == "turn/end":
                    ended["reason"] = d.get("reason")
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
                elif t == "step/think" and not failed_call(d):
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
                    if shows_answer(d):
                        live.console.print(Text("● Answer:", style=f"bold {theme.ACCENT}"))
                        live.console.print(Markdown(str(d["answer"])))
                    if stop_notice(d):
                        live.console.print(Text(stop_notice(d), style=theme.WARNING))

            try:
                options = {"state": state} if state is not None else {}
                answer = agent.run(goal, on_event=on_event, conversation=conversation, **options)
                agent.log(f"Q: {goal} -> A: {str(answer)[:120]}")
            except KeyboardInterrupt as e:
                ended["reason"] = "interrupted"
                stopped["status"] = interrupted_status(e)
                if once:
                    live.console.print(Text(interrupted_line(e), style=theme.WARNING))
                else:
                    live.console.print(Text("Interrupted. Type the next message, or later run "
                                            "`mini_agent -c` to continue this task.", style=theme.WARNING))
            except Exception as e:
                ended["reason"] = "error"
                live.console.print(Text(f"[internal error] {e}", style=theme.ERROR))
            finally:
                live_now.pop()
        return ended["reason"]

    if latest is not None:
        unfinished = log.unfinished_turn()
        if unfinished is not None:
            # -c: continue the task that did not finish, from its last completed step
            state = resume_state(unfinished)
            note_approval_now(state, unfinished, options.get("approve"), terminal)
            if state is None:                 # sent, but the program stopped before it began
                note = f"Running the task sent before the program stopped: {unfinished['goal']}"
            else:
                kept = len(state.steps)
                note = (f"Continuing the unfinished task ({unfinished['reason']}, "
                        f"{kept} step{'' if kept == 1 else 's'} kept): {unfinished['goal']}")
            console.print(Text(note, style=theme.INACTIVE))
            conversation = session_conversation(agent, log, conversation_text,
                                                until_seq=unfinished["before_seq"])
            if plan_active:
                conversation = PLAN_GUIDANCE + conversation
            if state is None:                 # its message is logged; record it in user.md once
                mark_message(agent, log, unfinished.get("message_seq"))
            reason = run_turn(unfinished["goal"], conversation, state=state)
            if once:                          # like a one-shot task: no prompt, its status
                return stopped.get("status") or exit_status(reason)
    if once:
        # nothing unfinished: the command line said so (2); a --keep-alive run has nothing left (0)
        return 0 if session is not None else 2

    while True:
        try:
            goal = console.input(Text("❯ ", style=f"bold {theme.ACCENT}"))
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
        mark_message(agent, log, log.append("user/message", content=goal, workspace=folder)["seq"])
        run_turn(goal, conversation)
    return 0


def main(resume: bool = False, agent_options: dict | None = None) -> int:
    return run_repl(resume=resume, agent_options=agent_options)


if __name__ == "__main__":
    main()
