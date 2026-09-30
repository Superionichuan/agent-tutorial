"""tui.py — interactive terminal UI (textual)

Layout (top to bottom):
    ┌──────────────────────────────┐
    |  message stream (scrollable)  |  user / think / tool cards / answers
    |    tool cards, three states: running (amber)  |
    |    -> success (green) / error (red)           |
    ├──────────────────────────────┤
    |  > input                      |  enter to submit, multi-turn
    ├──────────────────────────────┤
    |  footer: model/tokens/steps   |  live status
    └──────────────────────────────┘

Mechanics:
- agent.run() runs in a worker thread (UI never blocks)
- events are posted back to the UI thread via call_from_thread
- the same event stream is written to SessionLog (~/.mini_agent_sessions/*.jsonl) -> replayable
- widgets use the render() protocol returning rich objects (textual 8 compatible)
"""
from __future__ import annotations

import time
from pathlib import Path

from rich.markdown import Markdown
from rich.text import Text

from . import theme
from . import pets

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Input, Rule, Static

from ..core import Agent
from ..session import SessionLog

SESS_DIR = Path.home() / ".mini_agent_sessions"


# ---------- message widgets (render protocol) ----------

class Msg(Widget):
    """Message base: auto height; subclasses implement render()."""
    DEFAULT_CSS = "Msg { height: auto; margin: 1 0 0 0; }"


class UserMsg(Msg):
    """No frame, `>` prefix."""

    def __init__(self, content: str):
        super().__init__()
        self.content = content

    def render(self):
        t = Text()
        t.append("> ", style=f"bold {theme.SUBTLE}")
        t.append(self.content, style=f"bold {theme.TEXT}")
        return t


SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class StreamMsg(Msg):
    """Live thinking stream: breathing spinner + LLM deltas flowing in real time.

    This is where the "natural flow" comes from — replaced by a finalized ThinkMsg on completion.
    """

    TAIL = 200000  # full transparency: effectively no truncation

    def __init__(self):
        super().__init__()
        self.text = ""
        self.reasoning = ""
        self._frame = 0
        self._timer = None
        self.verb = theme.pick_verb()

    def on_mount(self):
        self._timer = self.set_interval(1 / 12, self._tick)

    def _tick(self):
        self._frame += 1
        self.refresh()

    def append_chunk(self, text: str, kind: str = "content"):
        if kind == "reasoning":
            self.reasoning += text
        else:
            self.text += text
        self.refresh(layout=True)

    def stop(self):
        if self._timer is not None:
            self._timer.stop()

    def render(self):
        spin = SPIN[self._frame % len(SPIN)]
        secs = self._frame / 12
        # flowing gradient: a glow band sweeps the verb per character (true shimmer)
        head = Text()
        head.append(f"{spin} ", style=f"bold {theme.SPIN}")
        verb = "Reasoning deeply" if (self.reasoning and not self.text) else self.verb
        head.append(theme.shimmer_text(f"{verb}…", self._frame))
        head.append(f" ({secs:.0f}s)", style=theme.SUBTLE)
        if self.reasoning:
            # deep-think reasoning chain: darkest + italic
            rt = self.reasoning
            head.append("\n")
            head.append(rt, style=f"italic {theme.SUBTLE}")
        if self.text:
            tail = self.text[-self.TAIL:]
            if len(self.text) > self.TAIL:
                tail = "…" + tail
            head.append("\n")
            # old text settles to static gray; the newest ~100 chars carry a gray-scale glow
            GLOW = 100
            if len(tail) > GLOW:
                head.append(tail[:-GLOW], style=theme.INACTIVE)
            head.append(theme.shimmer_text(
                tail[-GLOW:], self._frame,
                base=theme.INACTIVE, light=theme.INACTIVE_SHIMMER,
                wavelength=14.0, speed=1.6, bold=False))
        return head


class PetSplash(Msg):
    """Splash: the virtual pet enters (pixel animation); leaves on first submit."""

    def __init__(self, pet: str):
        super().__init__()
        self.pet = pet
        self._frame = 0

    def on_mount(self):
        self._timer = self.set_interval(0.5, self._tick)

    def _tick(self):
        self._frame += 1
        self.refresh()

    def stop(self):
        self._timer.stop()

    def render(self):
        art = pets.render_pet(self.pet, self._frame)
        art.append(pets.PETS[self.pet]["banner"], style=theme.INACTIVE)
        art.append("  — mini_agent · NTU · Schmidt Science · @shichuan", style=theme.SUBTLE)
        return art


class ThinkMsg(Msg):
    def __init__(self, step: int, thought: str, tokens: str):
        super().__init__()
        self.step, self.thought, self.tokens = step, thought, tokens

    def render(self):
        body = Text()
        body.append("✻ ", style=theme.CLAUDE)
        body.append(self.thought, style=theme.INACTIVE)
        body.append(f"  ({self.tokens})", style=theme.SUBTLE)
        return body


class ToolCard(Msg):
    """Tool card: three-state colors (pending amber -> green / red), folded result preview."""

    PREVIEW_LINES = 30  # full transparency: generous preview

    def __init__(self, name: str, args):
        super().__init__()
        self.tool_name = name
        self.tool_args = args
        self.result_text: str | None = None
        self.is_error = False
        self.started = time.time()
        self.elapsed = 0.0
        self._frame = 0

    def on_mount(self):
        self._timer = self.set_interval(1 / 12, self._tick)

    def _tick(self):
        if self.result_text is None:
            self._frame += 1
            self.refresh()
        else:
            self._timer.stop()

    def render(self):
        """`●` head line (dot color = state) + `⎿` indented result, no frame."""
        if self.result_text is None:
            dot = theme.WARNING
        elif self.is_error:
            dot = theme.ERROR
        else:
            dot = theme.SUCCESS

        body = Text()
        body.append("● ", style=f"bold {dot}")
        body.append(f"{self.tool_name}", style=f"bold {theme.TEXT}")
        body.append(f"({self.tool_args})", style=theme.INACTIVE)
        if self.result_text is None:
            spin = SPIN[self._frame % len(SPIN)]
            body.append(f"  {spin} ", style=theme.WARNING)
            body.append(theme.shimmer_text("Running…", self._frame,
                        base=theme.WARNING, light=theme.WARNING_SHIMMER,
                        wavelength=5.0, speed=1.4, bold=False))
        else:
            body.append(f"  ({self.elapsed:.1f}s)", style=theme.SUBTLE)
            lines = str(self.result_text).split("\n")
            shown = lines[: self.PREVIEW_LINES]
            for j, ln in enumerate(shown):
                prefix = "  ⎿ " if j == 0 else "    "
                body.append(f"\n{prefix}", style=theme.SUBTLE)
                body.append(ln, style=theme.INACTIVE if not self.is_error else theme.ERROR)
            rest = len(lines) - len(shown)
            if rest > 0:
                body.append(f"\n    … +{rest} lines", style=theme.SUBTLE)
        return body

    def set_result(self, content: str, is_error: bool):
        self.result_text = content
        self.is_error = is_error
        self.elapsed = time.time() - self.started
        self.refresh(layout=True)


class AnswerMsg(Msg):
    """Answer = body text, no frame, just a ● lead; typewriter reveal."""

    def __init__(self, content: str):
        super().__init__()
        self.content = str(content)
        self._shown = 0

    def on_mount(self):
        self._timer = self.set_interval(1 / 30, self._tick)

    def _tick(self):
        if self._shown >= len(self.content):
            self._timer.stop()
            return
        self._shown = min(self._shown + 4, len(self.content))
        self.refresh(layout=True)

    def render(self):
        done = self._shown >= len(self.content)
        if done:
            # once complete, render as markdown (code blocks/lists/headings/emphasis)
            return Markdown(self.content, style=theme.TEXT)
        t = Text()
        t.append("● ", style=f"bold {theme.CLAUDE}")
        t.append(self.content[: self._shown], style=theme.TEXT)
        return t


class TodoCard(Msg):
    """Todo card (CC-style checklist): done / in progress / pending"""

    def __init__(self, todos: list):
        super().__init__()
        self.todos = todos

    def render(self):
        t = Text()
        t.append("⚑ Plan", style=f"bold {theme.CLAUDE}")
        for item in self.todos:
            st = item.get("status")
            if st == "completed":
                t.append("\n  ☑ ", style=theme.SUCCESS)
                t.append(item['content'], style=f"strike {theme.SUBTLE}")
            elif st == "in_progress":
                t.append("\n  ▸ ", style=f"bold {theme.WARNING}")
                t.append(item['content'], style=f"bold {theme.TEXT}")
            else:
                t.append("\n  ☐ ", style=theme.SUBTLE)
                t.append(item["content"], style=theme.INACTIVE)
        return t


PLAN_GUIDANCE = ("[Plan mode] You are in plan mode: explore and plan only; do not run tools "
                 "with side effects (file writes / mutating bash). Produce a complete step-by-step "
                 "plan first (markdown, starting with a # heading) and wait for user confirmation.\n")


class StatusBar(Static):
    """Footer: backend-model | tokens | steps | state"""

    def __init__(self):
        super().__init__()
        self.model = "-"
        self.tokens_in = 0
        self.tokens_out = 0
        self.steps = 0
        self.state = "idle"
        self._pet_frame = 0

    def on_mount(self):
        self._refresh()
        self.set_interval(0.6, self._pet_tick)

    def _pet_tick(self):
        self._pet_frame += 1
        self._refresh()

    def _refresh(self):
        t = Text(no_wrap=True)
        mini = pets.PETS.get(getattr(self, "pet", ""), next(iter(pets.PETS.values())))["mini"]
        pet = mini.get(self.state, mini["idle"])
        t.append(pet[self._pet_frame % len(pet)] + " ")
        t.append(f" {self.model} ", style=f"{theme.CLAUDE} reverse")
        t.append(f"  ↑{self.tokens_in:,} ↓{self.tokens_out:,} tok", style=theme.SUBTLE)
        t.append(f"  step {self.steps}", style=theme.SUBTLE)
        color = {"idle": theme.SUCCESS, "running": theme.WARNING}.get(self.state, theme.ERROR)
        t.append(f"  ● {self.state}", style=color)
        t.append("   ctrl+c to quit", style=theme.SUBTLE)
        self.update(t)

    def set(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self._refresh()


# ---------- main app ----------

class MiniAgentApp(App):
    """The interactive terminal UI for mini_agent."""

    CSS = """
    VerticalScroll { padding: 0 1; height: 18; }
    #bottomzone { dock: bottom; height: auto; padding: 0 1; }
    #promptrow { height: 1; }
    #promptsym { width: 2; }
    Input { border: none; background: transparent; padding: 0; height: 1; }
    Input:focus { border: none; }
    Rule.inputline { color: rgb(70,70,70); margin: 0; }
    StatusBar { height: 1; }
    """

    def __init__(self, resume: bool = False):
        super().__init__()
        self.resume = resume
        self.agent = None
        self.session_log = None
        self._cards: dict[str, ToolCard] = {}
        self._stream: StreamMsg | None = None
        self.pet = pets.choose_pet()
        self.plan_active = False
        self._splash: PetSplash | None = None

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="stream")
        yield Vertical(
            Rule(classes="inputline"),
            Horizontal(
                Static(Text("❯", style=f"bold {theme.CLAUDE}"), id="promptsym"),
                Input(placeholder="Ask anything... (enter to submit)"),
                id="promptrow",
            ),
            Rule(classes="inputline"),
            StatusBar(),
            id="bottomzone",
        )

    def _latest_session(self):
        files = sorted(SESS_DIR.glob("*.jsonl"))
        return files[-1] if files else None

    def _conversation_text(self, limit: int = 1600) -> str:
        """Cross-turn context = a projection of the session log (resume includes history for free)."""
        msgs = self.session_log.derive_messages()
        lines = []
        for m in msgs:
            if m["role"] == "user":
                lines.append(f"User: {m['content']}")
            elif m["role"] == "assistant":
                lines.append(f"Assistant: {m['content']}")
        text = "\n".join(lines)
        return text[-limit:]

    def _rebuild_from_log(self):
        """Resume: statically rebuild the message stream from the log."""
        cards = {}
        for ev in self.session_log.events():
            t, d = ev["type"], ev["data"]
            if t == "user/message":
                self._append(UserMsg(d["content"]))
            elif t == "step/think":
                self._append(ThinkMsg(d["step"], d["thought"],
                             f"↑{d.get('tokens_in',0):,} ↓{d.get('tokens_out',0):,}"))
            elif t == "tool/call":
                c = ToolCard(d["name"], d["args"]); cards[d["call_id"]] = c
                self._append(c)
            elif t == "tool/result":
                c = cards.pop(d["call_id"], None)
                if c is None:  # invalid reply: no tool call preceded this result
                    c = ToolCard("(invalid reply)", {}); self._append(c)
                c.set_result(d["content"], d.get("is_error", False))
            elif t == "todo/write":
                self._append(TodoCard(d["todos"]))
            elif t == "plan/mode":
                self.plan_active = d["active"]   # state = fold of the log (resume recovers it)
            elif t == "turn/end":
                m = AnswerMsg(d["answer"]); m._shown = len(m.content)
                self._append(m)

    def on_mount(self):
        SESS_DIR.mkdir(exist_ok=True)
        latest = self._latest_session() if self.resume else None
        if latest is not None:
            self.session_log = SessionLog(latest)   # continue the same file
        else:
            self.session_log = SessionLog(SESS_DIR / f"{int(time.time())}.jsonl")
        self.agent = Agent(use_memory=True)
        backend = type(self.agent.llm).__name__.replace("LLM", "").lower()
        model = getattr(self.agent.llm, "model", "?")
        bar = self.query_one(StatusBar)
        bar.pet = self.pet
        bar.set(model=f"{backend}·{model}")
        if latest is not None:
            self._rebuild_from_log()   # restore history view
        else:
            self._splash = PetSplash(self.pet)
            self._append(self._splash)
        self.query_one(Input).focus()

    # ---- input submit -> run agent in background ----

    def on_input_submitted(self, ev: Input.Submitted):
        goal = ev.value.strip()
        if not goal:
            return
        if goal.startswith("/plan"):
            arg = goal[5:].strip()
            self.plan_active = arg != "off"
            self.session_log.append("plan/mode", active=self.plan_active)
            note = ThinkMsg(0, f"plan mode {'on' if self.plan_active else 'off'}", "/plan")
            self.query_one(Input).clear()
            self._append(note)
            if not self.plan_active or not arg or arg == "off":
                return
            goal = arg
        self.query_one(Input).clear()
        if self._splash is not None:
            self._splash.stop()
            self._splash.remove()
            self._splash = None
        self._append(UserMsg(goal))
        self.query_one(StatusBar).set(state="running", steps=0)
        self.run_goal(goal)

    @work(thread=True, exclusive=True)
    def run_goal(self, goal: str):
        """Worker thread: run the agent; post events to the UI + write SessionLog."""
        # long-running: compact when the session grows too large
        if len(self._conversation_text(limit=999999)) > 2500:
            self.session_log.compact(self.agent.llm)
        conversation = self._conversation_text()
        if self.plan_active:
            conversation = PLAN_GUIDANCE + conversation
        self.session_log.append("user/message", content=goal)
        def on_event(type: str, **d):
            self.session_log.append(type, **d)               # persist
            self.call_from_thread(self._on_event, type, d)   # render on UI thread
        try:
            answer = self.agent.run(goal, on_event=on_event,
                                    conversation=conversation)
            # durable memory: auto-log each turn to today's log (the agent can also use remember)
            self.agent.log(f"Q: {goal} -> A: {str(answer)[:120]}")
        except Exception as e:
            self.call_from_thread(self._on_event, "turn/end",
                                  {"answer": f"[internal error] {e}", "reason": "error"})

    # ---- events -> rendering (UI thread) ----

    def _append(self, widget):
        stream = self.query_one("#stream", VerticalScroll)
        stream.mount(widget)
        stream.scroll_end(animate=False)

    def _drop_stream(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.remove()
            self._stream = None

    def _on_event(self, type: str, d: dict):
        bar = self.query_one(StatusBar)
        if type == "step/start":
            # show the live stream (spinner) immediately so waiting feels alive
            self._drop_stream()
            self._stream = StreamMsg()
            self._append(self._stream)
        elif type == "assistant/chunk":
            if self._stream is not None:
                self._stream.append_chunk(d["text"], d.get("kind", "content"))
                stream = self.query_one("#stream", VerticalScroll)
                stream.scroll_end(animate=False)
        elif type == "step/think":
            self._drop_stream()  # finalize the live stream into a formatted Think line
            bar.set(steps=d["step"], tokens_in=d["tokens_in"],
                    tokens_out=d["tokens_out"])
            self._append(ThinkMsg(d["step"], d["thought"],
                                  f"↑{d['tokens_in']:,} ↓{d['tokens_out']:,}"))
        elif type == "tool/call":
            card = ToolCard(d["name"], d["args"])
            self._cards[d["call_id"]] = card
            self._append(card)
        elif type == "tool/result":
            card = self._cards.pop(d["call_id"], None)
            if card is None:  # invalid reply: no tool call preceded this result
                card = ToolCard("(invalid reply)", {})
                self._append(card)
            card.set_result(d["content"], d["is_error"])
        elif type == "todo/write":
            self._append(TodoCard(d["todos"]))
        elif type == "turn/end":
            self._drop_stream()
            self._append(AnswerMsg(d["answer"]))
            bar.set(state="idle")


def main():
    # inline mode: stays in the current terminal (no alt screen); the UI remains in history on exit
    MiniAgentApp().run(inline=True, inline_no_clear=True)


if __name__ == "__main__":
    main()
