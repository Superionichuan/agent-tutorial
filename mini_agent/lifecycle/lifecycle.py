"""Lifecycle: written memory, compaction before each model call, handoff at every stop.

The loop (core/agent/run.py) calls it in three places; with lifecycle=None nothing changes:

    emit = lifecycle.attach(emit, state, resumed)       # before turn/start; turn/end -> end()
    memory = lifecycle.start(goal, memory)              # records the user message
    memory = lifecycle.before_think(step, prompt_of)    # before every model call

Two limits are kept apart. The context window: compaction looks at the size of the
next prompt (prompt_of renders it; the last call's real input tokens calibrate the
estimate). The run's max_tokens: a spending budget over the whole turn; compaction
does not reset it, and when it stops the turn, end() writes the handoff so that
`mini_agent -c` continues with a fresh budget.
"""
import inspect
from pathlib import Path

from ..core.state import State
from ..core.state.format_history import format_history
from ..mem.memory import current as _running_memory
from . import compact as C
from . import notes as N

HINT = ("To keep a checked fact across compactions, call remember with content and evidence "
        "(how you know it: a command and its output, a file and line, a source).")
INTERRUPTED = ("Error: interrupted before this call finished (the session stopped); it may have run "
               "partly or not at all. Check its effect before repeating it.")
# format_history keeps at most this many characters of steps (it drops the oldest beyond)
HISTORY_CHARS = 2 * inspect.signature(format_history).parameters["max_tokens"].default


class Lifecycle:
    """What lets one agent keep working on a long task.

    workspace     where .mini_agent/ (user.md, handoff.md) is written; Agent fills it in
    llm           the agent's model: its window sets the thresholds; summaries use it
    compact_at    compact when the next prompt would pass this (share of the window when
                  <= 1, else tokens)
    compact_to    ... down to this, if the mechanical passes can
    keep_steps    the last steps of a turn, never cut
    session_budget  the earlier-session text between turns (conversation())
    base_memory   a function giving the [Memory] text without the lifecycle part (Agent sets
                  its own); without it, the memory given to start() is used
    """

    user_share = 0.25      # earlier user messages beyond this share: the oldest stay in user.md only
    notes_share = 0.15     # the same for notes with evidence

    def __init__(self, workspace=None, llm=None, compact_at=0.5, compact_to=0.25, keep_steps=6,
                 session_budget=0.1, base_memory=None):
        self.workspace = workspace
        self.llm = llm
        self.compact_at, self.compact_to = compact_at, compact_to
        self.keep_steps = max(0, int(keep_steps))
        self.session_budget = session_budget
        self.base_memory = base_memory
        self.tokens_per_byte = 1 / C.BYTES_PER_TOKEN
        self.state, self.goal, self.resumed = None, "", False
        self._emit, self._base, self._ended = None, "", True
        self._new_turn()

    # ---------- sizes ----------

    @property
    def folder(self) -> Path:
        return Path(self.workspace if self.workspace is not None else ".") / ".mini_agent"

    @property
    def window(self) -> int:
        return int(getattr(self.llm, "context_limit", 0) or 128_000)

    def limit(self, value) -> int:
        """Tokens for a share of the window (<= 1) or a token count (> 1)."""
        return int(value * self.window) if value <= 1 else int(value)

    def count(self, text: str) -> int:
        """Estimated tokens of a prompt, calibrated by the model's reported usage."""
        return int(len(text.encode("utf-8")) * self.tokens_per_byte) + 1

    # ---------- hooks ----------

    def attach(self, emit, state, resumed: bool = False):
        """Hook 1, once per turn before turn/start: follow this turn's state and events.

        Returns the emit for the loop to use: it passes every event on, and at turn/end
        calls end(). resumed: the turn continues an unfinished one (its message is
        already in user.md)."""
        self.state, self._emit, self.resumed = state, emit, resumed
        self._new_turn()
        self._ended = False
        self._mark = self._usage()

        def emit_and_follow(event, /, **data):   # positional: events have a field named kind
            emit(event, **data)
            if event == "turn/end":
                self.end(data.get("reason", ""), data.get("answer", ""))
        return emit_and_follow

    def start(self, goal, memory: str = "") -> str:
        """Hook 2, once per turn: append the user message to user.md (not when resumed) and
        return the [Memory] text: the memory given, plus the open handoff, the earlier user
        messages and the notes with evidence."""
        self.goal, self._base = str(goal), memory or ""
        try:
            folder = N.folder(self.folder.parent)          # created with a .gitignore
            if not self.resumed:
                N.append_user(folder / "user.md", self.goal)
        except OSError as e:
            self._warn(f"could not write {self.folder}: {e}")
        return self.memory()

    def before_think(self, step: int, prompt_of) -> str:
        """Hook 3, before every model call: re-read the written memory and, when the next
        prompt would pass compact_at, slim the old steps in place (emits context/compact).

        prompt_of(memory) renders the prompt the loop is about to send. Returns the
        [Memory] text for that call."""
        state = self.state
        self._follow(state)
        self._calibrate()
        memory = self.memory()
        prompt = prompt_of(memory)
        before = self.count(prompt)
        if self._due(before, state.steps) and len(state.steps) > self.keep_steps:
            return self._compact(step, prompt_of, before)
        self._sent = len(prompt.encode("utf-8"))
        return memory

    def end(self, reason: str, answer="") -> None:
        """Hook 4, at turn/end (through the attached emit) or when the run raised: any stop
        except done writes handoff.md; done closes an open handoff."""
        if self._ended:
            return
        self._ended = True
        if self.state is not None:
            self._follow(self.state)
        try:
            if reason == "done":
                N.close_handoff(self.folder / "handoff.md",
                                f"closed · the turn finished (done) at {N.now()}")
            else:
                self._write_handoff(f"open · stopped: {reason}", answer)
        except OSError as e:
            self._warn(f"could not write the handoff: {e}")

    # ---------- the [Memory] text ----------

    def memory(self) -> str:
        """The [Memory] text, read from the files: the same files give the same text."""
        base = self.base_memory() if self.base_memory is not None else self._base
        base = base or ""
        parts = [base] if base.strip() and base.strip() != "(none)" else []
        try:
            handoff = N.open_handoff(self.folder / "handoff.md")
            earlier = self._earlier_messages()
            facts = self._facts(base)
        except OSError as e:
            self._warn(f"could not read {self.folder}: {e}")
            handoff = earlier = facts = ""
        if handoff:
            parts.append("### Handoff (where the work stands; .mini_agent/handoff.md)\n" + handoff)
        if earlier:
            parts.append("### Earlier user messages (verbatim, oldest first; .mini_agent/user.md)\n"
                         + earlier)
        if facts:
            parts.append("### Notes with evidence\n" + facts)
        parts.append(HINT)
        return "\n\n".join(parts)

    def _earlier_messages(self) -> str:
        entries = N.user_messages(self.folder / "user.md")
        if entries and entries[-1][2].strip() == self.goal.strip():
            entries = entries[:-1]                      # this turn's message is the [Goal]
        blocks = [f"#{n} · {when}\n{text}" for n, when, text in entries]
        text, left = self._newest(blocks, self.user_share)
        if left:
            text = f"({left} older messages are in .mini_agent/user.md)\n\n" + text
        return text

    def _facts(self, base: str) -> str:
        mem = _running_memory.get()            # the running Agent's memory (None: memory is off)
        if mem is None or not mem.long_term_file.exists():
            return ""
        notes = [n for n in N.evidence_notes(mem.long_term_file.read_text(encoding="utf-8"))
                 if n not in base]
        text, left = self._newest(notes, self.notes_share)
        if left:
            text = f"({left} older notes with evidence are in {mem.long_term_file})\n\n" + text
        return text

    def _newest(self, blocks: list, share: float):
        """The newest blocks within the share of the window (at least one); and how many
        older ones were left out. Counted with the plain estimate, so the same files
        always give the same text."""
        budget, kept, used = self.limit(share), [], 0
        for block in reversed(blocks):
            size = C.count_tokens(block)
            if kept and used + size > budget:
                break
            kept.append(block)
            used += size
        kept.reverse()
        return "\n\n".join(kept), len(blocks) - len(kept)

    # ---------- compaction ----------

    @property
    def trigger(self) -> int:
        return self.limit(self.compact_at)

    @property
    def target(self) -> int:
        return self.limit(self.compact_to)

    def _due(self, before: int, steps: list) -> bool:
        # over the threshold, or so many steps that format_history would start dropping them
        if before <= self.trigger and _history_chars(steps) <= 0.9 * HISTORY_CHARS:
            return False
        # the last compaction could not reach the target: wait until the prompt has grown again
        return self._floor is None or before > self._floor + max(0, self.trigger - self.target)

    def _fits(self, after: int, steps: list) -> bool:
        return after <= self.target and _history_chars(steps) <= HISTORY_CHARS // 2

    def _compact(self, step: int, prompt_of, before: int) -> str:
        state = self.state
        original = list(state.steps)
        k = len(original) - self.keep_steps
        old, recent = original[:k], original[k:]
        self._write_handoff(f"open · compacted at step {step}")
        memory = self.memory()

        def measure(new_old):
            state.steps[:] = new_old + recent
            return self.count(prompt_of(memory))

        method, chosen = "mechanical", [C.trim(s) for s in old]
        after = measure(chosen)
        if not self._fits(after, state.steps):
            chosen = [C.fold(s) for s in old]
            after = measure(chosen)
        if not self._fits(after, state.steps):
            n = C.covered(old)
            summary = self._summarize(old)
            if summary:
                method, chosen = "llm", [C.compacted(summary, n)]
            else:
                chosen = [C.compacted(C.DROPPED.format(n=n), n)]
            after = measure(chosen)
        state.steps[:] = chosen + recent
        at = 0
        while at < min(k, len(chosen)) and original[at] == chosen[at]:
            at += 1
        # the event carries the replacement in full: the log alone rebuilds this context
        self._event("context/compact", step=step, before_tokens=before, after_tokens=after,
                    method=method, at=at, replaced=k - at, steps=chosen[at:])
        self._floor = None if after <= self.target else after
        self._seen = len(state.steps)
        self._sent = len(prompt_of(memory).encode("utf-8"))
        return memory

    def _summarize(self, steps: list):
        if self.llm is None:
            return None
        summary = C.summarize(self.llm, steps)
        self._mark = self._usage()           # the summary's tokens are not the next call's
        return summary

    def _usage(self) -> int:
        return int(getattr(self.llm, "total_prompt_tokens", 0) or 0)

    def _calibrate(self):
        """tokens per byte from the last call: its real input tokens / the bytes it sent."""
        used, self._mark = self._usage() - self._mark, self._usage()
        if used > 0 and self._sent > 0:
            self.tokens_per_byte = min(0.6, max(0.15, used / self._sent))

    # ---------- following the turn; the handoff ----------

    def _new_turn(self):
        self._seen = self._count = self._sent = self._mark = 0
        self._done, self._commands, self._files = [], [], {}
        self._todos, self._thought, self._floor = None, "", None

    def _follow(self, state):
        """Take note of the steps added since the last look (done list, files, plan)."""
        for s in state.steps[self._seen:]:
            self._count += C.covered([s])
            self._done.append(f"{self._count}. {C.line(s, 160)}")
            action, args = s.get("action"), s.get("args")
            if isinstance(args, dict) and not s.get("compacted"):
                path = args.get("path")
                if action == "write_file" and path:
                    self._files[str(path)] = "written"
                elif action == "read_file" and path:
                    self._files.setdefault(str(path), "read")
                elif action in ("bash", "run_python"):
                    command = str(args.get("cmd") or args.get("code") or "").strip().splitlines()
                    if command:
                        self._commands.append(C.cut(command[0], 100))
                elif action == "todo_write" and not str(s.get("result", "")).startswith("Error"):
                    self._todos = args.get("todos")
            if s.get("thought"):
                self._thought = str(s["thought"])
        self._seen = len(state.steps)

    def _write_handoff(self, status: str, answer=None):
        try:
            N.write_atomic(N.folder(self.folder.parent) / "handoff.md", self._handoff(status, answer))
        except OSError as e:
            self._warn(f"could not write the handoff: {e}")

    def _handoff(self, status: str, answer=None) -> str:
        goal = self.goal if len(self.goal) <= 600 else (
            self.goal[:600] + f"… ({len(self.goal)} characters; the full text is in user.md)")
        out = ["# Handoff", f"Status: {status}", f"Written: {N.now()}", "", "## Goal", goal, "",
               f"## Done ({self._count} steps)"]
        if len(self._done) > 30:
            out.append(f"({len(self._done) - 30} earlier steps are not listed; the session log has them)")
        out += self._done[-30:] or ["(nothing yet)"]
        out += ["", "## Next"]
        todos = [t for t in (self._todos or []) if isinstance(t, dict)
                 and t.get("status") in ("in_progress", "pending")]
        out += [f"- [{t.get('status')}] {t.get('content', '')}" for t in todos]
        if self._thought:
            out.append("Last thought: " + C.cut(" ".join(self._thought.split()), 400))
        if not todos and not self._thought:
            out.append("(not recorded)")
        written = [p for p, kind in self._files.items() if kind == "written"]
        read = [p for p, kind in self._files.items() if kind == "read"]
        out += ["", "## Files touched"]
        if written:
            out.append("written: " + ", ".join(written))
        if read:
            out.append("read: " + ", ".join(read))
        if self._commands:
            out.append("commands: " + " | ".join(self._commands[-10:]))
        if not (written or read or self._commands):
            out.append("(none)")
        if answer:
            out += ["", "## Stopped with", C.cut(str(answer), 500)]
        return "\n".join(out) + "\n"

    def _event(self, event: str, /, **data):
        if self._emit is not None:
            self._emit(event, **data)

    def _warn(self, message: str):
        self._event("lifecycle/warning", message=message)

    # ---------- between turns (plain terminal mode, full-screen interface) ----------

    def conversation(self, log) -> str:
        """The [Earlier in this session] text for the next turn, within session_budget.

        The projection already drops stream chunks and cuts tool results (mechanical);
        only if it is still over the budget, the model summarises older messages into
        the log (SessionLog.compact); then the newest lines that fit are kept. Every user
        message is in user.md, which memory() shows in full."""
        budget = self.limit(self.session_budget)
        text = _dialogue(log)
        if C.count_tokens(text) > budget and self.llm is not None:
            log.compact(self.llm)
            text = _dialogue(log)
        lines, used = [], 0
        for ln in reversed(text.split("\n")):
            used += C.count_tokens(ln) + 1
            if lines and used > budget:
                break
            lines.append(ln)
        return "\n".join(reversed(lines))


def _dialogue(log) -> str:
    out = []
    for m in log.derive_messages():
        if m["role"] == "user":
            out.append("User: " + str(m["content"]))
        elif m["role"] == "assistant":
            out.append("Assistant: " + str(m["content"]))
    return "\n".join(out)


def _history_chars(steps: list) -> int:
    """Characters of the steps as format_history writes them (without its cap)."""
    return sum(len(f"{i}. {s.get('action')}({s.get('args')}) → {s.get('result')}") + 1
               for i, s in enumerate(steps, 1))


def restore_steps(events) -> State:
    """The State of an unfinished turn, rebuilt from its events in the session log.

    events: from the turn's first turn/start (not resumed) to the end of the log; later
    segments that continued it start with turn/start {resumed: True}. Steps come back
    as the loop built them (step/think + tool/result; an invalid reply as
    "(invalid reply)"), and every context/compact replaces the steps it replaced, so
    the context is the compacted one, not the raw steps. A tool call without a result
    is not run again: it gets INTERRUPTED as its result. A thought without a call (the
    turn stopped before running it) is dropped. steps_done: the highest step number
    used, so a continued turn numbers its steps after it.
    """
    goal, steps, last = "", [], 0
    think = call = None

    def settle():                      # a segment ended here: what was in flight
        nonlocal think, call
        if think is not None and call is not None:
            steps.append({"thought": think.get("thought"), "action": call.get("name"),
                          "args": call.get("args"), "result": INTERRUPTED})
        think = call = None

    for event in events:
        kind, data = event.get("type"), event.get("data") or {}
        if isinstance(data.get("step"), int):
            last = max(last, data["step"])
        if kind == "turn/start":
            settle()
            if not data.get("resumed"):
                goal, steps, last = data.get("goal", ""), [], 0
        elif kind == "step/think":
            think, call = data, None
        elif kind == "tool/call":
            call = data
        elif kind == "tool/result" and think is not None:
            action = think.get("action")
            steps.append({"thought": think.get("thought"),
                          "action": action if action is not None else "(invalid reply)",
                          "args": think.get("args") if action is not None else {},
                          "result": data.get("content", "")})
            think = call = None
        elif kind == "context/compact":
            at = data.get("at", 0)
            steps[at:at + data.get("replaced", 0)] = data.get("steps", [])
        elif kind == "turn/end":
            think = call = None
    settle()
    return State(goal=goal, steps=steps, steps_done=last)
