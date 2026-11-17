"""Lifecycle: written memory, compaction before each model call, handoff at every stop.

The loop (core/agent/run.py) calls it in three places; with lifecycle=None nothing changes:

    emit = lifecycle.attach(emit, state, resumed)       # before turn/start; turn/end -> end()
    memory = lifecycle.start(goal, memory)              # records the user message
    memory = lifecycle.before_think(step, prompt_of)    # before every model call

Two limits are kept apart. The context window: compaction looks at the size of the
next prompt (prompt_of renders it; estimated as UTF-8 bytes / 3). The run's max_tokens:
a spending budget over the whole turn; compaction does not reset it, and when it stops
the turn, end() writes the handoff so that `mini_agent -c` continues with a fresh budget.

Every decision depends only on the steps, the files and the session log, never on what
one process happened to remember: a turn continued from the log (restore_steps) makes
the same decisions and sends the same prompt as the run that was cut off.
"""
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..core.state import State
from ..mem.memory import current as _running_memory
from . import compact as C
from . import notes as N

HINT = ("To keep a checked fact across compactions, call remember with content and evidence "
        "(how you know it: a command and its output, a file and line, a source).")
INTERRUPTED = ("Error: interrupted before this call finished (the session stopped); it may have run "
               "partly or not at all. Check its effect before repeating it.")
HARD = 0.8          # share of the window a prompt must stay under, even if the last steps are cut
SMALL = 2000        # results and arguments this short are never cut by the last resort


class Progress:
    """What a turn has done, noted from its steps as they ran (the handoff's lists)."""

    def __init__(self):
        self.count, self.done, self.commands, self.files = 0, [], [], {}
        self.todos, self.thought = None, ""

    def see(self, step: dict) -> None:
        self.count += C.covered([step])
        self.done.append(f"{self.count}. {C.line(step, 160)}")
        action, args = step.get("action"), step.get("args")
        if isinstance(args, dict) and not step.get("compacted"):
            path = args.get("path")
            if action == "write_file" and path:
                self.files[str(path)] = "written"
            elif action == "read_file" and path:
                self.files.setdefault(str(path), "read")
            elif action in ("bash", "run_python"):
                command = str(args.get("cmd") or args.get("code") or "").strip().splitlines()
                if command:
                    self.commands.append(C.cut(command[0], 100))
            elif action == "todo_write" and not str(step.get("result", "")).startswith("Error"):
                self.todos = args.get("todos")
        if step.get("thought"):
            self.thought = str(step["thought"])

    def copy(self) -> "Progress":
        other = Progress()
        other.count, other.done, other.commands = self.count, list(self.done), list(self.commands)
        other.files, other.todos, other.thought = dict(self.files), self.todos, self.thought
        return other


@dataclass
class Restored(State):
    """A State rebuilt from the session log, with what the lifecycle had noted of the turn."""
    progress: Progress = field(default_factory=Progress)
    floor: Optional[int] = None


class Lifecycle:
    """What lets one agent keep working on a long task.

    workspace     where .mini_agent/ (user.md, handoff.md) is written; Agent fills it in
    llm           the agent's model: its window sets the thresholds; summaries use it
    compact_at    compact when the next prompt would pass this (share of the window when
                  <= 1, else tokens)
    compact_to    ... down to this, if the mechanical passes can
    keep_steps    the last steps of a turn, never cut (unless one would not fit the window)
    session_budget  the earlier-session text between turns (conversation())
    base_memory   a function giving the [Memory] text without the lifecycle part (Agent sets
                  its own); without it, the memory given to start() is used

    message_id    set by the caller before a run (e.g. "<session file>#<seq>"): the next
                  start() records the message once, even if it is sent again (-c)
    overflow      after before_think: the estimated tokens of a prompt that, even compacted,
                  would not fit the window (the loop stops, reason context_full); else None
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
        self.state, self.goal, self.resumed = None, "", False
        self._emit, self._base, self._ended = None, "", True
        self.message_id, self.overflow = None, None
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

    @property
    def trigger(self) -> int:
        return self.limit(self.compact_at)

    @property
    def target(self) -> int:
        return self.limit(self.compact_to)

    @property
    def hard(self) -> int:
        return int(HARD * self.window)

    @property
    def history_tokens(self) -> float:
        """For format_history: no cap. The lifecycle keeps the prompt within the window by
        cutting what is too big, never by leaving steps out unseen."""
        return math.inf

    count = staticmethod(C.count_tokens)

    # ---------- hooks ----------

    def attach(self, emit, state, resumed: bool = False):
        """Hook 1, once per turn before turn/start: follow this turn's state and events.

        Returns the emit for the loop to use: it passes every event on, and at turn/end
        calls end(). resumed: the turn continues an unfinished one (its message is
        already in user.md); a state from restore_steps brings what was noted of it."""
        self.state, self._emit, self.resumed = state, emit, resumed
        self._new_turn()
        self._ended = False
        if resumed and isinstance(state, Restored):
            self.progress, self._floor = state.progress.copy(), state.floor
            self._seen = len(state.steps)

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
        message_id, self.message_id = self.message_id, None       # for this run only
        try:
            folder = N.folder(self.folder.parent)          # created with a .gitignore
            if not self.resumed:
                N.append_user(folder / "user.md", self.goal, message_id)
        except OSError as e:
            self._warn(f"could not write {self.folder}: {e}")
        return self.memory()

    def before_think(self, step: int, prompt_of) -> str:
        """Hook 3, before every model call: re-read the written memory and, when the next
        prompt would pass compact_at, slim the old steps in place (emits context/compact).

        prompt_of(memory) renders the prompt the loop is about to send. Returns the
        [Memory] text for that call; sets overflow if even then it would not fit the window."""
        self._follow(self.state)
        memory = self.memory()
        size = self.count(prompt_of(memory))
        if self._due(size):
            memory, size = self._compact(step, prompt_of, size)
        self.overflow = size if size > self.window else None
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
                N.close_handoff(N.inside(self.folder.parent) / "handoff.md",
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
            folder = N.inside(self.folder.parent)
            handoff = N.open_handoff(folder / "handoff.md")
            earlier = self._earlier_messages(folder)
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

    def _earlier_messages(self, folder: Path) -> str:
        entries = N.user_messages(folder / "user.md")
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
        older ones were left out."""
        budget, kept, used = self.limit(share), [], 0
        for block in reversed(blocks):
            size = self.count(block)
            if kept and used + size > budget:
                break
            kept.append(block)
            used += size
        kept.reverse()
        return "\n\n".join(kept), len(blocks) - len(kept)

    # ---------- compaction ----------

    def _due(self, before: int) -> bool:
        if before > self.hard:          # it would not fit: cut now (a summary still waits for _grown)
            return True
        if before <= self.trigger:
            return False
        return self._grown(before) and len(self.state.steps) > self.keep_steps

    def _grown(self, before: int) -> bool:
        """No compaction has missed the target yet, or the prompt has grown by
        compact_at - compact_to since the last one did: worth another full try."""
        return self._floor is None or before > self._floor + max(0, self.trigger - self.target)

    def _compact(self, step: int, prompt_of, before: int):
        """Slim the steps toward compact_to; returns the [Memory] text and the prompt's size."""
        state = self.state
        original = list(state.steps)
        k = max(0, len(original) - self.keep_steps)
        old, recent = original[:k], original[k:]
        previous = self._handoff_text()            # put back if nothing can be slimmed
        self._write_handoff(f"open · compacted at step {step}")
        memory = self.memory()

        def measure(new_old, new_recent=recent):
            state.steps[:] = new_old + new_recent
            return self.count(prompt_of(memory))

        method, chosen, full = "mechanical", [C.trim(s) for s in old], True
        after = measure(chosen)
        if after > self.target and old:
            chosen = [C.fold(s) for s in old]
            after = measure(chosen)
        if after > self.target and old:
            if self._grown(before):
                n, folded = C.covered(old), after
                summary = self._summarize(old)
                pointer = [C.compacted(C.DROPPED.format(n=n), n)]
                if summary and measure([C.compacted(summary, n)]) < folded:
                    method, chosen = "llm", [C.compacted(summary, n)]
                elif measure(pointer) < folded:
                    chosen = pointer
                after = measure(chosen)
            else:       # the last full try missed the target at about this size: no summary yet
                full = False
        kept, spent = list(recent), set()
        while after > self.hard:           # the last steps alone are too big: cut their largest part
            size, j, key = max(((len(str(s.get(key, ""))), j, key) for j, s in enumerate(kept)
                                for key in ("result", "args") if (j, key) not in spent),
                               default=(0, None, None))
            if size <= SMALL:
                break
            value = kept[j].get(key)
            smaller = C.cut(str(value), size // 2) if key == "result" else C.squeeze(value, size // 2)
            if len(str(smaller)) >= size:
                spent.add((j, key))
                continue
            kept[j] = {**kept[j], key: smaller}
            after = measure(chosen, kept)
        new = chosen + kept
        if new == original:                    # nothing to slim: no compaction, the handoff as it was
            state.steps[:] = original
            self._restore_handoff(previous)
            memory = self.memory()
            return memory, self.count(prompt_of(memory))
        state.steps[:] = new
        at = 0
        while at < min(len(original), len(new)) and original[at] == new[at]:
            at += 1
        tail = 0
        while tail < min(len(original), len(new)) - at and original[-1 - tail] == new[-1 - tail]:
            tail += 1
        if full:
            self._floor = None if after <= self.target else after
        # the event carries the replacement in full: the log alone rebuilds this context
        self._event("context/compact", step=step, before_tokens=before, after_tokens=after,
                    method=method, at=at, replaced=len(original) - at - tail,
                    steps=new[at:len(new) - tail], floor=self._floor)
        self._seen = len(state.steps)
        return memory, after

    def _summarize(self, steps: list):
        if self.llm is None:
            return None
        return C.summarize(self.llm, steps)

    # ---------- following the turn; the handoff ----------

    def _new_turn(self):
        self._seen, self._floor, self.progress, self.overflow = 0, None, Progress(), None

    def _follow(self, state):
        """Take note of the steps added since the last look (done list, files, plan)."""
        for s in state.steps[self._seen:]:
            self.progress.see(s)
        self._seen = len(state.steps)

    def _handoff_text(self):
        try:
            return N.read(N.inside(self.folder.parent) / "handoff.md")
        except OSError:
            return None

    def _restore_handoff(self, text):
        if text is None:
            return
        try:
            N.write_atomic(N.folder(self.folder.parent) / "handoff.md", text)
        except OSError as e:
            self._warn(f"could not write the handoff: {e}")

    def _write_handoff(self, status: str, answer=None):
        try:
            N.write_atomic(N.folder(self.folder.parent) / "handoff.md", self._handoff(status, answer))
        except OSError as e:
            self._warn(f"could not write the handoff: {e}")

    def _handoff(self, status: str, answer=None) -> str:
        p = self.progress
        goal = self.goal if len(self.goal) <= 600 else (
            self.goal[:600] + f"… ({len(self.goal)} characters; the full text is in user.md)")
        out = ["# Handoff", f"Status: {status}", f"Written: {N.now()}", "", "## Goal", goal, "",
               f"## Done ({p.count} steps)"]
        if len(p.done) > 30:
            out.append(f"({len(p.done) - 30} earlier steps are not listed; the session log has them)")
        out += p.done[-30:] or ["(nothing yet)"]
        out += ["", "## Next"]
        todos = [t for t in (p.todos or []) if isinstance(t, dict)
                 and t.get("status") in ("in_progress", "pending")]
        out += [f"- [{t.get('status')}] {t.get('content', '')}" for t in todos]
        if p.thought:
            out.append("Last thought: " + C.cut(" ".join(p.thought.split()), 400))
        if not todos and not p.thought:
            out.append("(not recorded)")
        written = [path for path, kind in p.files.items() if kind == "written"]
        read = [path for path, kind in p.files.items() if kind == "read"]
        out += ["", "## Files touched"]
        if written:
            out.append("written: " + ", ".join(written))
        if read:
            out.append("read: " + ", ".join(read))
        if p.commands:
            out.append("commands: " + " | ".join(p.commands[-10:]))
        if not (written or read or p.commands):
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

    def conversation(self, log, until_seq=None) -> str:
        """The [Earlier in this session] text for the next turn, within session_budget.

        The projection already drops stream chunks and cuts tool results (mechanical);
        only if it is still over the budget, the model summarises older messages into
        the log (SessionLog.compact); then the newest lines that fit are kept, the newest
        cut to head and tail if it alone is over. Every user message is in user.md, which
        memory() shows in full. until_seq: the text as it was up to that event (a turn
        continued with -c gets the text it started with); only read, never compacted."""
        budget = self.limit(self.session_budget)
        text = _dialogue(log, until_seq)
        if until_seq is None and self.count(text) > budget and self.llm is not None:
            log.compact(self.llm)
            text = _dialogue(log)
        lines, used = [], 0
        for ln in reversed(text.split("\n")):
            size = self.count(ln) + 1
            if used + size > budget:
                if not lines:                    # the newest line alone is over: head and tail
                    lines.append(C.fit(ln, budget - 1))
                break
            lines.append(ln)
            used += size
        return "\n".join(reversed(lines))


def _dialogue(log, until_seq=None) -> str:
    out = []
    for m in log.derive_messages(until_seq=until_seq):
        if m["role"] == "user":
            out.append("User: " + str(m["content"]))
        elif m["role"] == "assistant":
            out.append("Assistant: " + str(m["content"]))
    return "\n".join(out)


def restore_steps(events) -> Restored:
    """The State of an unfinished turn, rebuilt from its events in the session log.

    events: from the turn's first turn/start (not resumed) to the end of the log; later
    segments that continued it start with turn/start {resumed: True}. Steps come back
    as the loop built them (step/think + tool/result; an invalid reply as
    "(invalid reply)"), and every context/compact replaces the steps it replaced, so
    the context is the compacted one, not the raw steps. A tool call without a result
    is not run again: it gets INTERRUPTED as its result. A thought without a call (the
    turn stopped before running it) is dropped. steps_done: the highest step number
    used, so a continued turn numbers its steps after it. progress and floor: what the
    lifecycle had noted (from every step as it ran) and the state of its last compaction.
    """
    goal, steps, last, floor = "", [], 0, None
    progress = Progress()
    think = call = None

    def add(step):
        steps.append(step)
        progress.see(step)

    def settle():                      # a segment ended here: what was in flight
        nonlocal think, call
        if think is not None and call is not None:
            add({"thought": think.get("thought"), "action": call.get("name"),
                 "args": call.get("args"), "result": INTERRUPTED})
        think = call = None

    for event in events:
        kind, data = event.get("type"), event.get("data") or {}
        if isinstance(data.get("step"), int):
            last = max(last, data["step"])
        if kind == "turn/start":
            settle()
            if not data.get("resumed"):
                goal, steps, last, floor, progress = data.get("goal", ""), [], 0, None, Progress()
        elif kind == "step/think":
            think, call = data, None
        elif kind == "tool/call":
            call = data
        elif kind == "tool/result" and think is not None:
            action = think.get("action")
            add({"thought": think.get("thought"),
                 "action": action if action is not None else "(invalid reply)",
                 "args": think.get("args") if action is not None else {},
                 "result": data.get("content", "")})
            think = call = None
        elif kind == "context/compact":
            at = data.get("at", 0)
            steps[at:at + data.get("replaced", 0)] = data.get("steps", [])
            floor = data.get("floor")
        elif kind == "turn/end":       # a stop (interrupted, error, ...): what was in flight
            settle()
    settle()
    return Restored(goal=goal, steps=steps, steps_done=last, progress=progress, floor=floor)
