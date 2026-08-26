"""session.py — session event log: append-only JSONL + projections

Core idea (one line): the log is the single source of truth; all state is a projection.

- Whatever the model sees must already be in the log (model-visible means logged)
- derive_messages() projects the model history from the log — there is no second
  copy of "conversation state" anywhere
- fork / resume / replay / compaction are all derived from the same stream

Event taxonomy:
    turn/start           a turn begins (turn = all work triggered by one user input)
    step/start           a step begins (step = one model request + the tools it calls)
    user/message         user message (only counts once appended to the log)
    assistant/chunk      streaming fragment (preserves replay fidelity; not in model history)
    assistant/message    complete assistant message
    tool/call            tool invocation (name, args, call_id)
    tool/result          tool result (paired with the call via call_id)
    step/end             step ends
    turn/end             turn ends
    compaction/summary   compaction checkpoint (summary replaces older history)
    plan/mode            plan-mode flag (log-only state, folded on resume)
    todo/write           todo list snapshot (whole-list replace)

Usage:
    log = SessionLog("run1.jsonl")
    log.append("user/message", content="hi")
    messages = log.derive_messages()                  # -> messages for the LLM
    child = log.fork("run1b.jsonl", boundary_seq=5)   # branch at event 5
    log.replay(on_event=print)                        # replay event by event
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator


class SessionLog:
    """Append-only event log. The file is the session; one JSON line = one event."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = self._last_seq()

    # ---------- write ----------

    def append(self, type: str, **data: Any) -> dict:
        """Append one event and flush to disk. Returns the full event (with seq/time)."""
        self._seq += 1
        event = {
            "seq": self._seq,
            "time": round(time.time(), 3),
            "type": type,
            "data": data,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    # ---------- read ----------

    def events(self, until_seq: int | None = None) -> Iterator[dict]:
        """Yield events in order; until_seq is inclusive, None = all."""
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if until_seq is not None and event["seq"] > until_seq:
                    return
                yield event

    def _last_seq(self) -> int:
        last = 0
        for event in self.events():
            last = event["seq"]
        return last

    # ---------- projection: model history ----------

    def derive_messages(self, until_seq: int | None = None,
                        prune_tool_chars: int = 400) -> list[dict]:
        """Project LLM messages from the log. This is the model's only source of truth.

        Projection rules:
        - user/message      -> {"role": "user", ...}
        - assistant/message -> {"role": "assistant", ...}
        - tool/result       -> {"role": "tool", ...} (long results pruned in projection)
        - assistant/chunk   -> skipped (chunks serve UI replay; the model sees the merge)
        - turn/step bounds  -> skipped (structural events, not conversation)
        - compaction/summary-> summary replaces everything before its boundary
        """
        # Find the last compaction checkpoint: its summary replaces history
        # before the boundary (until_seq).
        last_comp = None
        for event in self.events(until_seq):
            if event["type"] == "compaction/summary":
                last_comp = event
        messages: list[dict] = []
        start_after = 0
        if last_comp is not None:
            messages.append({"role": "user", "content":
                f"<compacted-summary>\n{last_comp['data']['summary']}\n</compacted-summary>"})
            start_after = last_comp["data"].get("until_seq", last_comp["seq"])
        for event in self.events(until_seq):
            t, d = event["type"], event["data"]
            if event["seq"] <= start_after or t == "compaction/summary":
                continue
            if t == "user/message":
                messages.append({"role": "user", "content": d["content"]})
            elif t == "assistant/message":
                msg: dict = {"role": "assistant", "content": d.get("content", "")}
                if d.get("tool_calls"):
                    msg["tool_calls"] = d["tool_calls"]
                messages.append(msg)
            elif t == "tool/result":
                content = str(d["content"])
                if len(content) > prune_tool_chars:
                    # Tool-result pruning (projection layer only — the log keeps
                    # the full original): head + marker + tail.
                    head, tail_n = prune_tool_chars * 2 // 3, prune_tool_chars // 3
                    content = (content[:head]
                               + f"\n...[{len(content) - head - tail_n} chars pruned]...\n"
                               + content[-tail_n:])
                messages.append({
                    "role": "tool",
                    "tool_call_id": d["call_id"],
                    "content": content,
                })
        return messages

    # ---------- compaction: summarize old history into a checkpoint ----------

    def compact(self, llm, keep_tail: int = 6) -> str | None:
        """Summarize older history into a compaction checkpoint event.

        - Keeps the most recent keep_tail messages (boundary aligned so a bare
          tool result is never stranded at the head of the tail)
        - The summary must be shorter than its source (convergence guard)
        - Original events stay in the log forever — compaction only changes
          the projection
        """
        # Collect message-bearing events (with seq) to determine the boundary.
        items = []
        for ev in self.events():
            t, d = ev["type"], ev["data"]
            if t == "user/message":
                items.append((ev["seq"], "user", d["content"]))
            elif t == "assistant/message":
                items.append((ev["seq"], "assistant", d.get("content", "")))
            elif t == "tool/result":
                items.append((ev["seq"], "tool", str(d["content"])[:300]))
            elif t == "compaction/summary":
                items = [(ev["seq"], "summary", d["summary"])]
        if len(items) <= keep_tail + 2:
            return None
        old, tail = items[:-keep_tail], items[-keep_tail:]
        while tail and tail[0][1] == "tool" and old:
            tail.insert(0, old.pop())          # align to tool call/result pairing
        if not old:
            return None
        until_seq = tail[0][0] - 1 if tail else old[-1][0]
        src = "\n".join(f"{r}: {str(c)[:500]}" for _, r, c in old)
        summary = llm(
            "Compress the following conversation history into concise bullet points "
            "(keep: key facts / conclusions / user preferences / unfinished tasks; "
            "drop: small talk and process details):\n"
            + src + "\n\nSummary:")
        if not summary or len(summary) >= len(src):
            return None                        # convergence guard
        self.append("compaction/summary", summary=summary, covered=len(old),
                    until_seq=until_seq)
        return summary

    # ---------- derived: fork / replay / stats ----------

    def fork(self, new_path: str | Path, boundary_seq: int | None = None) -> "SessionLog":
        """Branch a new session at boundary_seq (inclusive). None = copy everything.

        Forking is copying a prefix — because state is a projection, copying the
        log prefix copies the entire state at that moment.
        """
        child = SessionLog(new_path)
        if child._seq:
            raise FileExistsError(f"fork target already has content: {new_path}")
        for event in self.events(boundary_seq):
            with child.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        child._seq = child._last_seq()
        return child

    def replay(self, on_event: Callable[[dict], None], speed: float = 0.0) -> int:
        """Replay events one by one (UIs use this for replay animation).
        speed > 0 plays at that multiple of the real inter-event intervals."""
        prev_time = None
        n = 0
        for event in self.events():
            if speed > 0 and prev_time is not None:
                delay = (event["time"] - prev_time) / speed
                if delay > 0:
                    time.sleep(min(delay, 2.0))  # cap to avoid long waits
            prev_time = event["time"]
            on_event(event)
            n += 1
        return n

    def stats(self) -> dict:
        """Whole-log statistics projection (turns/steps/tool calls/wall time)."""
        s = {"turns": 0, "steps": 0, "tool_calls": 0, "events": 0,
             "started": None, "ended": None}
        for event in self.events():
            s["events"] += 1
            t = event["type"]
            if t == "turn/start":
                s["turns"] += 1
            elif t == "step/start":
                s["steps"] += 1
            elif t == "tool/call":
                s["tool_calls"] += 1
            if s["started"] is None:
                s["started"] = event["time"]
            s["ended"] = event["time"]
        s["wall_seconds"] = round((s["ended"] or 0) - (s["started"] or 0), 3)
        return s


def new_call_id() -> str:
    """Short id for pairing tool calls with results."""
    return uuid.uuid4().hex[:8]
