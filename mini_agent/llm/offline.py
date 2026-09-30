"""Offline mode: answer from recorded model responses.

OfflineLLM replays responses recorded from a live run, so the tutorial runs
without an API key or network access. RecordingLLM wraps a live backend and
appends every (prompt, response) pair to a recording file.

Prompts are matched after removing what differs between machines and days
(the workspace path, the memory section, dates, times). If no exact match
exists, the recorded step with the same goal, the same step number, and the
most similar history is used. For an agent step the file stores a hash of
the prompt, the goal, and the step history, not the tool catalog, the memory
section, or the workspace path. For a direct question (a prompt without a
goal) it stores the normalized question text.
"""
import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from .base import BaseLLM

DEFAULT_RECORDING = Path(__file__).parent / "recordings" / "tutorial.jsonl"
NO_RECORDING = ("[offline mode] No recorded response matches this prompt. Offline "
                "mode replays the tutorial's recorded examples only; select a "
                "provider and API key in the setup cell to run new prompts.")


def _section(prompt: str, name: str) -> str:
    m = re.search(r"\[" + re.escape(name) + r"\]\n(.*?)(?=\n\n\[|\Z)", prompt, re.S)
    return m.group(1).strip() if m else ""


def normalize(text: str) -> str:
    """Remove the parts of a prompt that vary between machines and days."""
    t = re.sub(r"^Workspace: .*$", "Workspace: <ws>", text, flags=re.M)
    t = re.sub(r"(\[Memory\]\n).*?(\n\n\[Steps taken so far\])", r"\1<memory>\2", t, flags=re.S)
    t = t.replace(str(Path.home()), "~")
    t = re.sub(r"\d{4}-\d{2}-\d{2}", "<date>", t)
    t = re.sub(r"\b\d{1,2}:\d{2}(:\d{2})?\b", "<time>", t)
    return re.sub(r"\s+", " ", t).strip()


def describe(prompt: str) -> dict:
    """Key, goal, and step history of a prompt (what a recording stores)."""
    norm = normalize(prompt)
    goal = normalize(_section(prompt, "Goal"))
    raw_history = _section(prompt, "Steps taken so far")
    history = normalize(raw_history)
    steps = len(re.findall(r"^\d+\. ", raw_history, re.M))
    rec = {"key": hashlib.sha256(norm.encode()).hexdigest()[:24],
           "goal": goal, "steps": steps, "history": history}
    if not goal:                      # a direct question, not an agent step
        rec["text"] = norm[:2000]
    return rec


class OfflineLLM(BaseLLM):
    """Replays recorded responses; no network access."""

    _cursor: dict = {}                # shared: repeated prompts replay in order

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else DEFAULT_RECORDING
        self.records = [json.loads(line) for line in self.path.read_text().splitlines()
                        if line.strip()] if self.path.exists() else []
        source = self.records[0].get("model", "") if self.records else ""
        self.model = f"offline (recorded from {source})" if source else "offline"
        self.context_limit = 128_000
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _match(self, prompt: str):
        q = describe(prompt)
        same = [r for r in self.records if r["key"] == q["key"]]
        if same:
            i = OfflineLLM._cursor.get(q["key"], 0)
            OfflineLLM._cursor[q["key"]] = i + 1
            return same[min(i, len(same) - 1)]
        if q["goal"]:
            pool = [r for r in self.records if r["goal"] == q["goal"] and r["steps"] == q["steps"]]
            field = "history"
        else:
            pool = [r for r in self.records if not r["goal"]]
            field = "text"
        scored = [(SequenceMatcher(None, r.get(field, ""), q.get(field, "")).ratio(), r)
                  for r in pool]
        best = max(scored, key=lambda x: x[0], default=(0, None))
        return best[1] if best[0] >= (0.5 if q["goal"] else 0.9) else None

    def call(self, prompt: str) -> str:
        rec = self._match(prompt)
        if rec is None:
            return NO_RECORDING
        self.total_prompt_tokens += rec.get("prompt_tokens", 0)
        self.total_completion_tokens += rec.get("completion_tokens", 0)
        return rec["response"]

    def call_stream(self, prompt: str, on_chunk) -> str:
        text = self.call(prompt)
        try:
            on_chunk(text, kind="content")
        except TypeError:
            on_chunk(text)
        return text


class RecordingLLM(BaseLLM):
    """Wraps a live backend and appends each exchange to a recording file."""

    def __init__(self, inner: BaseLLM, path: str | Path):
        self.inner = inner
        self.path = Path(path)
        self.model = getattr(inner, "model", "")
        self.context_limit = getattr(inner, "context_limit", 128_000)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _exchange(self, prompt: str, send) -> str:
        p0 = self.inner.total_prompt_tokens
        c0 = self.inner.total_completion_tokens
        response = send()
        dp = self.inner.total_prompt_tokens - p0
        dc = self.inner.total_completion_tokens - c0
        self.total_prompt_tokens += dp
        self.total_completion_tokens += dc
        rec = describe(prompt)
        rec.update(response=response.replace(str(Path.home()), "~"),
                   prompt_tokens=dp, completion_tokens=dc, model=self.model)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return response

    def call(self, prompt: str) -> str:
        return self._exchange(prompt, lambda: self.inner.call(prompt))

    def call_stream(self, prompt: str, on_chunk) -> str:
        if not hasattr(self.inner, "call_stream"):
            return self.call(prompt)
        return self._exchange(prompt, lambda: self.inner.call_stream(prompt, on_chunk))

    def reset_usage(self):
        super().reset_usage()
        self.inner.reset_usage()
