"""Offline mode: answer from recorded model responses.

OfflineLLM replays responses recorded from a live run, so the tutorial runs
without an API key or network access. RecordingLLM wraps a live backend and
appends every (prompt, response) pair to a recording file.

Prompts are matched after removing what differs between machines and days:
the workspace path, the memory section, and dates and times in tool output.
A direct question (a prompt without a goal) must match exactly. For an agent
step without an exact match, the recorded step with the same goal, the same
step number, and the most similar history is used. For an agent step the file
stores a hash of the prompt, the goal, and the step history with the workspace
path removed; the tool catalog and the memory section are not stored. For a
direct question it stores the normalized question text.
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


def _workspace(prompt: str) -> str:
    m = re.search(r"^Workspace: (.*)$", prompt, re.M)
    ws = m.group(1).strip() if m else ""
    return ws if len(ws) > 1 else ""      # never replace a bare "/"


def _mask_output(text: str, ws: str = "") -> str:
    """Tool output: remove the workspace path, the home directory, dates and times."""
    if ws:
        text = text.replace(ws, "<ws>")
    text = text.replace(str(Path.home()), "~")
    text = re.sub(r"\d{4}-\d{2}-\d{2}", "<date>", text)
    return re.sub(r"\b\d{1,2}:\d{2}(:\d{2})?\b", "<time>", text)


def normalize(text: str) -> str:
    """Remove the parts of a prompt that vary between machines and days."""
    ws = _workspace(text)
    t = text.replace(ws, "<ws>") if ws else text
    t = t.replace(str(Path.home()), "~")
    t = re.sub(r"(\[Memory\]\n).*?(\n\n\[Steps taken so far\])", r"\1<memory>\2", t, flags=re.S)
    h = re.search(r"(\[Steps taken so far\]\n)(.*?)(?=\n\n\[|\Z)", t, re.S)
    if h:
        t = t[:h.start(2)] + _mask_output(h.group(2)) + t[h.end(2):]
    return re.sub(r"\s+", " ", t).strip()


def describe(prompt: str) -> dict:
    """Key, goal, and step history of a prompt (what a recording stores)."""
    norm = normalize(prompt)
    goal = re.sub(r"\s+", " ", _section(prompt, "Goal")).strip()
    raw_history = _section(prompt, "Steps taken so far")
    steps = len(re.findall(r"^\d+\. ", raw_history, re.M))
    history = re.sub(r"\s+", " ", _mask_output(raw_history, _workspace(prompt))).strip()
    rec = {"key": hashlib.sha256(norm.encode()).hexdigest()[:24],
           "goal": goal, "steps": steps, "history": history if goal else ""}
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

    def _next(self, key, candidates):
        """Return candidates in recorded order on repeated requests."""
        k = (str(self.path), key)
        i = OfflineLLM._cursor.get(k, 0)
        OfflineLLM._cursor[k] = i + 1
        return candidates[min(i, len(candidates) - 1)]

    def _match(self, prompt: str):
        q = describe(prompt)
        same = [r for r in self.records if r["key"] == q["key"]]
        if same:
            return self._next(q["key"], same)
        if not q["goal"]:                 # a direct question must match exactly
            return None
        pool = [r for r in self.records if r["goal"] == q["goal"] and r["steps"] == q["steps"]]
        scored = [(SequenceMatcher(None, r["history"], q["history"]).ratio(), r) for r in pool]
        best = max((sc for sc, _ in scored), default=0)
        if best < 0.5:
            return None
        return self._next("fallback:" + q["key"], [r for sc, r in scored if sc == best])

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
            text = self.call(prompt)
            try:
                on_chunk(text, kind="content")
            except TypeError:
                on_chunk(text)
            return text
        return self._exchange(prompt, lambda: self.inner.call_stream(prompt, on_chunk))

    def reset_usage(self):
        super().reset_usage()
        self.inner.reset_usage()
