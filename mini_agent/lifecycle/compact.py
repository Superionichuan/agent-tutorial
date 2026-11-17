"""Slimming the old steps of a turn so that the next prompt fits.

Only the old steps change (all but the last keep_steps). Lifecycle tries, in order,
until the projected prompt is under the target:
  1. trim     long results and arguments are cut to their head and tail;
  2. fold     each step becomes one line;
  3. the model summarises them (the instruction of SessionLog.compact; its convergence
     guard counted in tokens: the summary, as the model will see it, must be smaller
     than its source);
  4. without a usable summary, they become one line that points to the handoff.
Steps 1, 2 and 4 are mechanical. A step that was folded, summarised or dropped is
marked ("folded" / "compacted") and is not cut again.
"""
from ..llm.base import LLMError
from ..llm.offline import NO_RECORDING

BYTES_PER_TOKEN = 3        # a little over the tokens of English (about 4 bytes each); Chinese: one character
TRIM_RESULT, TRIM_ARG, TRIM_THOUGHT = 800, 400, 300
FOLD_RESULT, FOLD_ARG = 160, 80
SUMMARY_LINE = 500         # characters of each step given to the summary (as in SessionLog.compact)
SLACK = 32                 # room for the cut marker: cutting a cut text again changes nothing

# The instruction of SessionLog.compact, word for word
SUMMARY_PROMPT = ("Compress the following conversation history into concise bullet points "
                  "(keep: key facts / conclusions / user preferences / unfinished tasks; "
                  "drop: small talk and process details):\n")
DROPPED = ("{n} earlier steps were folded away to fit the context window; "
           "the handoff in [Memory] lists what was done.")


def count_tokens(text: str) -> int:
    """Estimated tokens of a text: its UTF-8 bytes / 3, rounded up."""
    return -(-len(text.encode("utf-8")) // BYTES_PER_TOKEN)


def cut(text, limit: int) -> str:
    """The text, or its head and tail with the number of characters cut between them."""
    text = str(text)
    if len(text) <= limit + SLACK:
        return text
    head, tail = limit * 2 // 3, limit // 3
    return f"{text[:head]}…[{len(text) - head - tail} chars cut]…{text[-tail:]}"


def fit(text: str, tokens: int) -> str:
    """The text, or its head and tail, within that many estimated tokens ("" if none fit)."""
    limit = len(text)
    while count_tokens(text) > tokens and limit > 0:
        limit = min(limit - 1, tokens * BYTES_PER_TOKEN) * 3 // 4
        text = cut(text, limit) if limit > 0 else ""
    return text


def shorten(value, limit: int):
    """Arguments with every string cut to the limit (dicts and lists keep their shape)."""
    if isinstance(value, str):
        return cut(value, limit)
    if isinstance(value, dict):
        return {k: shorten(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [shorten(v, limit) for v in value]
    return value


def longest(value) -> int:
    """The length of the longest string in arguments (0 if there is none)."""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return max((longest(v) for v in value.values()), default=0)
    if isinstance(value, list):
        return max((longest(v) for v in value), default=0)
    return 0


def squeeze(args, limit: int):
    """Smaller arguments: the longest strings cut to head and tail (the shape kept) while
    that helps; else the text of the arguments cut to the limit."""
    smaller = shorten(args, max(TRIM_ARG, longest(args) // 2))
    return smaller if len(str(smaller)) < len(str(args)) else cut(str(args), limit)


def render(step: dict) -> str:
    """The step as format_history shows it to the model."""
    return f"{step.get('action')}({step.get('args')}) → {step.get('result', '')}"


def kept(step: dict) -> bool:
    """Already folded or compacted: passes leave it as it is."""
    return bool(step.get("folded") or step.get("compacted"))


def trim(step: dict) -> dict:
    """Pass 1: long results, arguments and thoughts cut to head and tail."""
    if kept(step):
        return step
    return {**step, "thought": cut(step.get("thought") or "", TRIM_THOUGHT),
            "args": shorten(step.get("args"), TRIM_ARG),
            "result": cut(step.get("result", ""), TRIM_RESULT)}


def fold(step: dict) -> dict:
    """Pass 2: the step as one line (the first line of its result)."""
    if kept(step):
        return step
    result = str(step.get("result", ""))
    lines = result.strip().splitlines()
    first = lines[0].strip() if lines else ""
    more = len(result.strip()) - len(first)
    return {"thought": "", "action": step.get("action"), "args": shorten(step.get("args"), FOLD_ARG),
            "result": cut(first, FOLD_RESULT) + (f" …(+{more} chars)" if more > 0 else ""),
            "folded": True}


def covered(steps: list) -> int:
    """How many steps of the turn these entries stand for."""
    n = 0
    for s in steps:
        args = s.get("args")
        n += args.get("steps", 1) if s.get("compacted") and isinstance(args, dict) else 1
    return n


def compacted(text: str, n: int) -> dict:
    """One entry in place of n earlier steps (a summary, or a pointer to the handoff)."""
    return {"thought": "", "action": "compacted", "args": {"steps": n}, "result": text,
            "compacted": True}


def line(step: dict, limit: int) -> str:
    """action(args) → the first line of the result, cut to the limit."""
    result = str(step.get("result", "")).strip().splitlines()
    return cut(f"{step.get('action')}({step.get('args')}) → {result[0] if result else ''}", limit)


def summarize(llm, steps: list):
    """Pass 3: the model's summary of the steps, or None (failed, empty, or not shorter)."""
    lines = []
    for s in steps:
        if s.get("compacted"):
            lines.append(f"(summary of {covered([s])} earlier steps) {s.get('result', '')}")
        else:
            lines.append(cut(f"{s.get('action')}({s.get('args')}) → {s.get('result', '')}",
                             SUMMARY_LINE))
    source = "\n".join(lines)
    try:
        reply = llm(SUMMARY_PROMPT + source + "\n\nSummary:")
    except Exception:            # a summary is optional; the mechanical fallback follows
        return None
    if isinstance(reply, LLMError) or not isinstance(reply, str) or reply == NO_RECORDING:
        return None
    reply = reply.strip()
    # convergence guard, in tokens: the summary as the model will see it, framed as a step
    if not reply or count_tokens(render(compacted(reply, covered(steps)))) >= count_tokens(source):
        return None
    return reply
