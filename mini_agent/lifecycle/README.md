# lifecycle — long tasks

Lets one agent work on a task longer than its context window and pick it up again
after a stop. Off by default; the core loop is unchanged without it.

```python
from mini_agent import Agent
from mini_agent.lifecycle import Lifecycle

agent = Agent(lifecycle=True)                     # defaults
agent = Agent(lifecycle=Lifecycle(compact_at=0.5, compact_to=0.25, keep_steps=6))
```

## What it writes

In `<workspace>/.mini_agent/` (with a `.gitignore`, so git leaves it alone):

| file | written | content |
|---|---|---|
| `user.md` | at the start of every turn | every user message, verbatim, numbered |
| `handoff.md` | at every compaction, and when a turn stops for any reason except done | goal, steps done, next step (the plan from `todo_write`, the last thought), files written and read, commands run |

When a turn ends with done, an open handoff is marked closed and is no longer shown.

Notes with evidence: `remember(content, evidence)`. `evidence` says how the agent knows
it (a command and its output, a file and line, a source) and is stored under the note in
the long-term memory file. Without `evidence`, `remember` behaves as before.

## What the model sees

The `[Memory]` section is read from the files before every model call:

- the agent's memory as before (long-term notes, today's log);
- the handoff, while it is open;
- the earlier user messages, verbatim (this turn's message is the `[Goal]`);
- the notes with evidence, in full;
- one line on how to record a note with evidence.

None of these is shortened by compaction. If the user messages or the notes alone would
take more than a quarter (messages) or 15% (notes) of the window, the oldest are left in
their files and a line says how many.

## Compaction

Before every model call the next prompt is measured (UTF-8 bytes / 3, calibrated by the
input tokens the provider reports). Over `compact_at` (default: half the window), the old
steps are slimmed until the prompt is under `compact_to` (default: a quarter):

1. long results and arguments are cut to head and tail;
2. each old step is folded to one line;
3. only if that is not enough, the model summarises them (the instruction and the
   "must be shorter" guard of `SessionLog.compact`);
4. if there is no usable summary, they are replaced by one line pointing to the handoff.

The last `keep_steps` steps are never changed. Each compaction emits
`context/compact {step, before_tokens, after_tokens, method, at, replaced, steps}`:
`steps` replaces `state.steps[at:at+replaced]` and holds the new entries in full, so the
session log alone rebuilds the compacted context.

`compact_at` and `compact_to` are shares of the window when <= 1, token counts otherwise.
They are about the size of one prompt. `max_tokens` of a run is something else: a budget
summed over all calls of a turn; compaction does not reset it. When it stops a turn, the
handoff is written and `mini_agent -c` continues with a fresh budget.

Between turns, `lifecycle.conversation(log)` gives the earlier-session text within
`session_budget` (default 10% of the window); it asks the model to summarise the log
(`SessionLog.compact`) only when the pruned text is over the budget.

## Continuing a turn

`restore_steps(events)` rebuilds the state of an unfinished turn from its session-log
events (`SessionLog.unfinished_turn()["events"]`), with every compaction applied. A tool
call that started but has no result is not run again; its result says it was
interrupted. A decision that never ran (a stop before the call) is dropped. Steps are
numbered on from the last one in the log.

```python
state = restore_steps(log.unfinished_turn()["events"])
agent.run(state.goal, state=state)
```

## The hooks in the loop

`run_agent(..., lifecycle=None, state=None)` (core/agent/run.py):

```python
emit = lifecycle.attach(emit, state, resumed)      # once; turn/end calls lifecycle.end()
memory = lifecycle.start(goal, memory)             # once; appends to user.md
memory = lifecycle.before_think(step, prompt_of)   # before every model call
```

`Agent.run` also calls `lifecycle.end("error" | "interrupted", ...)` when the run raises,
so a crash or Ctrl-C leaves a handoff. A `kill -9` leaves the last one written.
