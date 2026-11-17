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
| `user.md` | at the start of every turn | every user message, verbatim; each entry's header gives its length, so no text inside a message can end it (an entry whose length no longer holds, edited by hand or cut short by a crash, is read up to the next header) |
| `handoff.md` | at every compaction, and when a turn stops for any reason except done | goal, steps done, next step (the plan from `todo_write`, the last thought), files written and read, commands run; the session that wrote it |

When a turn ends with done, an open handoff is marked closed and is no longer shown.
Nothing is read or written through a symbolic link: `.mini_agent` must be a real folder of
the workspace, otherwise the files are not used and a `lifecycle/warning` event says why.
On POSIX systems every file is opened through a handle on that folder, so a link swapped
in while a file is opened is not followed either.

A message sent again, such as a task continued with `mini_agent -c` before it started, is
recorded once when the caller names it: `lifecycle.message_id = "<session file>#<seq>"`
before the run (the command line does). The id goes into the entry's header; its part
before `#` names the session (one interactive session, or one one-shot task). Without a
name, one `Lifecycle` object is one session.

Notes with evidence: `remember(content, evidence)`. `evidence` says how the agent knows
it (a command and its output, a file and line, a source) and is stored under the note in
the long-term memory file. Without `evidence`, `remember` behaves as before.

## What the model sees

The `[Memory]` section is read from the files before every model call:

- the agent's memory as before (long-term notes, today's log);
- the handoff, while it is open and was written in this session (or the turn continues
  the task it describes);
- the earlier user messages of this session, verbatim (this turn's message is the `[Goal]`);
- the notes with evidence, in full;
- one line on how to record a note with evidence.

Messages and an open handoff of earlier sessions, such as earlier one-shot tasks in the
same folder, stay in the files and are not shown: a new task does not take them as
requests. A task continued with `-c` is shown its own.

None of these is shortened by compaction. If the user messages or the notes alone would
take more than a quarter (messages) or 15% (notes) of the window, the oldest are left in
their files and a line says how many.

## Compaction

Before every model call the next prompt is measured (UTF-8 bytes / 3: a little over the
tokens of English, about one token per Chinese character; not calibrated by the provider's
counts, so that a continued turn makes the same decisions). Over `compact_at` (default:
half the window), the old steps are slimmed until the prompt is under `compact_to`
(default: a quarter):

1. long results and arguments are cut to head and tail;
2. each old step is folded to one line;
3. only if that is not enough, the model summarises them (the instruction and the
   "must be shorter" guard of `SessionLog.compact`);
4. if there is no usable summary, they are replaced by one line pointing to the handoff.

A summary is used only if it makes the prompt smaller than the folded steps, and the
pointer only if it does. The last `keep_steps` steps are never changed, with one exception:
if the prompt would still take more than 80% of the window, the largest of their results
and arguments are cut to head and tail. With a lifecycle, format_history has no cap of its
own, so no step is left out of the prompt unseen. If even then the prompt would not fit
the window, nothing is sent: the turn stops with reason `context_full` and the handoff is
written (a shorter task, or a model with a larger window, is needed).

If the last compaction could not reach `compact_to`, the next model summary waits until
the prompt has grown by `compact_at - compact_to` (above 80% of the window only the
mechanical passes run meanwhile), so a summary is not requested at every step; a pass
that changes nothing emits nothing. Each compaction emits
`context/compact {step, before_tokens, after_tokens, method, at, replaced, steps, floor}`:
`steps` replaces `state.steps[at:at+replaced]` and holds the new entries in full, so the
session log alone rebuilds the compacted context.

`compact_at` and `compact_to` are shares of the window when <= 1, token counts otherwise.
They are about the size of one prompt. `max_tokens` of a run is something else: a budget
summed over all calls of a turn, summaries included; compaction does not reset it. When
it stops a turn (also right after a summary that used it up), the handoff is written and
`mini_agent -c` continues with a fresh budget.

Between turns, `lifecycle.conversation(log)` gives the earlier-session text within
`session_budget` (default 10% of the window; the newest line is cut to head and tail if it
alone is over); it asks the model to summarise the log (`SessionLog.compact`) only when
the pruned text is over the budget. `conversation(log, until_seq=n)` gives the text as it
was up to event `n` (a continued turn gets the text it started with), only reading.

## Continuing a turn

`restore_steps(events)` rebuilds the state of an unfinished turn from its session-log
events (`SessionLog.unfinished_turn()["events"]`), with every compaction applied. A tool
call that started but has no result is not run again, whatever stopped the turn; its
result says it was interrupted. A decision that never ran (a stop before the call) is
dropped. Steps are
numbered on from the last one in the log. The returned state also carries what the
lifecycle had noted of the turn (the handoff's lists, from every step as it ran) and the
state of its last compaction, so the continued turn sends the prompt the interrupted run
would have sent and its handoff keeps everything the earlier one listed.

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
