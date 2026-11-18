# Tutorial 2 · Agentic Memory · Hands-on speaker guide

For the presenter of the hands-on part. Session: Monday 16 November 2026, 13:05–14:30,
NTU NEC Angsana Theatre 2. Hands-on: about 46 minutes.

Notebook: `notebooks/agent_memory_colab.ipynb`, one self-contained file.

## 1. The notebook in one paragraph

The notebook follows the four parts of the memory lecture, on the same agent participants
took apart in Tutorial 1 (`mini_agent`). Part 1 shows a model forgetting an earlier turn
and the session log bringing it back. Part 2 keeps notes for a later session, updates one
note (the code chooses ADD, UPDATE or NOOP and keeps the superseded note), and checks a
wrong note against its source. Part 3 compares the three routes into the context: full
replay, a rolling summary and retrieval. Part 4 puts quality and cost side by side and
sizes the KV cache with the formula from the lecture.

## 2. How it lines up with the lecture

| Notebook | Lecture slides (Agent_Memory_Part_1.pptx) | What participants see |
|---|---|---|
| Part 1 · Why agents need memory | Part 1, slides 1–2: inputs to the next decision; the minimal memory loop; state and event log | a separate run cannot answer; the same question inside the conversation is right; the log file |
| Part 2 · Scope, content, writing and reading | Part 2, slides 6–10: scope and content; consolidation; ADD/UPDATE/DELETE/NOOP; check applicability; failure diagnosis | notes survive to a new session; a correction gives UPDATE; a wrong note gives a confident wrong answer until the source is read; one rule in the notes makes the agent read it |
| Part 3 · Routes into the context | Part 3, slides 11–15: full replay, rolling summary, external storage and retrieval, combining routes | the history grows; the summary keeps the detail; 100 notes are cut to the budget; retrieval finds the one note |
| Part 4 · Cost and evaluation | Part 4, slides 16 and 20; Part 1, slide 5 | a quality-and-cost table for the routes; repeated input; the KV-cache formula |

Two ways to run the session:

- **Lecture, then hands-on.** All four parts of the lecture, then the notebook from top to
  bottom. Simple, but participants wait a long time before they type anything.
- **Interleaved (suggested).** After each part of the lecture, run the matching part of the
  notebook. Each notebook part opens with a short *Idea* paragraph in the lecture's words.

## 3. Before the session

1. Open the notebook in Colab (File > Upload notebook, or the link once it is published).
2. Run the setup cell with `PROVIDER = offline`. It should end with "Setup complete". Run
   the 🔑 workshop-key cell and press Enter at its prompt without a key: it should say "No
   key entered: nothing changed". (With Runtime > Run all, the notebook also waits at this
   prompt; Enter lets it continue.)
3. Run the notebook once from top to bottom. Offline, the whole run takes about a second
   after setup, and every answer matches the table in section 5.
4. Participants with the workshop key (the same key as in Tutorial 1, used the same way):
   run the setup cell (offline is fine), then run the 🔑 cell and paste the key. It should
   print "Connected: deepseek/deepseek-v4-flash via OpenRouter". A workshop key pasted into
   the setup cell's prompt instead (any `PROVIDER` except offline) works too: the setup cell
   prints "… via OpenRouter (workshop key)", and the 🔑 cell then says "Using the workshop
   key entered earlier".

## 4. Part by part

Times are for the hands-on only.

### Part 0 · Where we left off (4 min)

Run the first task and look at the prompt. Point at `[Memory]` and `[Steps taken so far]`,
both `(none)`. Say: everything today is about what the code puts into those two sections.

### Part 1 · Why agents need memory (7 min)

- `Chat` is the whole mechanism: *update* = append to the session log; *restore* = put the
  earlier turns back into the prompt.
- Ask the class to predict the separate run. Recorded answer: **"I don't have the referenced
  thickness. Please provide its value and unit so I can convert it to micrometres."** The
  model first tried to list files through the calculator (refused). This is the point: with
  no earlier turn in the prompt, the model cannot know which thickness is meant. Some models
  invent a number instead, which is worse; the lesson is the same.
- The same question inside the conversation: **"0.2 micrometres"**, and the prompt shows the
  earlier turn under `[Earlier in this session]`.
- The log cell prints the four events. State is what the next step needs; the log is what
  actually happened.

### Part 2 · Scope, content, writing and reading (14 min)

- The `remember` tool decides the operation in code: a new subject is ADD, the same subject
  with new content is UPDATE (the old note moves to `notes/superseded.md`), the same content
  is NOOP. Participants can open both files in the Colab file browser.
- Session 1: two ADDs, a preference (procedural) and a fact (semantic).
- Session 2, a new agent: **"723.15 K"**. Both notes were used: the fact, and the preference
  for kelvin.
- Session 3, the correction: **UPDATE**. The model reused the subject it had chosen, so the
  store holds one current note and the old one with the time it was superseded. If a live
  model picks a new subject instead, the result says ADD and the store holds two
  contradicting notes: a good moment to say why real systems compare a new note with
  similar stored ones.
- Session 4: **"753.15 K"**.
- Check before use, three runs that change one thing at a time:
  1. Without file access: **"Yes. Sample B was annealed at 450 °C, which is 50 °C below the
     500 °C glass substrate limit."** Confident and wrong.
  2. With file access: in the recording the agent read `lab/runs.csv` and answered **"No.
     Sample B was annealed at 540 °C, which is above the 500 °C glass substrate limit."**
  3. With file access and one rule in the notes (procedural memory): **"No. Sample B was
     annealed at 540 °C, which exceeds the 500 °C limit of its glass substrate."**

  Over the workshop key, run 2 behaves differently: in our test runs the agent trusted the
  note every time (it opened the file in 0 of 14 runs), and with the rule it checked every
  time (13 of 13). Over DeepSeek's own service it usually checked even without the rule
  (9 of 10). So most of the
  room will see run 2 go wrong and run 3 put it right: the clearest picture of procedural
  memory you will get. Offline participants see both runs check.
- `audit` prints MISMATCH, NOT CHECKED and, after the *Try it*, NO SOURCE. Message: a rule
  in the prompt is a request; a check in code always runs.

### Part 3 · Routes into the context (12 min)

- Twenty turns are written straight into the log (no model calls); the bar chart shows the
  history growing.
- Full replay: **"Your film is 0.2 micrometres (µm) thick."**
- Compaction: the recorded summary kept "200 nm" (printed as True), and the question after
  compaction is right: **"Your film is 0.2 micrometres (µm) thick."** A live summary may drop
  it; then the answer goes wrong, which is the lesson of this route.
- 100 notes: 6,787 characters in the file, 3,053 in the prompt, and S042 is not in it. The
  agent answers **"Sample S042 is not in the available notes, so its film thickness is not
  recorded/accessible."** Say clearly: here the code removes the
  middle, which is different from the lecture's "lost in the middle" result, where the model
  sees everything and uses the middle less well.
- Retrieval: the search finds S042 first; the agent answers **"282 nm"**. The *Try it*
  ("S-42 film") does not find it: keyword search matches spellings, not meanings.

### Part 4 · Cost and evaluation (6 min)

- The table (recorded run):

  | route | prompt tokens | right? |
  |---|---|---|
  | full replay | 2,377 | yes |
  | rolling summary | 1,377 | yes |
  | all notes, cut to budget | 2,542 | no |
  | retrieval, top 3 | 335 | yes |

  Counts add up every step of a run, from the provider (offline: as recorded). In the
  recording the first three runs took two steps each (a calculator call, then the answer)
  and retrieval took one, so retrieval is cheapest by far. The cut notes cost more than full
  replay because the notes fill the whole budget, about 1,200 tokens per step. Live numbers
  differ.
- Repeated input: one call grows from 680 to 2,417 characters of history over 20 turns, and
  all calls together send 26,328; after compaction one call sends 895.
- KV cache: 0.5 GiB at 4,000 tokens, 3.9 GiB at 32,000, 15.6 GiB at 128,000, for the example
  configuration (32 layers, 8 KV heads, 128 dimensions per head, 16-bit). The *Try it* with
  32 KV heads makes it four times larger, which links to the head-sharing point on slide 16.

### Part 5 · Summary (3 min)

The table maps the lecture's failure symptoms to what the class saw. End with the four
habits and the handover to tomorrow (MCP and skills) and Wednesday (evaluation).

## 5. Expected answers, offline

| Cell | Answer in the recording |
|---|---|
| Part 0 task | 391 |
| Part 1, separate run | asks for the value and its unit |
| Part 1, in the conversation | 0.2 micrometres |
| Part 2, session 2 | 723.15 K |
| Part 2, session 3 | UPDATE; 480 °C (753.15 K) |
| Part 2, session 4 | 753.15 K |
| Part 2, no file access | Yes … 450 °C … below the limit (wrong) |
| Part 2, with file access | No … 540 °C, above the limit |
| Part 2, with file access and the rule | No … 540 °C, exceeds the limit |
| Part 3, full replay | 0.2 micrometres (µm) |
| Part 3, after compaction | 0.2 micrometres (µm) |
| Part 3, all notes | not in the available notes |
| Part 3, retrieval | 282 nm |

Offline, every cell gives exactly these answers. Live, the wording and sometimes the
outcome change; the notebook text tells participants what to look for either way.

**Live runs over the workshop key.** OpenRouter passes each request to one of several hosts
of the same model, so answers vary more between participants than between two offline runs.
In eight complete test runs over the workshop key (16 November), every run finished without
errors, and every teaching point came out as above except two: the separate run in Part 1
always asked for the value, and in Part 2 the agent with file access trusted the wrong note
until the rule was added (see Part 2).

## 6. If something goes wrong

| What you see | Cause | What to do |
|---|---|---|
| `[offline] This prompt is not in the recording` | a cell was edited, or cells were run out of order | rerun the setup cell and then the parts in order; own variants need a live provider |
| `NameError` (for example `notes`, `chat3`, `re`) | a part was run before the part that defines it | run the parts in order from Part 0 |
| the 🔑 workshop-key cell waits for input | it asks for the workshop key | paste the key and press Enter; without a key, just press Enter |
| a different answer from the table | a live provider | expected; offline gives the table exactly |
| no network in the room | — | use `offline`; it needs no network after the setup cell has installed the package |
| `python -m mini_agent` in a Colab terminal fails (`ModuleNotFoundError`, or `Connection refused`) | the setup cell has not run in this runtime yet | run the setup cell (and the 🔑 cell for a workshop key), then run the command again |

## 7. Questions you may get

- **Is this how chat assistants remember?** The details differ, but the principle is the
  same: text the system chose is put back into the model's context.
- **Why not just use a bigger context window?** Every token is sent again on every call
  (Part 4, repeated input), costs memory on the device (KV cache), and evidence in a long
  prompt can be used less well (lecture, Part 1).
- **Why exact subjects instead of a database?** To keep the write rule visible in a dozen
  lines. Real systems compare a new note with similar records, and some let a model choose
  the operation.
- **Can the agent edit its own notes?** Yes: `remember` writes the file. That is why the
  write step needs checks in code (Part 2, `audit`).
- **Can I use the agent from a terminal, as in Tutorial 1?** Yes. After the setup cell (and
  the 🔑 cell, for a workshop key), open a terminal in Colab and run
  `python -m mini_agent "your task"`, or `python -m mini_agent` for a conversation. The
  setup saves the key and the model in a `.env` file on that Colab machine, which is how the
  new process finds them. The terminal runs the plain agent of Tutorial 1, without this
  notebook's notes.
- **Why can't I choose another model, as in Tutorial 1?** Part 2 depends on how this model
  behaves over the workshop key: it trusted the wrong note in all 14 test runs without the
  rule, and checked its source in all 13 with it. Another model might check on its own or
  ignore the rule, and the demonstration would no longer show the difference.
