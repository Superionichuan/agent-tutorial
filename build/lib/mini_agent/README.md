# Mini Agent — architecture

A minimal real agent. Every file has a single responsibility.

## Design philosophy

```
The LLM is not the decision engine — it is a translator.
Execution is done by deterministic functions.
One function per file: smallest change, biggest reuse.
```

## Layout

```
mini_agent/
├── __init__.py          # package entry
├── __main__.py          # python -m entry
│
├── prompts/             # prompt templates (user-editable)
│   └── agent.md         # main loop template
│
├── llm/                 # LLM backends (pluggable)
│   ├── base.py          # interface + token accounting
│   ├── factory.py       # create from .env config
│   └── backends/        # ollama / qwen / deepseek / claude
│
├── core/                # the agent core (pure functions)
│   ├── agent/           # Agent class + the loop (run/think/check_done)
│   ├── prompt/          # prompt assembly
│   ├── parser/          # LLM response -> (thought, action, args)
│   ├── executor/        # tool dispatch
│   ├── state/           # step history
│   └── config/          # agent config
│
├── tools/               # tool registry (math / file / shell / memory / todo)
├── mem/                 # durable memory (long-term + daily log + distill)
├── skills/              # skill plugins (SKILL.md loader)
├── session.py           # append-only event log + projections (fork/replay/compaction)
├── ui/                  # plain REPL + textual TUI (theme, pets)
└── cli/                 # entry: plain (default) / --tui / one-shot
```

## The loop (core/agent/run.py)

```
turn/start
  step/start
    think   -> (thought, action, args)     one LLM request
    done?   -> answer
    execute -> tool result
  step/end
turn/end
```

Every node emits an event — one stream feeds the console renderer, the
rich/textual UIs, and the SessionLog (persistence -> resume/fork/replay).

## Module -> course map (for the notebook series)

| Module | One-liner | Covered in |
|--------|-----------|-----------|
| core/agent | the ReAct loop (think/act/observe) | T1 |
| core/{prompt,parser,executor,state} | the four helpers around the loop | T1 |
| tools/ | deterministic functions + registry | T1 |
| prompts/ | editable templates (agent / coding profiles) | T1 exercise |
| session.py | event log: projection, fork, replay, compaction | T1 Act 5 + T3 |
| mem/ | durable memory + distill | T2 |
| skills/ | SKILL.md plugins | T2 |
| llm/ | pluggable backends, streaming, token accounting | T3 |
| ui/ | two renderers over one event stream | T3 |
| cli/ | entry: plain / --tui / --coding / -c | T3 |
