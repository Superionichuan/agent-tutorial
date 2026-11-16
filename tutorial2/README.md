# AgentForge Tutorial 2: Agentic Memory (hands-on)

Hands-on Workshop on AI Agents as Scientific Research Assistants
Nanyang Technological University · Supported by Schmidt Sciences

`notebooks/agent_memory_colab.ipynb` adds memory to `mini_agent`, the agent from
Tutorial 1, following the four parts of the memory lecture: why agents need
memory (Part 1), scope, content, writing and reading (Part 2), the routes into
the context (Part 3), and cost and evaluation (Part 4). It is a single
self-contained file: open it in Google Colab (File > Upload notebook, or the
Colab link once published), run the setup cell first, then the parts in order.

- `offline` needs no API key: every model call is answered from a recording
  of a live run, embedded in the setup cell; the tools run for real.
- With your own key (DeepSeek, Qwen, Claude), or the workshop key through the
  optional OpenRouter cell, the same cells run live.

## Files

| Path | What it is |
|---|---|
| `notebooks/agent_memory_colab.ipynb` | the notebook (generated) |
| `build_notebook.py` | the source of every cell; writes the notebook and embeds the recording |
| `recordings/memory_tutorial.jsonl` | the recorded model replies used by offline mode |
| `tools/run_fresh.py` | runs the notebook top to bottom in a fresh workspace, offline or live |
| `guide/speaker_guide.md` | notes for the presenter |

## Rebuilding

```bash
pip install nbformat nbclient ipykernel
python build_notebook.py                         # writes the notebook
T2_RECORD=new.jsonl ...                          # a live run with this variable set records its replies
```

A new recording must come from one live run of the current cells: offline mode
matches each prompt in full, memory section included, so any change to a cell
that reaches the model needs a new recording.
