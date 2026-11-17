# AgentForge Tutorial 1: From Chatbot to Agent

*The anatomy of an agent*

Hands-on Workshop on AI Agents as Scientific Research Assistants
Nanyang Technological University · Supported by Schmidt Sciences

## Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Superionichuan/agent-tutorial/blob/main/notebooks/agent_tutorial_colab.ipynb)

No installation required. The first cell installs the package and asks for
your own API key (DeepSeek, Qwen, or Claude). The key stays in your Colab
runtime and is sent only to the provider you select. Without a key, choose
`offline`: every cell then runs from a recorded run of the notebook.

## Install on your computer

One command installs `mini_agent` in your user account (no administrator rights, no Python
needed beforehand). It sets up [uv](https://docs.astral.sh/uv/), which brings its own
Python 3.12, puts the `mini_agent` command on your PATH, and then asks for a provider and
a key (`mini_agent setup`).

macOS or Linux, in Terminal:

```bash
curl -LsSf https://raw.githubusercontent.com/Superionichuan/agent-tutorial/main/install.sh | sh
```

Windows, in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/Superionichuan/agent-tutorial/main/install.ps1 | iex"
```

Then open a new terminal and run `mini_agent`. `mini_agent doctor` shows what is set up.
Running the command again upgrades. To uninstall: `uv tool uninstall mini_agent` (your
settings stay in `~/.mini_agent`; delete that folder to remove them too).

## Using mini_agent

### First run

```bash
mini_agent setup     # choose a provider, paste a key, one test call
mini_agent doctor    # what is installed, which model is set up, what is missing
```

`setup` offers OpenRouter (one key, many models), DeepSeek, Anthropic Claude, OpenAI,
Google Gemini, Alibaba Qwen, any other OpenAI-compatible server, a local Ollama server,
and the offline demo. It saves your choice in `~/.mini_agent/config.env` (on Windows
`C:\Users\<name>\.mini_agent\config.env`), not in the folder you work in.

### Running

```bash
mini_agent "Count the lines in every .py file in this folder"   # one task, then exit
mini_agent                                                       # interactive; /exit to quit
mini_agent --tui                                                 # full screen; Ctrl+Q to quit
```

The agent works in the current folder; `--workspace DIR` chooses another. With OpenRouter,
`--model` picks another of its models for one run, e.g.
`mini_agent --model qwen/qwen3.8-flash "What is 947 * 863?"`.

### Safety

The agent asks before each risky call: a shell command (`bash`), Python code
(`run_python`) or a file write (`write_file`). Its other tools (reading files, listing a
folder, the calculator, notes) run without asking.

- `--yes` (same as `--approve auto`) runs risky calls without asking; `--approve deny`
  never runs them.
- With no terminal to ask (a script, a scheduled job), the first risky call stops the task
  with exit status 3; `mini_agent -c --yes` continues it with risky calls allowed.
- The file tools only reach paths inside the workspace; `--no-confine` lifts that limit.
  The limit is a convenience, not isolation: shell commands and Python are not confined.
  For real isolation, run mini_agent inside a virtual machine or container of your choice.

### Long tasks

mini_agent keeps a folder `.mini_agent/` in the workspace (git ignores it):

- `user.md`: every message you sent, word for word;
- `handoff.md`: where a stopped task stands: the goal, the steps done, the next step, the
  files written and read.

Notes the agent saves with its `remember` tool go to `~/.mini_agent/memory/`, with the
evidence it gives for them. All of these are read before every model call, so a long task
keeps its instructions. When the next prompt would fill more than half of the model's
context window, old steps are shortened: long results are cut, old steps are folded to one
line, and only if that is not enough does the model summarise them. A one-line notice
says when this happens. `--no-lifecycle` runs the plain loop instead: no `.mini_agent`
files and no shortening, so a task stops when the context window is full.

A task stops after `--budget` tokens (all its model calls together; default 1,000,000, or
`MINI_AGENT_BUDGET`). `mini_agent -c` continues the latest unfinished task after a stop, a
crash or Ctrl+C; a tool call that was cut off is not run again.

Exit status of `mini_agent "task"`: 0 done; 1 stopped before the end (token budget, step
limit, a model error, ...); 3 a risky call needs approval and there was no terminal to ask;
2 a usage or set-up error; 130 interrupted. After 1, 3 or 130, `mini_agent -c` continues
the task.

### Your own tool

```python
from mini_agent import Agent, default_tools, tool

@tool
def word_count(text: str) -> str:
    """Count the words in a text."""
    return str(len(text.split()))

agent = Agent(tools=default_tools() + [word_count])
print(agent.run("How many words are in 'the quick brown fox jumps over the lazy dog'?"))
```

The function's name, docstring and typed parameters tell the model what the tool does.
`@tool(risky=True)` marks a tool that needs approval, like `bash`. `Agent` uses the model
set up with `mini_agent setup`; in Python it runs every tool without asking unless you
pass `approve="ask"`.

### Models

`mini_agent setup` sets the model for the command line and for Python. Other choices:

- an environment variable instead of setup: `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`,
  `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`, and optionally `LLM_MODEL`;
- any OpenAI-compatible server (vLLM, LM Studio, a company gateway):
  `mini_agent setup --provider compat --base-url http://localhost:8000/v1`;
- a local Ollama server: `mini_agent setup --provider ollama` (model `qwen3:8b` unless you
  add `--model NAME`; pull the model in Ollama first). With no settings at all, mini_agent
  uses a local Ollama server if one is running;
- offline: `mini_agent --backend offline "Compute 3**7."` replays a recorded run: no key,
  no network, and only the tutorial's own tasks.

## Run locally

```bash
git clone https://github.com/Superionichuan/agent-tutorial.git
cd agent-tutorial
pip install -e . pandas jupyterlab   # pandas for the tables, JupyterLab to open the notebook
cp .env.example .env      # then edit .env with your key, or set LLM_BACKEND=offline
jupyter lab notebooks/agent_tutorial.ipynb
```

## Contents

The tutorial dissects `mini_agent`, a complete working agent built from
minimal modules, one concern per module. Sections cover the language model,
tools, the ReAct loop, the event log, task state, memory, skills, prompt
profiles, and the harness. Optional appendices cover failure modes and
evaluation, safety, and offline mode.

## License

MIT; see [LICENSE](LICENSE).
