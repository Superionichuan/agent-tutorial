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

Tutorial 2: Agentic memory, in [tutorial2/](tutorial2/):
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Superionichuan/agent-tutorial/blob/main/tutorial2/notebooks/agent_memory_colab.ipynb)

## Install on your computer

One command installs `mini_agent` in your user account (no administrator rights, no Python
needed beforehand). It sets up [uv](https://docs.astral.sh/uv/), which uses a Python 3.12 it
finds on your computer or downloads one, puts the `mini_agent` command on your PATH, and
then asks for a provider and a key (`mini_agent setup`).

macOS or Linux, in Terminal:

```bash
curl -LsSf https://raw.githubusercontent.com/Superionichuan/agent-tutorial/main/install.sh | sh
```

Windows, in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/Superionichuan/agent-tutorial/main/install.ps1 | iex"
```

Then open a new terminal and run `mini_agent`. `mini_agent doctor` shows what is set up.
Running the command again upgrades and keeps your settings. To uninstall:
`uv tool uninstall mini_agent`; what it leaves behind is listed under
[What mini_agent writes](#what-mini_agent-writes).

## Using mini_agent

### Getting a key

mini_agent needs a language model. The simplest is [OpenRouter](https://openrouter.ai): a
service that sells many companies' models under one key, pay as you go. Sign up, add credit
(Settings, Credits), and create a key at
[openrouter.ai/settings/keys](https://openrouter.ai/settings/keys); `mini_agent setup` asks
for it. With the default model (DeepSeek V4 Flash), four everyday tasks of two or three
steps (1,400 to 5,700 tokens) cost between US$0.0001 and US$0.0005 each when we measured
them in November 2026, so a dollar of credit lasts for a few thousand such tasks; longer
tasks cost more.

Without a key: a local model with [Ollama](https://ollama.com) (free, on your own computer;
`mini_agent setup --provider ollama`), or the offline demo, which replays only the
tutorial's recorded tasks (`mini_agent --backend offline "Compute 3**7."`).

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

In the interactive modes, `/plan TASK` asks the agent to explore and write a step-by-step
plan for the task, without changing anything, and to wait for your go-ahead; `/plan off`
ends plan mode. It is an instruction to the model, not a lock: risky calls still ask first,
unless you allowed that tool with `a` earlier in the session.

### Safety

The agent asks before each risky call: a shell command (`bash`), Python code
(`run_python`) or a file write (`write_file`). Its other tools (reading files, listing a
folder, the calculator, notes) run without asking.

- At the prompt, `y` allows this one call, `n` refuses it (the agent is told; it may ask
  again or try another way), and `a` allows that tool for the rest of the session without
  asking again.
- `--yes` (same as `--approve auto`) runs risky calls without asking; `--approve deny`
  never runs them.
- With no terminal to ask (a script, a scheduled job), the first risky call stops the task
  with exit status 3; `mini_agent -c --yes` continues it with risky calls allowed.
- Tool commands run without a terminal, so a command that asks for a password or a
  confirmation (sudo, ssh, git) fails at once instead of waiting; use keys, an SSH agent or
  a credential helper.
- To leave a process running after the tool returns, redirect its output, e.g.
  `server > server.log 2>&1 &`. Otherwise the tool waits until its 120 s limit. On macOS
  and Linux it then ends the process (unless the process started a session of its own, as
  `setsid` and most daemons do); on Windows a process started in the background keeps
  running, so end it yourself (in Task Manager, or with `taskkill /F /IM name.exe`, which
  ends every process with that name).
- The file tools only reach paths inside the workspace; `--no-confine` lifts that limit.
  The limit is a convenience, not isolation: shell commands and Python are not confined.
  For real isolation, run mini_agent inside a virtual machine or container of your choice.

### Long tasks

mini_agent keeps a folder `.mini_agent/` in the workspace (git ignores it):

- `user.md`: every message you sent, word for word;
- `handoff.md`: where an unfinished task stands (the goal, the steps done, the next step,
  the files written and read), written when old steps are shortened and when a task stops
  before it is done; `mini_agent -c` picks it up.

Notes the agent saves with its `remember` tool go to `~/.mini_agent/memory/`, with the
evidence it gives for them. Before every model call the agent rereads the messages of the
current session (one task, or one interactive session), its open handoff and the saved
notes, so a long task keeps its instructions even after old steps are shortened. Messages
of earlier sessions stay in `user.md` but are not shown.

When the next prompt would fill more than half of the model's context window, old steps
are shortened: long results are cut, old steps are folded to one line, and only if that is
not enough does the model summarise them. A one-line notice says when this happens.
`--no-lifecycle` runs the plain loop instead: no `.mini_agent` files and no shortening, so
a task stops when the context window is full.

A task stops after `--budget` tokens (all its model calls together; default 1,000,000, or
`MINI_AGENT_BUDGET`). `mini_agent -c` continues work in the folder you run it in. In a
terminal it reopens that folder's latest session; in a script it continues that folder's
latest unfinished task and then exits with that task's status. A task that stopped, crashed
or was cut off with Ctrl+C goes on from its last step; a tool call that was cut off is not
run again. In a folder with nothing to continue, `-c` runs nothing and names the folder of
the latest unfinished task.

Exit status of `mini_agent "task"`: 0 the agent finished (its answer says whether it
managed); 1 stopped before the end (token budget, step limit, a model error, a task the
offline demo has no recording of, ...); 3 a risky call needs approval and there was no
terminal to ask; 2 a usage or set-up error, or nothing for `-c` to continue here; 4 the
session is in use by another mini_agent process; 130 interrupted. After 1, 3 or 130,
`mini_agent -c` in the same folder continues the task.

### Keep a long task going

```bash
mini_agent --keep-alive --yes "the task"     # or, for an unfinished task here: mini_agent -c --keep-alive --yes
```

With `--keep-alive`, a passing failure does not end the task: when the provider is down
or cannot be reached (a network error, a timeout, HTTP 408, 429 or 5xx), or the process
is killed (out of memory, `kill -9`), it waits and continues the same task with `-c`. The
waits grow: 30 s, then 1, 2, 4 and 8 minutes, then every 10 minutes. Each continue prints
one line and is written to the session log. A stop that waiting will not change ends it
at once, with that stop's exit status: the task is done, a risky call needs approval, the
token budget or the step limit is reached, a wrong key, no credit or an unknown model
(HTTP 4xx), or you press Ctrl+C.

All the continues share the task's `--budget`. `--keep-alive N` gives up after N
continues in a row that complete no step (default 5) and exits with status 75 ("try
again later"); a continue that completes a step starts the count again.
`mini_agent -c --keep-alive` in a folder with nothing unfinished prints one line and exits
0, so it is harmless to run at every login.

Ctrl+C, a service stop (SIGTERM) or closing the terminal also stops the tool the agent is
running. A process killed outright (`kill -9`, out of memory, `taskkill /F`) cannot do that:
a command its tool had started may still finish. When the task continues, that call is not
run again; the model is told it was cut off.

`-c` does not remember the options of the earlier run: repeat `--yes`, `--budget` and the
rest. Without a terminal nobody can approve a risky call, so without `--yes` the first one
stops the task with exit status 3; add `--yes` (or `--approve deny`) if the task may run
shell commands, Python or file writes unattended.

To start `mini_agent -c --keep-alive` again after the computer restarts, use the system's
own tools. Use the full path of `mini_agent` (`which mini_agent` on macOS and Linux,
`where mini_agent` on Windows): these tools do not read your shell's settings. The key is
not needed there; the settings saved by `mini_agent setup` are used.

macOS (tested on macOS 15): save as `~/Library/LaunchAgents/org.miniagent.continue.plist`,
with your paths. It runs at every login; `launchctl bootstrap gui/$(id -u)
~/Library/LaunchAgents/org.miniagent.continue.plist` also starts it now, and `plutil -lint`
checks the file. To remove it: `launchctl bootout gui/$(id -u)/org.miniagent.continue`,
then delete the file.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>org.miniagent.continue</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/YOU/.local/bin/mini_agent</string>
    <string>-c</string>
    <string>--keep-alive</string>
  </array>
  <key>WorkingDirectory</key><string>/Users/YOU/my-project</string>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>/Users/YOU/my-project/.mini_agent/launchd.log</string>
  <key>StandardErrorPath</key><string>/Users/YOU/my-project/.mini_agent/launchd.log</string>
</dict>
</plist>
```

Linux with systemd (tested on Ubuntu 24.04): save as
`~/.config/systemd/user/mini_agent-continue.service`, with your paths, then run
`systemctl --user daemon-reload` and `systemctl --user enable --now mini_agent-continue`.
It starts now and at every login; `journalctl --user -u mini_agent-continue` shows its
output. To remove it: `systemctl --user disable --now mini_agent-continue`, delete the file
and run `systemctl --user daemon-reload` again.

```ini
[Unit]
Description=Continue the unfinished mini_agent task in my-project

[Service]
WorkingDirectory=/home/YOU/my-project
ExecStart=/home/YOU/.local/bin/mini_agent -c --keep-alive

[Install]
WantedBy=default.target
```

Windows (tested on Windows 11): save as `mini_agent-continue.cmd` in your Startup folder
(press Win+R and type `shell:startup`), with your paths. Windows runs it at every login, in
a console window that stays open while the task runs; the output goes to
`mini_agent-continue.log` in the project folder, in UTF-8. To stop the task, press Ctrl+C
in that window or close it; in Task Manager use "End process tree", not "End task" on one
process. To stop running it at login, delete the file.

```bat
@echo off
cd /d "C:\Users\YOU\my-project"
"C:\Users\YOU\.local\bin\mini_agent.exe" -c --keep-alive >> mini_agent-continue.log 2>&1
```


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

The one-line install keeps mini_agent in its own environment, so run your script with that
environment's Python (the one `mini_agent doctor` shows on its first lines):

```bash
"$(uv tool dir)/mini-agent/bin/python" my_tool.py             # macOS, Linux
& "$(uv tool dir)\mini-agent\Scripts\python.exe" my_tool.py   # Windows PowerShell
```

In any Python 3.10 or newer, `pip install "git+https://github.com/Superionichuan/agent-tutorial"`
installs it too (it needs git).

### Models

`mini_agent setup` sets the model for the command line and for Python. Other choices:

- an environment variable instead of setup: `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`,
  `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`, and optionally `LLM_MODEL`;
- any OpenAI-compatible server (vLLM, LM Studio, a company gateway):
  `mini_agent setup --provider compat --base-url http://localhost:8000/v1 --model NAME`
  (press Enter at the key prompt if the server needs no key);
- a local Ollama server: `mini_agent setup --provider ollama` (model `qwen3:8b` unless you
  add `--model NAME`; pull the model in Ollama first). With no settings at all, mini_agent
  uses a local Ollama server if one is running;
- offline: `mini_agent --backend offline "Compute 3**7."` replays a recorded run: no key,
  no network, and only the tutorial's own tasks.

### What mini_agent writes

- `~/.mini_agent/config.env`: your settings from `mini_agent setup`, including the key
  (readable only by you on macOS and Linux);
- `~/.mini_agent/memory/`: notes the agent saves with `remember`, and a short daily log;
- `~/.mini_agent/skills/`: skills you add;
- `~/.mini_agent/sessions/`: one log per task or interactive session, with every message,
  step, tool result and answer; `mini_agent -c` reads them. While a session runs, a
  `.lock` file next to its log keeps a second mini_agent off it;
- `.mini_agent/` in each folder the agent worked in: `user.md` and `handoff.md`.

On Windows, `~` is `C:\Users\<name>`. `uv tool uninstall mini_agent` removes the program
only: delete `~/.mini_agent` and the `.mini_agent` folders to remove the rest. uv itself,
and a Python it downloaded, stay; uv's documentation says how to remove them.

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

MIT for the mini_agent package and Tutorial 1; see [LICENSE](LICENSE). The Tutorial 2
material in [tutorial2/](tutorial2/) follows the workshop's lecture on agent memory and is
shared for the workshop; it is not covered by the MIT license.
