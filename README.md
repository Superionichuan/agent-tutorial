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
