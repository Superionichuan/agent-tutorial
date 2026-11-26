# AgentForge Tutorial 1: From Chatbot to Agent

*The anatomy of an agent*

Hands-on Workshop on AI Agents as Scientific Research Assistants
Nanyang Technological University · Supported by Schmidt Sciences

The Tutorial 1 notebook here uses mini_agent 2.0.0, which now lives in its own repository,
[Superionichuan/mini_agent](https://github.com/Superionichuan/mini_agent) (see
[Versions](#versions) below).

## Two notebooks for Tutorial 1

This notebook takes a working agent apart: what the model sees, how a reply becomes a tool
call, the event log, task state, memory, skills, sessions, approval, and evaluation, using the
small `mini_agent` package. It assumes no earlier agent code: section 1 begins with a single
model call.

The workshop's other Tutorial 1 notebook,
[What is an agent?](https://colab.research.google.com/github/AgentForgeWorkshop/Tutorials/blob/main/Tutorial1_frameworks/01a_what_is_an_agent.ipynb)
([folder](https://github.com/AgentForgeWorkshop/Tutorials/tree/main/Tutorial1_frameworks)),
builds the loop itself step by step (a model call, tools as functions, the loop, guardrails)
and then the same agent in LangChain and LangGraph. The two notebooks cover different ground
and can be done in either order. If you would like to write a loop yourself before taking a
complete one apart, start with that notebook.

## Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Superionichuan/agent-tutorial/blob/main/notebooks/agent_tutorial_colab.ipynb)

No installation required. The setup cell installs mini_agent 2.0.0. With the default
`offline`, every cell runs from a recorded run of the notebook, without a key. To use a live
model, choose a provider in the setup cell and paste your own API key (DeepSeek, Qwen, or
Claude), or paste an [OpenRouter](https://openrouter.ai) key into the cell after it, which
also lets you choose the model. The *Try it* exercises need a live model. A key stays in your
Colab runtime and is sent only to the provider you choose.

What you should see: the setup cell ends with `Setup complete. Continue with the next cell.`,
and the first full run, a few cells further down, ends with the PV energy, about 5.82 eV.

## Run locally

You need Python 3.10 or newer with `pip` and `venv`. On Ubuntu they come with
`sudo apt install python3-pip python3-venv`. The local notebook has been checked on Linux; on
Windows, Colab is the checked route. To run it on Windows anyway, write `py -3 -m venv .venv`
for the third line, and activate with `.venv\Scripts\Activate.ps1` (PowerShell) or
`.venv\Scripts\activate.bat` (Command Prompt):

```bash
git clone https://github.com/Superionichuan/agent-tutorial.git
cd agent-tutorial
python3 -m venv .venv
. .venv/bin/activate
python -m pip install "https://github.com/Superionichuan/mini_agent/releases/download/v2.0.0/mini_agent-2.0.0-py3-none-any.whl" pandas jupyterlab
```

Without administrator rights, or without `pip` and `venv`, use
[uv](https://docs.astral.sh/uv/getting-started/installation/) instead; it downloads Python
if needed:

```bash
git clone https://github.com/Superionichuan/agent-tutorial.git
cd agent-tutorial
uv venv --python 3.12 .venv
. .venv/bin/activate              # Windows: .venv\Scripts\Activate.ps1 (PowerShell)
uv pip install "https://github.com/Superionichuan/mini_agent/releases/download/v2.0.0/mini_agent-2.0.0-py3-none-any.whl" pandas jupyterlab
```

Then choose a model. Run one of these lines on its own, not pasted together with other
lines, because `mini_agent setup` reads your answers from the terminal:

```bash
mini_agent setup --provider offline   # no key: the tutorial's recorded runs
mini_agent setup                      # a provider and your own key
```

Check the install (it prints 2187), then open the notebook:

```bash
mini_agent --backend offline "Compute 3**7."
jupyter lab notebooks/agent_tutorial.ipynb
```

In the notebook, choose the Python 3 kernel if JupyterLab asks. What you should see: the
first full run ends with the PV energy, about 5.82 eV.

`mini_agent setup` saves your choice in `~/.mini_agent/config.env`; run it again to change it.
Start the command line as `mini_agent`, not as `python -m mini_agent` from this folder: the
`mini_agent/` folder here holds the 1.2.2 source, and Python would run that instead.

## Contents

The tutorial dissects `mini_agent`, a complete working agent built from
minimal modules, one concern per module. Sections cover the language model,
tools, the ReAct loop, the event log, task state, memory, skills, prompt
profiles, and the harness. The last section gives the agent a research task,
fitting an equation of state to a data file, and checks the bulk modulus it
reports with a rule in code. Optional appendices cover failure modes and
evaluation, safety, and offline mode.

## Versions

mini_agent now lives in its own repository,
[Superionichuan/mini_agent](https://github.com/Superionichuan/mini_agent) (version 2.0.0,
with a one-line install for macOS, Linux and Windows). This repository keeps the AgentForge
tutorials and the development history. The Tutorial 1 notebook installs mini_agent 2.0.0.
The Tutorial 2 notebook (`tutorial2/`) installs 1.2.2, the version used in the workshop; the
1.2.2 command line has no `--version` or `--help`: anything typed after `mini_agent` is run
as a task. The `mini_agent/` folder here is the 1.2.2 source; Tutorial 1 quotes the 2.0.0
source in the mini_agent repository.

## License

MIT, see [LICENSE](LICENSE).
