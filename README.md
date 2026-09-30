# AgentForge Tutorial 1: The Anatomy of an Agent

Hands-on Workshop on AI Agents as Scientific Research Assistants
Nanyang Technological University · Supported by Schmidt Sciences

## Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Superionichuan/agent-tutorial/blob/main/notebooks/agent_tutorial_colab.ipynb)

No installation required. The first cell installs the package and asks for
your own API key (DeepSeek, Qwen, or Claude). The key stays in your Colab
runtime and is sent only to the provider you select. Without a key, choose
`offline`: every cell then runs from a recorded run of the notebook.

## Run locally

```bash
git clone https://github.com/Superionichuan/agent-tutorial.git
cd agent-tutorial
pip install -e .
cp .env.example .env      # then edit .env with your key, or set LLM_BACKEND=offline
jupyter lab notebooks/agent_tutorial.ipynb
```

## Contents

The tutorial dissects `mini_agent`, a complete working agent built from
minimal modules, one concern per module. Sections cover the language model,
tools, the ReAct loop, the event log, task state, memory, skills, prompt
profiles, and the harness. Optional appendices cover failure modes and
evaluation, safety, and offline mode.
