"""Build notebooks/agent_memory_colab.ipynb from the cell sources below.

The recorded model replies for offline mode (recordings/memory_tutorial.jsonl)
are embedded in the setup cell, so the notebook is one self-contained file.

    python build_notebook.py            # writes the notebook
"""
import base64
import gzip
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).parent
RECORDING = ROOT / "recordings" / "memory_tutorial.jsonl"
OUT = ROOT / "notebooks" / "agent_memory_colab.ipynb"
DEFAULT_PROVIDER = "offline"
PROVIDERS = ["offline", "deepseek", "qwen", "claude"]


def md(text):
    return nbformat.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbformat.v4.new_code_cell(text.strip("\n"))


# --------------------------------------------------------------------------
# Title and setup
# --------------------------------------------------------------------------

TITLE = md(r"""
<div align="center">
<img src="__LOGO__" width="420" alt="AgentForge"/>

<h2>Tutorial 2: Agentic Memory</h2>
<b>Hands-on Workshop on AI Agents as Scientific Research Assistants</b><br/>
<i>Nanyang Technological University · Supported by Schmidt Sciences</i>
</div>

---

In Tutorial 1 you took `mini_agent` apart: a language model, plus tools, plus a loop.
One line from it is where this notebook starts: **memory is just text that code decides
to show the model.** The model keeps nothing between calls, so anything it should know
has to be put back into its prompt by the agent's code.

This notebook follows the four parts of the memory lecture, with one small example at a
time, all on the same agent.

| Part | Lecture | Here you will | Minutes |
|---|---|---|---|
| 0 | | set up, and look at one prompt | 4 |
| 1 | Why agents need memory | see the model forget an earlier turn, and the session log bring it back | 7 |
| 2 | Scope, content, writing and reading | keep notes for a later session, update one, check one against its source | 14 |
| 3 | Routes into the context | compare full replay, a rolling summary and retrieval | 12 |
| 4 | Cost and evaluation | compare what each route costs, and size the KV cache | 6 |
| 5 | Summary | | 3 |

**How to work.** Run the cells in order with Shift-Enter. Before you run a cell, predict
what it will do. After it runs, read the trace, and look at the prompt: that is all the
model saw. No programming is needed: each *Try it* cell is ready to run, and you can
change one line in it to try your own variant.
""")

COLAB_NOTE = md(r"""
> **Running in Google Colab.** Run the setup cell below first. With `offline`, every cell runs
> from a recorded run, without a key. A key is sent only to the provider you choose; it is also
> saved in a `.env` file on this Colab machine, so that `python -m mini_agent` in a terminal can
> use it. Run the parts in order: later parts use what earlier ones define.
""")

SETUP = r'''
# @title ⚙ Setup: install mini_agent and choose a provider  { display-mode: "form" }
# Run this cell first. It installs mini_agent (the agent from Tutorial 1) and
# connects it to a model.
#
# PROVIDER = "offline" needs no API key: every model call is answered from a
# recorded run of this notebook, while the tools run for real. The other
# providers need your own key, read from Colab Secrets (left sidebar, key icon)
# if it is stored there, and otherwise asked for below. The key is sent only to
# the provider you select. It is also saved in a .env file on this machine, so
# that `python -m mini_agent` in a terminal finds it (step 5 below).

PROVIDER = "__DEFAULT_PROVIDER__"  # @param __PROVIDERS__

import base64, gzip, hashlib, json, os, re, subprocess, sys
from pathlib import Path

# ---- 1. install mini_agent 1.2.2 from a fixed commit of the Tutorial 1 repository ----
from importlib.metadata import PackageNotFoundError, version

VERSION = "1.2.2"
COMMIT = "72c413274025af372c718aa916263eeae318207d"
WHEEL = ("https://github.com/Superionichuan/agent-tutorial/raw/"
         f"{COMMIT}/dist/mini_agent-{VERSION}-py3-none-any.whl")
try:
    _installed = version("mini_agent")
except PackageNotFoundError:
    _installed = ""
if _installed != VERSION:            # rerunning this cell to switch provider skips this
    if "mini_agent" in sys.modules:
        raise SystemExit("Another mini_agent version is loaded in this session. "
                         "Use Runtime > Restart session, then run this cell again.")
    print(f"Installing mini_agent {VERSION} ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
                    "--force-reinstall", "--no-deps", WHEEL], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "httpx", "rich"],
                   check=True)
import importlib.util
if importlib.util.find_spec("textual") is None:   # interactive mode in a terminal needs it
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "textual"], check=True)

import mini_agent

_pkg = Path(mini_agent.__file__).parent
_base = f"https://raw.githubusercontent.com/Superionichuan/agent-tutorial/{COMMIT}/mini_agent/"
_files = ["prompts/agent.md", "prompts/coding.md", "llm/recordings/tutorial.jsonl"]
_missing = [f for f in _files if not (_pkg / f).exists()]
if _missing:                         # repair an incomplete download
    import urllib.request
    for _f in _missing:
        (_pkg / _f).parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_base + _f, _pkg / _f)
    print(f"Repaired {len(_missing)} data file(s).")

# A new Python process (a terminal) sends DeepSeek-backend calls to OpenRouter
# only while this file exists; it is written for a workshop key.
_PTH = _pkg.parent / "mini_agent_base_url.pth"

from mini_agent.llm.base import BaseLLM


# ---- 2. every model call goes through one small wrapper ----
def _workspace(prompt):
    m = re.search(r"^Workspace: (.*)$", prompt, re.M)
    ws = m.group(1).strip() if m else ""
    return ws if len(ws) > 1 else ""         # never replace a bare "/"


def _normalize(prompt):
    """Mask what differs between machines and days: the workspace path, dates, times."""
    text = prompt
    if _workspace(text):
        text = text.replace(_workspace(text), "<ws>")
    text = text.replace(str(Path.home()), "~")
    text = re.sub(r"\d{4}-\d{2}-\d{2}", "<date>", text)
    text = re.sub(r"\b\d{1,2}:\d{2}(:\d{2})?\b", "<time>", text)
    return re.sub(r"\s+", " ", text).strip()


class Model(BaseLLM):
    """The model this notebook talks to.

    llm.prompts keeps every prompt that was sent, so any cell can show what the
    model saw. Live, each call goes to the provider. Offline, the reply comes
    from the recording, matched on the whole prompt, memory section included;
    a prompt that is not in the recording gets a notice, never another reply.
    """

    def __init__(self, live=None, recording=None, record_to=None):
        self.live, self.record_to = live, record_to
        self.replies = {}
        for rec in recording or []:
            self.replies.setdefault(rec["key"], []).append(rec)
        if live is not None:
            self.model = live.model
        else:
            self.model = f"offline (recorded with {recording[0]['model']})"
        self.context_limit = getattr(live, "context_limit", 128_000)
        self.prompts = []
        self._seen = {}
        self.total_prompt_tokens = self.total_completion_tokens = 0

    def call(self, prompt):
        self.prompts.append(prompt)
        key = hashlib.sha256(_normalize(prompt).encode()).hexdigest()[:24]
        n = self._seen.get(key, 0)           # the same prompt again replays the next reply
        self._seen[key] = n + 1
        if self.live is None:
            options = self.replies.get(key)
            if not options:
                return _not_recorded(prompt)
            rec = options[min(n, len(options) - 1)]
            self.total_prompt_tokens += rec["prompt_tokens"]
            self.total_completion_tokens += rec["completion_tokens"]
            return rec["response"].replace("<ws>", _workspace(prompt) or ".")
        p0 = self.live.total_prompt_tokens
        c0 = self.live.total_completion_tokens
        reply = self.live.call(prompt)
        dp = self.live.total_prompt_tokens - p0
        dc = self.live.total_completion_tokens - c0
        self.total_prompt_tokens += dp
        self.total_completion_tokens += dc
        if self.record_to:                   # paths in a reply become portable
            saved = reply.replace(_workspace(prompt), "<ws>") if _workspace(prompt) else reply
            saved = saved.replace(str(Path.home()), "~")
            with open(self.record_to, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "n": n, "response": saved,
                                    "prompt_tokens": dp, "completion_tokens": dc,
                                    "model": self.live.model}, ensure_ascii=False) + "\n")
        return reply

    def reset_usage(self):
        self.total_prompt_tokens = self.total_completion_tokens = 0
        if self.live is not None:
            self.live.reset_usage()


def _not_recorded(prompt):
    print("[offline] This prompt is not in the recording: something you changed made it new. "
          "To run your own variant, choose a provider in the setup cell.")
    if "[Goal]" in prompt:                   # an agent step: end the run with a note
        return json.dumps({"thought": "This step is not in the recording.",
                           "action": "done",
                           "args": {"answer": "[offline] not in the recording"}})
    return ""


# ---- 3. a helper that prints a prompt section by section ----
SECTIONS = ["Goal", "Tools", "Memory", "Steps taken so far", "Instructions"]


def show_prompt(prompt, only=None):
    """Print a prompt section by section, with the size of each section."""
    pattern = r"^\[(" + "|".join(map(re.escape, SECTIONS)) + r")\]\n"
    parts = re.split(pattern, prompt, flags=re.M)
    for name, body in [("intro", parts[0])] + list(zip(parts[1::2], parts[2::2])):
        if only and name not in only:
            continue
        body = body.strip()
        print(f"── [{name}] · {len(body):,} characters ".ljust(64, "─"))
        print(body or "(empty)")
        print()


# ---- 4. connect to the model ----
def _connect():
    global _env
    if PROVIDER == "offline":
        _PTH.unlink(missing_ok=True)
        os.environ["LLM_BACKEND"] = "offline"
        _env = "LLM_BACKEND=offline\n"
        lines = gzip.decompress(base64.b64decode(RECORDING)).decode().splitlines()
        recording = [json.loads(line) for line in lines if line.strip()]
        model = Model(recording=recording)
        print(f"Offline mode: {model.model}, {len(recording)} model replies.")
        print("No API key is used. Variants of your own need a live provider.")
        return model

    secret = {"deepseek": "DEEPSEEK_API_KEY", "qwen": "QWEN_API_KEY",
              "claude": "CLAUDE_API_KEY"}[PROVIDER]
    key = ""
    try:                                     # Colab Secrets, if available
        from google.colab import userdata
        key = userdata.get(secret) or ""
        if key:
            print(f"Key loaded from Colab Secrets ({secret}).")
    except Exception:
        pass
    if not key:                              # otherwise, a one-time prompt
        import getpass
        bar = "=" * 68
        print("\n" + bar)
        print(f"  ACTION REQUIRED: paste your {PROVIDER} API key below, then press Enter")
        print(bar)
        print("  The input box appears under this cell. Typing is hidden.")
        print("  This cell waits until you paste the key; it is not stuck.")
        print("  No key? Rerun this cell with PROVIDER = offline.")
        print("  A workshop key (it starts with sk-or-) works here too, with any PROVIDER.")
        print(f"  To skip this prompt next time, store the key in Colab Secrets as {secret}.")
        print(bar)
        key = getpass.getpass(f"{PROVIDER} API key: ").strip()
        if not key:
            raise ValueError(f"No key entered. Rerun this cell and paste your {PROVIDER} "
                             f"key, store it in Colab Secrets as {secret}, or choose offline.")
        print("Key received.")

    workshop = key.startswith("sk-or-")
    from mini_agent.llm.backends import DeepSeekLLM
    if workshop:
        # A workshop key (OpenRouter) pasted here works with any PROVIDER choice.
        # OpenRouter speaks the same protocol as DeepSeek's API, so the DeepSeek
        # backend is reused with OpenRouter's address and the same model.
        OPENROUTER_URL = "https://openrouter.ai/api/v1"
        DeepSeekLLM.BASE_URL = OPENROUTER_URL
        os.environ.update(LLM_BACKEND="deepseek", DEEPSEEK_API_KEY=key,
                          DEEPSEEK_MODEL="deepseek/deepseek-v4-flash",
                          MINI_AGENT_DEEPSEEK_URL=OPENROUTER_URL)
        _PTH.write_text(
            'import os; setattr(__import__("mini_agent.llm.backends.deepseek", '
            'fromlist=["DeepSeekLLM"]).DeepSeekLLM, "BASE_URL", '
            'os.environ.get("MINI_AGENT_DEEPSEEK_URL", "https://openrouter.ai/api/v1"))\n')
        _env = ("LLM_BACKEND=deepseek\nDEEPSEEK_MODEL=deepseek/deepseek-v4-flash\n"
                f"DEEPSEEK_API_KEY={key}\n")
        live = DeepSeekLLM(api_key=key, model="deepseek/deepseek-v4-flash")
    else:
        _PTH.unlink(missing_ok=True)                      # undo an earlier workshop key
        os.environ.pop("MINI_AGENT_DEEPSEEK_URL", None)
        os.environ.pop("DEEPSEEK_MODEL", None)
        if PROVIDER == "deepseek":
            DeepSeekLLM.BASE_URL = "https://api.deepseek.com/v1"
        os.environ["LLM_BACKEND"] = PROVIDER
        os.environ[secret] = key
        _env = f"LLM_BACKEND={PROVIDER}\n{secret}={key}\n"
        from mini_agent.llm import create_llm
        config = {"LLM_BACKEND": PROVIDER, secret: key}
        config.update({k: v for k, v in os.environ.items() if k.endswith("_MODEL")})
        live = create_llm(config)
    try:
        reply = live("Reply with the single word: ready").strip()
    except Exception as e:
        reply = f"Error: {type(e).__name__}: {e}"
    if reply.startswith("Error"):
        raise RuntimeError(f"Connection failed: {reply[:200]}\n"
                           "Most likely an invalid key, or no credit on the account. "
                           "Check the key, or continue offline (rerun this cell with offline).")
    via = " via OpenRouter (workshop key)" if workshop else ""
    print(f"\nConnected: {live.model}{via}. Test call returned: {reply[:40]}")
    model = Model(live=live, record_to=os.environ.get("T2_RECORD"))
    model.workshop_key = workshop
    return model


# Recorded model replies for offline mode (gzip, then base64, of JSON lines).
RECORDING = "__RECORDING__"

llm = _connect()

# ---- 5. make the configuration visible to the command line ----
# A terminal starts a new process that does not see this notebook's settings, so
# they are also written to a .env file where `python -m mini_agent` looks for it:
# the current folder, or else next to the package.
for _d in {Path("/content"), Path.cwd(), _pkg.parent}:
    if _d.is_dir():
        (_d / ".env").write_text(_env)

print("Setup complete. Continue with the next cell.")
'''

WORKSHOP_KEY_NOTE = md(r"""
> **Workshop key.** Run the next cell and paste your workshop key when asked. Every model call
> then goes to a live model, so your own variants in the *Try it* cells work too. No workshop
> key? Press Enter at the prompt and the notebook keeps running as it is (offline by default).
> A workshop key pasted into the setup cell's prompt works too.
""")

OPENROUTER = r'''
# @title 🔑 Workshop key (OpenRouter): run this cell and paste your key  { display-mode: "form" }
# Run the setup cell above first (offline is fine). Then run this cell and paste
# your workshop key when asked; press Enter without a key to skip. Every later
# model call, including `python -m mini_agent` in a terminal, then goes to
# deepseek/deepseek-v4-flash through OpenRouter. The key is sent only to
# OpenRouter; it is also saved in a .env file on this machine for the terminal.
# To undo: Runtime > Restart session, then run the setup cell again.
import os
import pathlib

if "Model" not in globals():
    raise SystemExit("Run the setup cell above first (offline is fine).")
import mini_agent
from mini_agent.llm.backends import DeepSeekLLM

OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "deepseek/deepseek-v4-flash"


def _workshop_key():
    key = ""
    if os.environ.get("MINI_AGENT_DEEPSEEK_URL") == OPENROUTER_URL:    # entered earlier
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if key:
            print("Using the workshop key entered earlier.")
    if not key:
        try:                                     # Colab Secrets, if stored there
            from google.colab import userdata
            key = userdata.get("OPENROUTER_API_KEY") or ""
            if key:
                print("Key loaded from Colab Secrets (OPENROUTER_API_KEY).")
        except Exception:
            pass
    if not key:
        import getpass
        print("Paste your workshop key below and press Enter (typing is hidden).")
        print("No workshop key? Just press Enter: the notebook keeps running as it is.")
        key = getpass.getpass("Workshop key: ").strip()
    if not key:
        print("No key entered: nothing changed. Run this cell again to use a workshop key.")
        return None
    # OpenRouter speaks the same OpenAI-compatible protocol as DeepSeek's own
    # API, so the DeepSeek backend is reused; only the address and the model change.
    test = DeepSeekLLM(api_key=key, model=OPENROUTER_MODEL)
    test.BASE_URL = OPENROUTER_URL                  # this test call only
    try:
        reply = test("Reply with the single word: ready").strip()
    except Exception as e:
        reply = f"Error: {type(e).__name__}: {e}"
    if reply.startswith("Error"):
        print(f"Connection failed: {reply.removeprefix('Error: ')[:200]}")
        print("Check the key, or continue offline (rerun the setup cell with offline).")
        return None
    DeepSeekLLM.BASE_URL = OPENROUTER_URL           # every model call from now on
    os.environ.update(LLM_BACKEND="deepseek", DEEPSEEK_API_KEY=key,
                      DEEPSEEK_MODEL=OPENROUTER_MODEL,
                      MINI_AGENT_DEEPSEEK_URL=OPENROUTER_URL)
    # a new Python process (a terminal) uses the same address and reads the key
    # and the model from .env
    _site = pathlib.Path(mini_agent.__file__).parent.parent
    (_site / "mini_agent_base_url.pth").write_text(
        'import os; setattr(__import__("mini_agent.llm.backends.deepseek", '
        'fromlist=["DeepSeekLLM"]).DeepSeekLLM, "BASE_URL", '
        'os.environ.get("MINI_AGENT_DEEPSEEK_URL", "https://openrouter.ai/api/v1"))\n')
    for _d in {pathlib.Path("/content"), pathlib.Path.cwd(), _site}:
        if _d.is_dir():
            (_d / ".env").write_text(f"LLM_BACKEND=deepseek\nDEEPSEEK_MODEL={OPENROUTER_MODEL}\n"
                                     f"DEEPSEEK_API_KEY={key}\n")
    live = DeepSeekLLM(api_key=key, model=OPENROUTER_MODEL)
    print(f"Connected: {OPENROUTER_MODEL} via OpenRouter")
    print(f"Test call returned: {reply[:40]}")
    print("The same key and model now work in a terminal too: python -m mini_agent")
    model = Model(live=live, record_to=os.environ.get("T2_RECORD"))
    model.workshop_key = True
    return model


_model = _workshop_key()
if _model is not None:
    llm = _model
'''

TROUBLE = md(r"""
<details>
<summary><b>Troubleshooting</b> (open this if a cell fails)</summary>

| Symptom | Cause | What to do |
|---|---|---|
| `NameError`, for example `llm` or `notes` is not defined | a cell above was skipped, or the runtime restarted | run the setup cell, then the cells of the section from its first cell |
| `Connection failed`, or a reply that starts with `Error:` | invalid key, or no credit left on it | check the key and rerun the cell that printed it (the setup cell or the workshop-key cell), or continue `offline` |
| the 🔑 workshop-key cell waits for input | it asks for the workshop key | paste the key and press Enter; without a key, just press Enter |
| `HTTP 429`, or a timeout | the provider limits the request rate | wait a minute and rerun the cell, or choose `offline` |
| `[offline] This prompt is not in the recording` | offline mode replays this notebook's own prompts only; your change made a new one | use a live provider for your own variants |
| the setup cell asks for a restart | another version of `mini_agent` is loaded | Runtime > Restart session, then run the setup cell again |
| `python -m mini_agent` in a terminal fails (`ModuleNotFoundError`, or `Connection refused`) | the setup cell has not run in this runtime yet | run the setup cell (and the 🔑 cell for a workshop key), then run the command again |
| a different answer, or a different number of steps, than on the speaker's screen | model output varies between runs | expected with a live provider; offline mode gives the same outputs every time |

</details>
""")

# --------------------------------------------------------------------------
# §0 · Where we left off
# --------------------------------------------------------------------------

S0 = [
    md(r"""
---
## Part 0 · Where we left off

**Idea.** At every step, the agent's code rebuilds the whole prompt from scratch: the
instructions, the goal, the tools, a memory section, and the steps taken so far. Run one
small task, then print the first prompt the model received. Each agent in this notebook
gets only the tools its task needs (this one has a calculator) and stops after at most six
steps: limits set in code, as in Tutorial 1.
"""),
    code(r'''
# ▶ one small task, then the first prompt the model received
from mini_agent import Agent
from mini_agent.tools import TOOLS

CALC = {"calc": TOOLS["calc"]}                   # this agent's only tool
CALC_DESC = "- calc(expr): evaluate a math expression"

llm.prompts.clear()                              # llm.prompts keeps every prompt sent from now on
Agent(llm=llm, tools=CALC, tool_desc=CALC_DESC, use_memory=False, skills=[],
      max_steps=6).run(
    "What is 17 * 23?")
show_prompt(llm.prompts[0])
'''),
    md(r"""
**What to look for.** `[Memory]` says `(none)` and `[Steps taken so far]` says `(none)`:
on its first step the model sees no past at all. Everything in this notebook is about what
the agent's code puts into those two sections, and when.
"""),
]

P1 = [
    md(r"""
---
## Part 1 · Why agents need memory

**Idea.** Each model call works from three things: what the model learned in training, the
request in front of it, and whatever task state and evidence the system supplies. The
weights do not change during a task, so anything that happened earlier in it has to be
supplied again by the agent's code. The loop is: restore state, decide and act, observe,
update memory.

In `mini_agent`, *updating* means appending an event to the session log (Tutorial 1, §5),
and *restoring* means projecting the earlier turns from that log into the next prompt. The
class below is the whole mechanism.
"""),
    code(r'''
# ▶ a conversation: each turn goes into a session log, and the log goes back into the next prompt
from pathlib import Path
from mini_agent import SessionLog


class Chat:
    def __init__(self, path):
        Path(path).unlink(missing_ok=True)       # start a fresh log
        self.log = SessionLog(path)
        self.agent = Agent(llm=llm, tools=CALC, tool_desc=CALC_DESC,
                           use_memory=False, skills=[], max_steps=6)

    def history(self):
        """Restore: the conversation so far, as text, projected from the log."""
        return "\n".join(f"{m['role'].title()}: {m['content']}"
                         for m in self.log.derive_messages())

    def say(self, text):
        earlier = self.history()
        self.log.append("user/message", content=text)             # update
        answer = self.agent.run(text, conversation=earlier)       # decide, act, observe
        self.log.append("assistant/message", content=answer)      # update
        return answer


chat = Chat("chat.jsonl")
chat.say("My thin film is 200 nm thick. Please just note it.")
'''),
    md(r"""
**Predict first.** The next cell asks a follow-up question in a *separate* run, without the
conversation. Will the model know which thickness you mean?
"""),
    code(r'''
# ▶ the follow-up as a separate run: nothing from turn 1 is in the prompt
Agent(llm=llm, tools=CALC, tool_desc=CALC_DESC, use_memory=False, skills=[],
      max_steps=6).run(
    "What is that thickness in micrometres?")
'''),
    md(r"""
**What to look for.** Without turn 1, the model either asks which thickness you mean, or
makes one up and answers with confidence. Both come from the same cause: the earlier turn
is not in its prompt. If the trace shows the model trying to run other code through
`calc`, the calculator refuses: it accepts only numbers and arithmetic (Tutorial 1, §2).
"""),
    code(r'''
# ▶ the same follow-up inside the conversation: turn 1 is restored into the prompt
chat.say("What is that thickness in micrometres?")
show_prompt(llm.prompts[-1], only=["Steps taken so far"])
'''),
    md(r"""
**What to look for.** In `[Steps taken so far]`, the block `[Earlier in this session]`
holds turn 1, and `[This turn]` holds the steps of the current run. Same model, same
question; the only difference is that text.

**State and log.** The prompt carries what the next step needs. The log keeps the
original events, so that any answer can be traced back to what was actually said. The log
is an ordinary file of JSON lines:
"""),
    code(r'''
# ▶ the event log behind the conversation, as recorded
import json

for line in open("chat.jsonl"):
    event = json.loads(line)
    print(f"{event['type']:20s}{event['data'].get('content', '')[:90]}")
'''),
]

P2 = [
    md(r"""
---
## Part 2 · Scope, content, writing and reading

**Idea.** Memory can be sorted two ways. By *scope*: short-term memory serves the task in
progress (the conversation in Part 1), and long-term memory carries over to later tasks.
By *content*: semantic memory holds facts, episodic memory holds what happened (the event
log), and procedural memory holds how to act (rules and preferences).

Here the long-term store is a plain file, `notes/long_term.md`, read into `[Memory]` at the
start of every run (Tutorial 1, §7). You can open it from the file browser on the left.

**Writing.** Before a note is stored, compare it with what is already there: add it,
update an old note, delete one, or change nothing (ADD, UPDATE, DELETE, NOOP). The
`remember` tool below files each note under a short subject. A new subject is added. The
same subject with new content replaces the old note, and the old note moves to
`notes/superseded.md`, so the earlier record is kept. The same content again changes
nothing.
"""),
    code(r'''
# ▶ the long-term store, a remember tool that chooses ADD, UPDATE or NOOP, and an agent that reads the notes
from datetime import datetime
from mini_agent.mem import Memory

notes = Memory("notes")                          # the folder notes/ in this runtime
notes.long_term_file.write_text("")              # start with an empty file
SUPERSEDED = Path("notes/superseded.md")
SUPERSEDED.write_text("")


def remember(subject: str = "", content: str = "", source: str = "", **extra) -> str:
    """Save one note under a short subject: ADD, UPDATE (the old note is kept), or NOOP."""
    subject = " ".join(str(subject).lower().split()) or "general"
    content = str(content or next(iter(extra.values()), "")).strip()
    line = f"- [{subject}] {content}" + (f" (source: {source.strip()})" if str(source).strip() else "")
    lines = [l for l in notes.long_term_file.read_text().splitlines() if l.strip()]
    old = [l for l in lines if l.startswith(f"- [{subject}]")]
    if old and old[-1] == line:
        return f"NOOP: [{subject}] is already stored with this content"
    if old:
        notes.long_term_file.write_text("\n".join([l for l in lines if l not in old] + [line]) + "\n")
        with SUPERSEDED.open("a") as f:
            for l in old:
                f.write(f"{l}  (superseded {datetime.now():%Y-%m-%d %H:%M})\n")
        return f"UPDATE: [{subject}] replaced; the old note is kept in notes/superseded.md"
    notes.long_term_file.write_text("\n".join(lines + [line]) + "\n")
    return f"ADD: [{subject}] saved"


TOOLBOX = {                                      # name: (function, the line the model sees)
    "calc": (TOOLS["calc"], "calc(expr): evaluate a math expression"),
    "remember": (remember, "remember(subject, content, source): save one note under a short "
                           "subject; a note with the same subject replaces the old one"),
    "read_file": (TOOLS["read_file"],
                  "read_file(path): read a file; a relative path starts in the workspace"),
}


def agent_with_notes(tools=("calc", "remember")):
    """An agent with the named tools; its [Memory] section is read from notes/ at every run."""
    agent = Agent(llm=llm, tools={t: TOOLBOX[t][0] for t in tools},
                  tool_desc="\n".join(f"- {TOOLBOX[t][1]}" for t in tools),
                  use_memory=False, skills=[], max_steps=8)   # room to read a source
    agent.memory = notes
    return agent
'''),
    code(r'''
# ▶ session 1: a preference (how to act) and a fact (what is known)
agent_with_notes().run("Please remember two things for later: I want temperatures "
                       "reported in kelvin, and sample A was annealed at 450 °C for 2 h.")
print()
print(notes.long_term_file.read_text())
'''),
    md(r"""
**Predict first.** The next cell is a new session: a brand-new agent, with no
conversation at all. Will it know the annealing temperature of sample A? In which unit
will it answer?
"""),
    code(r'''
# ▶ session 2: a new agent and no conversation; only the notes file carries over
agent_with_notes().run("At what temperature was sample A annealed?")
show_prompt(llm.prompts[-1], only=["Memory"])
'''),
    md(r"""
**What to look for.** The notes appear in `[Memory]`, and the answer follows both of them:
the fact (450 °C) and the preference (kelvin). The notes are plain text: anyone can read,
correct, or delete them, and so can the agent.

**Updating.** Facts change. Send a correction, and see which operation the code chooses.
"""),
    code(r'''
# ▶ session 3: a correction; the code compares it with the stored notes before writing
agent_with_notes().run("Correction: sample A was annealed at 480 °C, not 450 °C. "
                       "Please update your note.")
print()
print("notes/long_term.md:")
print(notes.long_term_file.read_text())
print("notes/superseded.md:")
print(SUPERSEDED.read_text())
'''),
    md(r"""
**What to look for.** The tool result says `UPDATE` when the model files the correction
under the subject it used before: the store then holds one current note, and the old one
is kept with the time it was superseded. If the result says `ADD`, the model chose a new
subject, and the store now holds two notes that contradict each other. Exact subjects are
the simplest way to compare; real systems compare a new note with similar stored ones.
"""),
    code(r'''
# ▶ session 4: ask again
agent_with_notes().run("At what temperature was sample A annealed?")
'''),
    md(r"""
**Reading: check before you use.** The agent treats its notes as true, just as it treats
its instructions. A note can be wrong: a typo, an outdated value, or a line that someone
else slipped in. Before a stored record is used, check that it applies: its source, its
date, its version. Three defences: write each note with its source; let the agent check a
note against its source; and put every check that can be automated into code.

The lab's run sheet below is the source. The note on sample B was copied from it, with two
digits swapped.
"""),
    code(r'''
# ▶ the run sheet (the source), and two notes: one of them was copied wrongly
Path("lab").mkdir(exist_ok=True)
Path("lab/runs.csv").write_text(
    "sample,anneal_temperature_C,anneal_time_h,substrate\n"
    "A,400,2,silicon\n"
    "B,540,1,glass\n"
    "C,380,3,glass\n")
NOTES = ("- [sample b anneal] Sample B was annealed at 450 °C for 1 h (source: lab/runs.csv)\n"
         "- [glass substrate limit] Glass substrates are damaged above 500 °C "
         "(source: supplier datasheet)\n")
notes.long_term_file.write_text(NOTES)

print(Path("lab/runs.csv").read_text())
print(notes.long_term_file.read_text())
'''),
    md(r"""
**Predict first.** This agent reads its notes but cannot open files. What will it answer?
"""),
    code(r'''
# ▶ a question that the wrong note answers; this agent cannot open files
QUESTION = "Was sample B annealed below the 500 °C limit of its glass substrate?"
agent_with_notes(tools=("calc",)).run(QUESTION)
'''),
    md(r"""
Now change one thing: the same notes and the same question, but this agent may also read
files. The note names its source, so the agent can look it up.
"""),
    code(r'''
# ▶ the same notes and question; this agent may also read files
notes.long_term_file.write_text(NOTES)           # the notes exactly as before
agent_with_notes(tools=("calc", "read_file")).run(QUESTION)
'''),
    md(r"""
**What to look for.** Did the agent open `lab/runs.csv` before it answered? It could only
do so because the note names its source and the agent has a tool to read it. Whether it
does is up to the model: some models check on their own, others trust the note. The next
cell changes one more thing: it adds a rule to the notes, a piece of procedural memory
(how to act), and asks again.
"""),
    code(r'''
# ▶ the same notes and question, plus one rule in the notes
RULE = "- [rule] Before you use a number from these notes, check it against its source file.\n"
notes.long_term_file.write_text(NOTES + RULE)
agent_with_notes(tools=("calc", "read_file")).run(QUESTION)
'''),
    md(r"""
A rule in the notes makes the check much more likely, but it is still a request: the model
can ignore it. A check in code always runs.
"""),
    code(r'''
# ▶ a check in code: every number needs a source, and notes that cite runs.csv must agree with it
import csv
import re


def audit(text):
    """Print a verdict for every note that contains a number."""
    runs = {row["sample"]: row for row in csv.DictReader(open("lab/runs.csv"))}
    for line in text.splitlines():
        if not line.startswith("- ") or not re.search(r"\d", line):
            continue
        m = re.search(r"[Ss]ample (\w+) was annealed at (\d+) °C", line)
        if "(source:" not in line:
            verdict = "NO SOURCE"
        elif m and "lab/runs.csv" in line:
            true_value = runs[m[1]]["anneal_temperature_C"]
            verdict = "AGREES" if m[2] == true_value else f"MISMATCH: runs.csv says {true_value} °C"
        else:
            verdict = "NOT CHECKED: the source is not a file here"
        print(f"{verdict:44s}{line}")


audit(notes.long_term_file.read_text())
'''),
    md(r"""
*Try it.* Add a note without a source, and run the check again.
"""),
    code(r'''
# ▶ Try it: a note without a source
notes.append_memory("- [sample c anneal] Sample C was annealed at 830 °C")
audit(notes.long_term_file.read_text())
'''),
    md(r"""
The same check can run inside `remember`, so that a note that fails it is never written.
That turns the habit into a rule the model cannot argue with.
"""),
]

P3 = [
    md(r"""
---
## Part 3 · Routes into the context

**Idea.** There are three ways to bring stored information into the next call.
*Full-history replay* sends the original turns again. A *rolling summary* replaces the older
turns with a summary and keeps the recent ones word for word. *External retrieval* keeps the
store outside the prompt and puts in only what the current step needs. Each question in
this part is recorded with its answer and the number of prompt tokens it used, for Part 4.
"""),
    code(r'''
# ▶ a record of every measured question, and a conversation with turn 1 plus twenty more turns
RESULTS = []


def measure(route, expected, run):
    """Run a question; record the answer, whether it is right, and the prompt tokens of all its
    steps (an agent run counts its tokens from zero)."""
    answer = str(run())
    RESULTS.append({"route": route, "prompt tokens": llm.total_prompt_tokens,
                    "correct": expected in answer, "answer": answer})
    return answer


chat3 = Chat("chat3.jsonl")                      # turn 1 as in Part 1, written straight to the log
chat3.log.append("user/message", content="My thin film is 200 nm thick. Please just note it.")
chat3.log.append("assistant/message", content="Noted: the thin film is 200 nm thick.")
SIZES = []
for turn in range(2, 22):
    chat3.log.append("user/message", content=f"Run {turn}: sample S{turn:02d} was annealed "
                     f"at {300 + 10 * turn} °C for {turn % 4 + 1} h. Please note it.")
    chat3.log.append("assistant/message", content=f"Noted: sample S{turn:02d}, "
                     f"{300 + 10 * turn} °C, {turn % 4 + 1} h.")
    SIZES.append(len(chat3.history()))
    print(f"after turn {turn:2d}: {SIZES[-1]:5,d} characters  " + "█" * (SIZES[-1] // 100))
'''),
    md(r"""
**Full-history replay.** Every earlier turn goes back into the prompt, at every step. Ask
about turn 1 now:
"""),
    code(r'''
# ▶ full replay: the whole history is in the prompt
measure("full replay", "0.2", lambda: chat3.say("How thick is my film, in micrometres?"))
'''),
    md(r"""
**Rolling summary.** When the history gets long, the agent can ask the model to summarize
the older turns. `compact()` keeps the last few messages as they are and replaces everything
before them with the summary. The log file itself keeps every original event; only what
goes into the prompt changes.

**Predict first.** Will the summary keep the film thickness from turn 1?
"""),
    code(r'''
# ▶ compaction: the model summarizes the older turns; the summary replaces them in the prompt
summary = chat3.log.compact(llm, keep_tail=4)
print(summary)
print()
COMPACTED = len(chat3.history())
print(f"history after compaction: {COMPACTED:,} characters")
print("Is '200 nm' still in the history?", "200 nm" in chat3.history())
'''),
    code(r'''
# ▶ the same question, answered from the summary and the recent turns
measure("rolling summary", "0.2", lambda: chat3.say("How thick is my film, in micrometres?"))
'''),
    md(r"""
**External retrieval.** Notes pile up, and everything in `[Memory]` is sent again at every
step. `mini_agent` caps that section at about 3,000 characters: beyond that it keeps the
beginning and the end of the file and leaves out the middle (Tutorial 1, §7). This is a
different effect from the one in the lecture, where the model sees every document but uses
the ones in the middle less well: here the code removes the middle before the model sees
anything.
"""),
    code(r'''
# ▶ one note for each of 100 samples, and what of them fits into [Memory]
import random

rng = random.Random(7)                           # a fixed seed: the same notes every time
lines = []
for i in range(1, 101):
    thickness = 80 + int(320 * rng.random())
    hours = 1 + int(4 * rng.random())
    substrate = "glass" if rng.random() < 0.5 else "silicon"
    lines.append(f"- Sample S{i:03d}: film thickness {thickness} nm, "
                 f"annealed {hours} h, grown on {substrate}")
notes.long_term_file.write_text("\n".join(lines) + "\n")

shown = notes.get_long_term()                    # what the agent puts into [Memory]
print(f"{len(notes.long_term_file.read_text()):,} characters in the file, "
      f"{len(shown):,} in the prompt")
print("Is S042 in the prompt?", "S042" in shown)
print()
cut = shown.index("...[")
print(shown[cut - 140:cut + 140])
'''),
    md(r"""
**Predict first.** The note on sample S042 is in the file but not in the prompt. This
agent has only a calculator, so it can answer only from its notes. What will it say?
"""),
    code(r'''
# ▶ all notes, cut to the budget: ask about a sample whose note fell into the left-out middle
QUESTION = ("According to your notes, what is the film thickness of sample S042? "
            "If it is not in your notes, say so.")
measure("all notes, cut to budget", "282",
        lambda: agent_with_notes(tools=("calc",)).run(QUESTION))
'''),
    md(r"""
**What to look for.** The agent either says that the note is missing, or guesses a value.
It cannot see the note: the code left it out of the prompt.

**Retrieval in ten lines.** Score every note by the words it shares with the question, and
keep the best three. Real systems use better scores, but the idea is the same.
"""),
    code(r'''
# ▶ a keyword search over the notes
def words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def search(question, k=3):
    """The k notes that share the most words with the question."""
    all_notes = [n for n in notes.long_term_file.read_text().splitlines() if n.strip()]
    return sorted(all_notes, key=lambda n: len(words(question) & words(n)), reverse=True)[:k]


search(QUESTION)
'''),
    code(r'''
# ▶ retrieval: [Memory] holds only the notes found for this goal
class SearchingAgent(Agent):
    def run(self, goal, **kwargs):
        self.found = search(goal)                # choose the notes for this goal
        return super().run(goal, **kwargs)

    def _get_memory_prompt(self):
        return "\n".join(self.found)


measure("retrieval, top 3", "282",
        lambda: SearchingAgent(llm=llm, tools=CALC, tool_desc=CALC_DESC, use_memory=False,
                               skills=[], max_steps=6).run(QUESTION))
show_prompt(llm.prompts[-1], only=["Memory"])
'''),
    md(r"""
**What to look for.** `[Memory]` is now three lines long instead of 3,000 characters, and
it holds the one note that matters.

*Try it.* Write the sample name differently, and search again.
"""),
    code(r'''
# ▶ Try it: the same question in other words
search("How thick is the S-42 film?")
'''),
    md(r"""
A keyword search finds only the words it is given: "S-42" is not "S042", and "thick" is
not "thickness". Better retrieval compares meanings instead of spellings, or lets the model
choose the search words. Real agents combine the routes: restore the current state, keep
the recent turns and a summary, and retrieve more only when the step needs it.
"""),
]

P4 = [
    md(r"""
---
## Part 4 · Cost and evaluation

**Idea.** Compare the routes on the same question, with the model and the tools held
fixed, and record quality and cost together. Part 3 recorded every question it asked.
"""),
    code(r'''
# ▶ quality and cost of each route, from the questions asked in Part 3
print(f"{'route':28s}{'prompt tokens':>14s}   {'right?':8s}answer")
for r in RESULTS:
    print(f"{r['route']:28s}{r['prompt tokens']:>14,d}   {'yes' if r['correct'] else 'no':8s}"
          f"{r['answer'][:70]}")
'''),
    md(r"""
**What to look for.** Each count covers every step of the run, as reported by the provider
(offline: as recorded). Compare the two pairs: full replay against the summary, on the
conversation; all the notes against retrieval, on the samples. For each route, ask both
questions: was it right, and what did it cost? A route that saves tokens by dropping the
evidence is not a saving.

**Repeated input.** With full replay, every call sends the whole history again. One call
grows with the conversation, and the total over all calls grows faster still. The sizes
measured in Part 3 show it:
"""),
    code(r'''
# ▶ repeated input: with full replay, every call sends the whole history again
total = 0
for n, size in enumerate(SIZES, start=1):
    total += size
    if n in (5, 10, 20):
        print(f"call {n:2d}: this call sends {size:5,d} characters of history; "
              f"all calls so far: {total:6,d}")
print(f"after compaction, one call sends {COMPACTED:,} characters of history")
'''),
    md(r"""
**The KV cache.** Every token in the prompt also occupies memory on the device that runs
the model: for each layer, a key and a value per KV head. A standard estimate, in bytes per
sequence, is 2 × layers × tokens × KV heads × head dimension × bytes per element. The
configuration below is an example, not a particular product.
"""),
    code(r'''
# ▶ the size of the KV cache for one sequence, for an example configuration
layers, kv_heads, head_dim, bytes_per_element = 32, 8, 128, 2      # 16-bit values

for tokens in (4_000, 32_000, 128_000):
    size = 2 * layers * tokens * kv_heads * head_dim * bytes_per_element
    print(f"{tokens:>7,d} tokens: {size / 2**30:5.1f} GiB")
'''),
    md(r"""
*Try it.* Set `kv_heads = 32`, as if every attention head kept its own keys and values,
and run the cell again: the cache is four times larger. Sharing KV heads, storing them at
lower precision, and keeping the prompt short all reduce it; keeping stored memory outside
the prompt costs nothing here until it is retrieved.
"""),
]

S5 = [
    md(r"""
---
## Part 5 · Summary

The model remembers nothing; the agent's code decides what goes back into the prompt.
When memory goes wrong, look at one stage at a time:

| Symptom | Seen in this notebook | Where to look |
|---|---|---|
| Stored, but absent from the input | S042 left out by the budget (Part 3) | what goes into `[Memory]`: budget, retrieval |
| A stale version was used | the 450 °C note, until the correction (Part 2) | updates: `notes/superseded.md` |
| A wrong record was used | the note copied with two digits swapped (Part 2) | sources: `audit`, `read_file` |
| A detail lost on the way | a summary that drops the thickness (Part 3) | what the summary keeps |
| Missing from the store | (not shown) a fact the agent never saved | the write step: `remember` |

Four habits:

1. **Write with a source.** A note without a source cannot be checked.
2. **Show only what is relevant.** Retrieval keeps the prompt small and the right note in it.
3. **Check before you rely on it.** A remembered number is not a measured one.
4. **Put the checks in code.** What code enforces holds; what you only ask for may not.

Tomorrow morning, Dmytro and Chengran cover MCP and skills, which is procedural memory
packaged for reuse. On Wednesday, Nanta covers evaluation: how to know that any of this
works.
"""),
]

APPENDIX = [
    md(r"""
---
## Appendix · How offline mode works

Offline mode answers every model call from a recording of a live run of this notebook. The
recording is embedded in the setup cell, and the tools still run for real. Each prompt is
looked up by its full text, after masking what differs between machines and days (the
workspace path, dates, and times). The memory section is part of the match, so a prompt
with a different note in it is a different prompt. A prompt that is not in the recording
gets the notice `[offline] This prompt is not in the recording`, never a reply recorded
for another prompt. Your own variants therefore need a live provider.
"""),
]


def build():
    blob = ""
    if RECORDING.exists():
        blob = base64.b64encode(gzip.compress(RECORDING.read_bytes(), mtime=0)).decode()
    setup = (SETUP.replace("__DEFAULT_PROVIDER__", DEFAULT_PROVIDER)
                  .replace("__PROVIDERS__", json.dumps(PROVIDERS))
                  .replace("__RECORDING__", blob))
    logo = base64.b64encode((ROOT / "assets" / "agentforge_logo.png").read_bytes()).decode()
    title = md(TITLE.source.replace("__LOGO__", f"data:image/png;base64,{logo}"))
    cells = ([title, COLAB_NOTE, code(setup), WORKSHOP_KEY_NOTE, code(OPENROUTER), TROUBLE]
             + S0 + P1 + P2 + P3 + P4 + S5 + APPENDIX)
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, OUT)
    n_code = sum(c.cell_type == "code" for c in cells)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(cells)} cells ({n_code} code), "
          f"recording {'embedded' if blob else 'absent'} ({len(blob):,} characters)")


if __name__ == "__main__":
    build()
