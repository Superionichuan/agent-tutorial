"""Run the notebook top to bottom as a participant would, in a fresh workspace.

    python tools/run_fresh.py NOTEBOOK OUT.ipynb offline
    python tools/run_fresh.py NOTEBOOK OUT.ipynb deepseek   --env-file PATH [--record REC.jsonl]
    python tools/run_fresh.py NOTEBOOK OUT.ipynb openrouter --env-file PATH [--stand-in]
    ... [--compare REF.ipynb]   compare every code cell's output with a reference run

Run it with the Python of a fresh virtual environment (nbclient, ipykernel,
pandas). Differences from a Colab VM, all deliberate:
  * no google.colab module, so a live run takes the getpass path; a hidden first
    cell makes the kernel's getpass return the key from this kernel's
    environment once, and an empty answer after that, as when a participant
    presses Enter at a prompt they do not need (the hidden cell is removed
    before the notebook is saved);
  * deepseek: PROVIDER is set to "deepseek" in the setup cell;
    openrouter: PROVIDER stays "offline" and the workshop key is pasted into the
    workshop-key cell, which is what participants do with the workshop key;
    --key-in-setup pastes it into the setup cell instead (PROVIDER = deepseek);
    --stand-in points the workshop-key cell at DeepSeek's own endpoint and model,
    with the DeepSeek key, to test the cell's path without an OpenRouter key.
    All changes to cell sources are undone before the notebook is saved.
Keys are read from --env-file (only DEEPSEEK_API_KEY / DEEPSEEK_MODEL, or
OPENROUTER_API_KEY), never printed, and searched for in every file the run
produced before the summary is printed.
"""
import argparse
import json
import os
import re
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient

p = argparse.ArgumentParser()
p.add_argument("notebook")
p.add_argument("out")
p.add_argument("provider", choices=["offline", "deepseek", "openrouter"])
p.add_argument("--env-file")
p.add_argument("--record")
p.add_argument("--compare")
p.add_argument("--no-network", action="store_true",
               help="refuse every network connection from the kernel and its subprocesses")
p.add_argument("--base", default=os.environ.get("SIM_BASE"))
p.add_argument("--stand-in", action="store_true",
               help="openrouter mode only: use DeepSeek's endpoint and the DeepSeek key")
p.add_argument("--key-in-setup", action="store_true",
               help="openrouter mode only: paste the workshop key into the setup cell")
a = p.parse_args()


def read_env(path, names):
    values = {}
    for line in open(path):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() in names:
                values[k.strip()] = v.strip().strip('"').strip("'")
    return values


nb = nbformat.read(a.notebook, as_version=4)
setup = [c for c in nb.cells if c.cell_type == "code" and "@title ⚙ Setup" in c.source]
assert len(setup) == 1, "setup cell not found"
PARAM = re.compile(r'^PROVIDER = "offline"(?=  # @param)', re.M)
assert len(PARAM.findall(setup[0].source)) == 1, "PROVIDER line not found"

key, extra_env = "", {}
if a.provider == "deepseek":
    env = read_env(a.env_file, {"DEEPSEEK_API_KEY", "DEEPSEEK_MODEL"})
    key = env.get("DEEPSEEK_API_KEY", "")
    if env.get("DEEPSEEK_MODEL"):
        extra_env["DEEPSEEK_MODEL"] = env["DEEPSEEK_MODEL"]
    setup[0].source = PARAM.sub('PROVIDER = "deepseek"', setup[0].source, count=1)
elif a.provider == "openrouter":
    orc = [c for c in nb.cells if c.cell_type == "code" and "@title 🔑 Workshop key" in c.source]
    assert len(orc) == 1, "workshop-key cell not found"
    original_or = orc[0].source
    assert not (a.stand_in and a.key_in_setup), "--stand-in tests the key cell only"
    if a.stand_in:
        for old, new in (('OPENROUTER_URL = "https://openrouter.ai/api/v1"',
                          'OPENROUTER_URL = "https://api.deepseek.com/v1"'),
                         ('OPENROUTER_MODEL = "deepseek/deepseek-v4-flash"',
                          'OPENROUTER_MODEL = "deepseek-v4-flash"')):
            assert old in orc[0].source, old
            orc[0].source = orc[0].source.replace(old, new)
        key = read_env(a.env_file, {"DEEPSEEK_API_KEY"}).get("DEEPSEEK_API_KEY", "")
    else:
        key = read_env(a.env_file, {"OPENROUTER_API_KEY"}).get("OPENROUTER_API_KEY", "")
    if a.key_in_setup:
        setup[0].source = PARAM.sub('PROVIDER = "deepseek"', setup[0].source, count=1)
online = a.provider != "offline"
if online:
    assert key, "no key found in --env-file"
if True:                                        # every run: answer getpass like a participant
    hidden = nbformat.v4.new_code_cell(
        "import os as _os\n"
        "get_ipython().kernel.getpass = (\n"
        "    lambda prompt='', *args, **kwargs: _os.environ.pop('SIM_PROVIDER_KEY', ''))\n")
    hidden.metadata["sim_hidden"] = True
    nb.cells.insert(0, hidden)

if a.no_network:                               # any attempt to connect fails loudly
    blocker = nbformat.v4.new_code_cell(
        "import socket as _s\n"
        "def _refuse(*args, **kwargs):\n"
        "    raise OSError('network disabled for this test')\n"
        "_s.socket.connect = _refuse\n"
        "_s.socket.connect_ex = _refuse\n"
        "_s.create_connection = _refuse\n")
    blocker.metadata["sim_hidden"] = True
    nb.cells.insert(0, blocker)
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ[k] = "http://127.0.0.1:9"    # nothing listens there

work = tempfile.mkdtemp(prefix="content_", dir=a.base)
home = tempfile.mkdtemp(prefix="home_", dir=a.base)
for k in ("PYTHONPATH", "LLM_BACKEND", "LLM_MODEL", "LLM_RECORD", "LLM_RECORDING", "T2_RECORD",
          "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "QWEN_API_KEY", "CLAUDE_API_KEY",
          "OPENROUTER_API_KEY", "JUPYTER_PATH", "JUPYTER_CONFIG_DIR", "IPYTHONDIR"):
    os.environ.pop(k, None)
os.environ.update({"HOME": home, "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", "MPLBACKEND": "Agg"})
os.environ.update(extra_env)
if a.record:
    os.environ["T2_RECORD"] = os.path.abspath(a.record)
if online:
    os.environ["SIM_PROVIDER_KEY"] = key        # inherited by the kernel, popped by its getpass

client = NotebookClient(nb, timeout=900, kernel_name="python3", allow_errors=True,
                        record_timing=True, resources={"metadata": {"path": work}})
t0 = time.time()
client.execute()
seconds = round(time.time() - t0)
os.environ.pop("SIM_PROVIDER_KEY", None)
nb.cells = [c for c in nb.cells if not c.metadata.get("sim_hidden")]
setup[0].source = re.sub(r'^PROVIDER = "deepseek"(?=  # @param)', 'PROVIDER = "offline"',
                         setup[0].source, count=1, flags=re.M)
if a.provider == "openrouter":
    orc[0].source = original_or                  # the saved notebook keeps the delivered cell

# ---- key scan: the saved notebook, the workspace, HOME, the recording ----
text = nbformat.writes(nb)
scanned = 0
if online:
    assert key not in text, "key found in the executed notebook"
    paths = [os.path.join(r, f) for top in (work, home) for r, _, fs in os.walk(top) for f in fs]
    if a.record:
        paths.append(os.path.abspath(a.record))
    for path in paths:
        if os.path.isfile(path) and not os.path.islink(path):
            with open(path, errors="ignore") as fh:
                assert key not in fh.read(), f"key found in {path}"
            scanned += 1
sk = len(re.findall(r"sk-[A-Za-z0-9_-]{8,}", text))
nbformat.write(nb, a.out)


# ---- summary ----
def outputs(cell):
    parts = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            parts.append(o.get("text", ""))
        elif o.get("output_type") in ("execute_result", "display_data"):
            parts.append(o.get("data", {}).get("text/plain", ""))
        elif o.get("output_type") == "error":
            parts.append(f"ERROR {o.get('ename')}: {o.get('evalue')}")
    return re.sub(r"/[^\s'\"]*content_[A-Za-z0-9_]+", "<ws>", "".join(parts))


code_cells = [c for c in nb.cells if c.cell_type == "code"]
errors = [(i, o.get("ename"), o.get("evalue", "")[:160]) for i, c in enumerate(code_cells)
          for o in c.get("outputs", []) if o.get("output_type") == "error"]
not_recorded = sum(outputs(c).count("not in the recording") for c in code_cells)
def cell_seconds(cell):
    t = cell.get("metadata", {}).get("execution", {})
    try:
        from datetime import datetime
        t0 = datetime.fromisoformat(t["iopub.execute_input"].replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(t["shell.execute_reply"].replace("Z", "+00:00"))
        return round((t1 - t0).total_seconds(), 1)
    except (KeyError, ValueError):
        return None


summary = {"provider": a.provider, "python": sys.version.split()[0], "seconds": seconds,
           "setup_cell_seconds": cell_seconds(setup[0]), "no_network": a.no_network,
           "code_cells": len(code_cells), "errors": errors,
           "offline_not_recorded_notices": not_recorded,
           "key_scan": f"{scanned} files + notebook, key not found" if online else "n/a",
           "sk_pattern_in_notebook": sk, "workdir": work}
if a.compare:
    ref = [c for c in nbformat.read(a.compare, as_version=4).cells if c.cell_type == "code"]
    diff = [i for i, (x, y) in enumerate(zip(code_cells, ref))
            if "@title" not in x.source and outputs(x) != outputs(y)]
    summary["cells_differing_from_reference"] = diff
print(json.dumps(summary, indent=1, ensure_ascii=False))
