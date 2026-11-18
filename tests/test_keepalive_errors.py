"""Keep-alive and model errors, for every backend: which errors stop the task and which continue it.

The agreed rule: a failed model call is temporary (keep-alive continues the task) only without an
HTTP status (the server unreachable, a timeout, a reply that is not an answer), or with 408, 429 or
a 5xx; any other 4xx stops the task at once, with exit 1 and no wait.

Each test takes one backend (and one LLM_STREAM setting) and starts
`mini_agent --yes --keep-alive 1 "Say hi."` once per kind of error, all at the same time, each in
its own home and folder, against a fake server that answers every request with that error. A
permanent error must end with exit 1 and no keepalive/continue; a temporary one with exit 75 after
one continue. The error messages echo the key the request carried (as some gateways do), and the
key must appear nowhere: not on the screen, not in the session log, not in any file written.

Streaming: the agent loop streams whenever the backend can (think() uses call_stream), so
LLM_STREAM changes the request only for the Anthropic API (the claude backend, and an
Anthropic-compatible server named with LLM_PROVIDER=anthropic). Both settings run everywhere.
A timeout is real but short: that case loads a sitecustomize that caps httpx timeouts at 0.5 s,
and the server waits 3 s before answering. Qwen has no setting for its address, so it cannot be
pointed at the fake server; it is the OpenAI-compatible backend with fixed addresses.
"""
import collections
import http.server
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

KEY = "test-key-sweep-0123456789abcdef0123456789"
PERMANENT = ("400", "401", "402", "403", "404", "413", "422")
TEMPORARY = ("408", "429", "500", "502", "503", "529", "refused", "timeout", "html200")
STALL = 3.0                 # seconds the server waits on a "timeout" request
PASS_THROUGH = ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATHEXT", "COMSPEC")

SITECUSTOMIZE = '''\
import os
_cap = float(os.environ.get("SWEEP_HTTP_TIMEOUT") or 0)
if _cap:
    import httpx
    _init = httpx.Timeout.__init__

    def _capped(self, *args, **kwargs):
        _init(self, *args, **kwargs)
        for name in ("connect", "read", "write", "pool"):
            value = getattr(self, name)
            if value is None or value > _cap:
                setattr(self, name, _cap)

    httpx.Timeout.__init__ = _capped
'''


def body_for(family: str, status: int, message: str) -> dict:
    """An error body in the provider's own format."""
    if family == "anthropic":
        return {"type": "error", "error": {"type": "api_error", "message": message}}
    if family == "ollama":
        return {"error": message}
    return {"error": {"message": message, "type": "api_error", "code": status}}


class ErrorServer:
    """Answers every request under /<kind>/ with that kind of error; .hits counts requests."""

    def __init__(self, family: str):
        hits = self.hits = collections.Counter()

        class Handler(http.server.BaseHTTPRequestHandler):
            def answer(self):
                kind = self.path.strip("/").split("/")[0]
                hits[kind] += 1
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                key = (self.headers.get("x-api-key")
                       or self.headers.get("Authorization", "").replace("Bearer ", "").strip())
                try:
                    if kind == "timeout":
                        time.sleep(STALL)
                        status, kind_of, data = 504, "application/json", b"{}"
                    elif kind == "html200":
                        status, kind_of = 200, "text/html"
                        data = ("<html><head><title>502 Bad Gateway</title></head><body>"
                                f"<h1>Bad Gateway</h1><p>key {key}</p></body></html>").encode()
                    else:
                        status, kind_of = int(kind), "application/json"
                        message = f"request refused ({kind}); the key sent was {key}"
                        data = json.dumps(body_for(family, status, message)).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", kind_of)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:              # the client gave up (the timeout case)
                    pass

            do_GET = do_POST = answer

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.port = self.httpd.server_address[1]

    def close(self):
        self.httpd.shutdown()


def closed_port() -> int:
    """A port nothing listens on (connection refused)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# backend -> (error format, the settings that point it at a base URL)
MODES = {
    "openai_compat": ("openai", lambda url: dict(LLM_BACKEND="openai_compat", LLM_BASE_URL=url,
                                                 LLM_API_KEY=KEY)),
    "openrouter_preset": ("openai", lambda url: dict(LLM_BACKEND="openrouter", LLM_BASE_URL=url,
                                                     OPENROUTER_API_KEY=KEY)),
    "deepseek": ("openai", lambda url: dict(LLM_BACKEND="deepseek", DEEPSEEK_BASE_URL=url,
                                            DEEPSEEK_API_KEY=KEY)),
    "claude": ("anthropic", lambda url: dict(LLM_BACKEND="claude", ANTHROPIC_BASE_URL=url,
                                             ANTHROPIC_API_KEY=KEY)),
    "anthropic_by_provider": ("anthropic", lambda url: dict(
        LLM_BACKEND="openai_compat", LLM_PROVIDER="anthropic", LLM_BASE_URL=url, LLM_API_KEY=KEY)),
    "ollama": ("ollama", lambda url: dict(LLM_BACKEND="ollama", OLLAMA_HOST=url)),
}


def events_of(home: Path) -> list:
    out = []
    for log in sorted((home / ".mini_agent" / "sessions").glob("*.jsonl")):
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def leaks(*roots: Path) -> list:
    """Files under the roots that contain the key."""
    found = []
    for root in roots:
        for p in root.rglob("*"):
            if p.is_file() and not p.is_symlink() and KEY.encode() in p.read_bytes():
                found.append(str(p.relative_to(root.parent)))
    return found


# LLM_STREAM switches only the Claude backend; the others take the same path either way
GROUPS = [(mode, "false") for mode in MODES] + [("claude", "true"), ("anthropic_by_provider", "true")]


@pytest.mark.parametrize("mode,stream", GROUPS)
def test_each_model_error_stops_or_continues_as_agreed_and_never_shows_the_key(
        tmp_path, mode, stream):
    family, settings = MODES[mode]
    server = ErrorServer(family)
    refused = closed_port()
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
    runs = {}
    try:
        for kind in PERMANENT + TEMPORARY:
            port = refused if kind == "refused" else server.port
            url = f"http://127.0.0.1:{port}/{kind}" + ("" if family == "ollama" else "/v1")
            cell = tmp_path / kind
            home, work = cell / "home", cell / "work"
            home.mkdir(parents=True)
            work.mkdir()
            env = {k: os.environ[k] for k in PASS_THROUGH if k in os.environ}
            env.update(HOME=str(home), USERPROFILE=str(home), PYTHONIOENCODING="utf-8",
                       MINI_AGENT_HOME=str(home / ".mini_agent"), MINI_AGENT_KEEPALIVE_WAITS="0",
                       LLM_MODEL="fake", LLM_STREAM=stream, **settings(url))
            if kind == "timeout":
                env.update(PYTHONPATH=str(site), SWEEP_HTTP_TIMEOUT="0.5")
            out = open(cell / "stdout.txt", "w+", encoding="utf-8")
            err = open(cell / "stderr.txt", "w+", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, "-m", "mini_agent", "--yes", "--keep-alive",
                                     "1", "Say hi."], cwd=work, env=env,
                                    stdin=subprocess.DEVNULL, stdout=out, stderr=err)
            runs[kind] = (proc, out, err, home, work)
        deadline = time.monotonic() + 240
        for proc, *_ in runs.values():
            try:
                proc.wait(max(1.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    finally:
        server.close()

    table, failures = [], []
    for kind, (proc, out, err, home, work) in runs.items():
        out.close()
        err.close()
        stdout = Path(out.name).read_text(encoding="utf-8", errors="replace")
        stderr = Path(err.name).read_text(encoding="utf-8", errors="replace")
        events = events_of(home)
        continues = sum(e.get("type") == "keepalive/continue" for e in events)
        expected = "stop" if kind in PERMANENT else "continue"
        if expected == "stop":
            right = proc.returncode == 1 and continues == 0
        else:
            right = proc.returncode == 75 and continues >= 1
        shown = KEY in stdout or KEY in stderr
        files = leaks(home, work)
        why = next((e["data"].get("answer") for e in reversed(events)
                    if e.get("type") == "turn/end"), "") or ""
        line = (f"{kind:8} want {expected:8} got exit {proc.returncode}, {continues} continue(s), "
                f"{server.hits[kind]} request(s); last error: {str(why)[:90]!r}")
        if shown or files:
            line += f"; KEY SHOWN: {'screen ' if shown else ''}{' '.join(files)}"
        table.append(line)
        if not right or shown or files:
            failures.append(line)
    print(f"\n{mode}, LLM_STREAM={stream}:\n" + "\n".join(table))
    assert not failures, f"{mode}, LLM_STREAM={stream}:\n" + "\n".join(failures)
