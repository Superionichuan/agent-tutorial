"""The command line, run for real: python -m mini_agent against a fake OpenAI-compatible server."""
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mini_agent import __version__

BASH_THEN_DONE = [
    json.dumps({"thought": "run it", "action": "bash", "args": {"cmd": "echo $((6*7))-ran"}}),
    json.dumps({"thought": "done", "action": "done", "args": {"answer": "finished"}}),
]


class FakeServer:
    """Answers /v1/chat/completions with scripted replies, in order (streamed when asked)."""

    def __init__(self, replies, status=200):
        self.replies, self.status, self.requests = list(replies), status, []
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append(body)
                if server.status != 200:
                    data = json.dumps({"error": {"message": "invalid api key"}}).encode()
                    self.send_response(server.status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                text = server.replies.pop(0) if server.replies else '{"action": "done", "args": {"answer": "?"}}'
                usage = {"prompt_tokens": 10, "completion_tokens": 5}
                if body.get("stream"):
                    chunks = [{"choices": [{"index": 0, "delta": {"content": text}}]},
                              {"choices": [], "usage": usage}]
                    data = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
                    data, kind = data.encode(), "text/event-stream"
                else:
                    data = json.dumps({"choices": [{"message": {"role": "assistant", "content": text},
                                                    "finish_reason": "stop"}], "usage": usage}).encode()
                    kind = "application/json"
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def close(self):
        self.httpd.shutdown()


def env_for(server=None, **extra):
    home = tempfile.mkdtemp()
    env = {"PATH": os.environ.get("PATH", ""), "HOME": home, "USERPROFILE": home,
           "MINI_AGENT_HOME": os.path.join(home, ".mini_agent"), "PYTHONIOENCODING": "utf-8",
           "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
    if server is not None:
        env.update(LLM_BACKEND="openai_compat", LLM_BASE_URL=server.url, LLM_MODEL="fake")
    env.update(extra)
    return env


def cli(*args, env, stdin=subprocess.DEVNULL, cwd=None):
    return subprocess.run([sys.executable, "-m", "mini_agent", *args], env=env, stdin=stdin,
                          capture_output=True, text=True, timeout=120, cwd=cwd or tempfile.mkdtemp())


def test_help_and_version():
    r = cli("--help", env=env_for())
    assert r.returncode == 0 and "--approve" in r.stdout and "OPENROUTER_API_KEY" in r.stdout
    assert "Query:" not in r.stdout
    r = cli("--version", env=env_for())
    assert r.returncode == 0 and r.stdout.strip() == f"mini_agent {__version__}"


def test_no_model_configured_is_explained():
    r = cli("hello", env=env_for())
    # nothing configured: unless an Ollama server happens to run here, the CLI explains what to set
    if r.returncode == 2:
        assert "no model is configured" in r.stderr and "OPENROUTER_API_KEY" in r.stderr
        assert "Traceback" not in r.stderr


def test_risky_tool_is_not_run_without_a_terminal_to_ask():
    server = FakeServer(BASH_THEN_DONE)
    try:
        r = cli("do", "it", env=env_for(server))
    finally:
        server.close()
    assert r.returncode == 0, r.stderr
    assert "needs approval, but there is no terminal" in r.stderr
    assert "was not approved" in r.stdout and "42-ran" not in r.stdout
    assert "Answer: finished" in r.stdout


def test_yes_runs_risky_tools():
    server = FakeServer(BASH_THEN_DONE)
    try:
        r = cli("--yes", "do", "it", env=env_for(server))
    finally:
        server.close()
    assert r.returncode == 0 and "42-ran" in r.stdout and "Answer: finished" in r.stdout


def test_deny_never_asks():
    server = FakeServer(BASH_THEN_DONE)
    try:
        r = cli("--approve", "deny", "do", "it", env=env_for(server))
    finally:
        server.close()
    assert "no terminal" not in r.stderr and "was not approved" in r.stdout


def test_bad_key_stops_after_one_call():
    server = FakeServer([], status=401)
    try:
        r = cli("hello", env=env_for(server))
    finally:
        server.close()
    assert r.returncode == 0 and "Answer: Error: HTTP 401: invalid api key" in r.stdout
    assert len(server.requests) == 1 and "Traceback" not in r.stderr


def test_offline_recording_from_the_command_line():
    r = cli("Compute 3**7.", env=env_for(LLM_BACKEND="offline"))
    assert r.returncode == 0, r.stderr
    assert "2187" in r.stdout.split("Answer:")[-1]


def test_options_before_the_task_and_dashes_inside_it():
    server = FakeServer([json.dumps({"action": "done", "args": {"answer": "ok"}})])
    try:
        r = cli("--yes", "compute", "-3*2", env=env_for(server))
    finally:
        server.close()
    assert "Query: compute -3*2" in r.stdout and "Answer: ok" in r.stdout


@pytest.mark.skipif(os.name == "nt", reason="pseudo-terminals are POSIX-only")
@pytest.mark.parametrize("mode", ["one-shot", "interactive"])
def test_asks_on_a_terminal_and_runs_after_yes(mode):
    import pty
    import select

    server = FakeServer(BASH_THEN_DONE)
    master, slave = pty.openpty()
    args = ["do it"] if mode == "one-shot" else []
    proc = subprocess.Popen([sys.executable, "-m", "mini_agent", *args], env=env_for(server, TERM="xterm"),
                            stdin=slave, stdout=slave, stderr=slave, cwd=tempfile.mkdtemp())
    os.close(slave)
    seen = b""

    def wait_for(text: bytes, timeout=60):
        nonlocal seen
        end = time.time() + timeout
        while text not in seen:
            if time.time() > end:
                raise AssertionError(f"did not see {text!r}; output so far:\n{seen[-2000:]!r}")
            ready, _, _ = select.select([master], [], [], 0.2)
            if ready:
                try:
                    seen += os.read(master, 65536)
                except OSError:
                    break

    try:
        if mode == "interactive":
            wait_for(b"\xe2\x9d\xaf")                   # the prompt
            os.write(master, b"do it\r")
        wait_for(b"Allow bash?")
        os.write(master, b"y\r")
        wait_for(b"finished")
        if mode == "interactive":
            os.write(master, b"/exit\r")
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
        os.close(master)
        server.close()
    assert b"42-ran" in seen
