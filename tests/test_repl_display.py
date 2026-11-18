"""The interactive mode's live area: an approval prompt stays on the screen, and the area
(reasoning, the streamed reply, the running tool) never leaves lines behind in the scrollback."""
import http.server
import io
import json
import os
import re
import sys
import threading

from rich.console import Console
from rich.live import Live
from rich.text import Text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mini_agent.ui import repl

WIDTH, HEIGHT = 80, 24
CSI = re.compile(r"\x1b\[([0-9;?]*)([A-Za-z])")
PROMPT = "[y]es / [n]o / [a]lways"


class Terminal:
    """Just enough of a terminal for what Rich writes: text, CR, LF, cursor up, erase line.
    A line that scrolls off the top of the screen stays in the scrollback, unchangeable."""

    def __init__(self, width=WIDTH, height=HEIGHT):
        self.width, self.height = width, height
        self.lines, self.row, self.col, self.bottom = [""], 0, 0, 0

    def _down(self):
        self.row += 1
        if self.row == len(self.lines):
            self.lines.append("")
        self.bottom = max(self.bottom, self.row)
        self.col = 0

    def feed(self, data: str):
        i = 0
        while i < len(data):
            m = CSI.match(data, i)
            if m:
                arg, cmd = m.groups()
                if cmd == "A":                         # cursor up, not above the screen's top
                    self.row = max(max(0, self.bottom - self.height + 1), self.row - int(arg or 1))
                elif cmd == "K" and arg == "2":        # erase the whole line
                    self.lines[self.row] = ""
                i = m.end()
                continue
            ch, i = data[i], i + 1
            if ch == "\r":
                self.col = 0
            elif ch == "\n":                           # a terminal turns LF into CR LF
                self._down()
            elif ch >= " ":
                if self.col >= self.width:
                    self._down()
                line = self.lines[self.row].ljust(self.col)
                self.lines[self.row] = line[:self.col] + ch + line[self.col + 1:]
                self.col += 1


def make_console(buf):
    return Console(file=buf, force_terminal=True, force_interactive=True, width=WIDTH,
                   height=HEIGHT, color_system=None, legacy_windows=False)


def test_the_live_area_stays_short():
    console = make_console(io.StringIO())
    status = repl.LiveStatus()
    status.reasoning = " ".join(f"Reasoning line {i:03d}." for i in range(400))
    status.text = '{"thought": "' + "word " * 2000 + 'last-word"}'
    lines = console.render_lines(status, console.options)
    assert len(lines) <= 1 + repl.REASONING_LINES + repl.TEXT_LINES
    shown = "".join(seg.text for line in lines for seg in line)
    assert "399." in shown and "last-word" in shown       # the newest part is what is shown
    status.mode = "tool"
    status.tool_line = "write_file({'path': 'a.txt', 'content': '" + "y" * 5000 + "'})"
    assert len(console.render_lines(status, console.options)) == 1


def test_the_live_area_starts_again_below_the_prompt():
    buf = io.StringIO()
    console = make_console(buf)
    live = Live(Text("frame one\nframe two\nframe three"), console=console, transient=True,
                auto_refresh=False)
    live.start(refresh=True)
    live.stop()                                        # as when an approval is asked
    console.print(Text("Allow bash?"))
    console.print(Text(f"{PROMPT} for this tool: y"))
    mark = len(buf.getvalue())
    fresh = repl._restarted(live)
    fresh.refresh()
    assert not re.search(r"\x1b\[\d*A", buf.getvalue()[mark:])   # nothing above it is erased
    fresh.stop()
    term = Terminal()
    term.feed(buf.getvalue())
    assert f"{PROMPT} for this tool: y" in term.lines
    assert not [line for line in term.lines if "frame" in line]   # transient: cleared


class ReasoningServer:
    """An OpenAI-compatible server: each reply is streamed as reasoning first, then the text."""

    def __init__(self, replies):
        self.replies = list(replies)
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                why, text = server.replies.pop(0) if server.replies else (
                    "", '{"action": "done", "args": {"answer": "?"}}')
                usage = {"prompt_tokens": 10, "completion_tokens": 5}
                if body.get("stream"):
                    parts = [{"reasoning_content": why[k:k + 300]} for k in range(0, len(why), 300)]
                    parts += [{"content": text[k:k + 300]} for k in range(0, len(text), 300)]
                    chunks = [{"choices": [{"index": 0, "delta": d}]} for d in parts]
                    chunks.append({"choices": [], "usage": usage})
                    data = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
                    data, kind = data.encode(), "text/event-stream"
                else:
                    msg = {"role": "assistant", "content": text, "reasoning_content": why}
                    data = json.dumps({"choices": [{"message": msg, "finish_reason": "stop"}],
                                       "usage": usage}).encode()
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


def test_an_approval_keeps_its_prompt_and_leaves_no_reasoning(tmp_path, monkeypatch):
    content = "".join(f"note {i:03d}: a line of the file\n" for i in range(120))
    reasoning = " ".join(f"Reasoning line {i:03d} about the notes." for i in range(120))
    server = ReasoningServer([
        (reasoning, json.dumps({"thought": "write the notes", "action": "write_file",
                                "args": {"path": "notes.txt", "content": content}})),
        (reasoning, json.dumps({"thought": "done", "action": "done",
                                "args": {"answer": "Wrote notes.txt."}})),
    ])
    for name in list(os.environ):
        if name.startswith(("LLM_", "DEEPSEEK_", "OPENROUTER_", "ANTHROPIC_", "CLAUDE_", "QWEN_",
                            "OPENAI_", "GEMINI_", "OLLAMA_", "MINI_AGENT_")):
            monkeypatch.delenv(name)
    home = tmp_path / "home"
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("MINI_AGENT_HOME", str(home / ".mini_agent"))
    monkeypatch.setenv("LLM_BACKEND", "openai_compat")
    monkeypatch.setenv("LLM_BASE_URL", server.url)
    monkeypatch.setenv("LLM_MODEL", "fake")
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.chdir(work)
    monkeypatch.setattr(repl, "SESS_DIR", home / ".mini_agent" / "sessions")

    buf = io.StringIO()
    console = make_console(buf)
    monkeypatch.setattr(repl, "Console", lambda: console)
    monkeypatch.setattr(repl, "stdin_is_terminal", lambda: True)
    monkeypatch.setattr(repl.pets, "intro_frames", lambda pet: [(repl.pets.render_pet(pet, 0), 0)])
    typed = ["Write the notes.", "y"]

    def keyboard(prompt=""):             # what is typed, echoed by the terminal
        if not typed:
            raise EOFError
        line = typed.pop(0)
        buf.write(line + "\n")
        return line

    monkeypatch.setattr("builtins.input", keyboard)
    try:
        status = repl.run_repl(agent_options={"approve": "ask"})
    finally:
        server.close()
    assert status == 0
    assert (work / "notes.txt").read_text() == content
    term = Terminal()
    term.feed(buf.getvalue())
    lines = term.lines
    transcript = "\n".join(lines)
    assert sum("Allow write_file?" in line for line in lines) == 1, transcript
    assert sum(PROMPT in line for line in lines) == 1, transcript
    # the live area's parts are never left behind: the reasoning, the streamed reply (its
    # JSON), the running tool's line
    assert not [line for line in lines if "Reasoning line" in line], transcript
    assert not [line for line in lines if '"thought"' in line or '"action"' in line], transcript
    assert not [line for line in lines if "Running…" in line], transcript
    assert any("Wrote notes.txt." in line for line in lines), transcript
