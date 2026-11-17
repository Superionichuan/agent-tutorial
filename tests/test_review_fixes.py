"""Regression tests for the findings of the first independent review of v2.

Approval of wrapped risky tools; the streamed reply parser and malformed replies;
the order of the settings sources; setup / doctor (keys never printed, no backup
copy, symlinks, damaged files, a closed terminal); argument schemas and docstrings.
"""
import functools
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from mini_agent import Agent, tool
from mini_agent.cli import doctor, setup
from mini_agent.core.executor import execute
from mini_agent.core.policy import Approval
from mini_agent.llm import factory
from mini_agent.llm.backends.claude import ClaudeLLM
from mini_agent.llm.backends.deepseek import DeepSeekLLM
from mini_agent.llm.backends.openai_compat import OpenAICompatLLM
from mini_agent.llm.backends.qwen import QwenLLM
from mini_agent.llm.base import LLMError, mask, redact
from mini_agent.tools import ToolSet
from test_harness import J, ScriptedLLM

SECRET = "sk-fake-THIS_KEY_MUST_NEVER_BE_PRINTED"


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """No settings from the real environment, home, working directory or package."""
    for key in factory.CONFIG_KEYS:
        monkeypatch.delenv(key, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("MINI_AGENT_HOME", str(home))
    monkeypatch.setattr(factory, "PACKAGE_ENV", tmp_path / "package.env")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return tmp_path


# ---- approval: a wrapper of a risky tool is risky too ----

def run_tool(tools, action, args, policy):
    llm = ScriptedLLM([J(action, args), J("done", {"answer": "ok"})])
    Agent(llm=llm, tools=tools, use_memory=False, skills=[], workspace=tempfile.mkdtemp(),
          approve=Approval(policy, ask=lambda a, k: "no")).run("goal", verbose=False)


@pytest.mark.parametrize("policy", ["ask", "deny"])
def test_partial_of_a_risky_tool_stays_risky(policy):
    called = []

    @tool(risky=True)
    def dangerous(target: str):
        called.append(target)

    run_tool([functools.partial(dangerous, target="EXECUTED")], "dangerous", {}, policy)
    assert called == []


@pytest.mark.parametrize("policy", ["ask", "deny"])
def test_decorated_wrapper_of_a_risky_tool_stays_risky(policy):
    called = []

    @tool(risky=True)
    def dangerous(target: str):
        called.append(target)

    @functools.wraps(dangerous)
    def logged(target: str):
        return dangerous(target)

    run_tool([logged], "dangerous", {"target": "x"}, policy)
    assert called == []


def test_agent_has_no_sandbox_option():
    with pytest.raises(TypeError):
        Agent(llm=ScriptedLLM([]), use_memory=False, skills=[], sandbox="docker")


# ---- streamed replies ----

def backend(cls=OpenAICompatLLM, text="", response=None, status=200, key="fake"):
    if response is None:
        handler = lambda req: httpx.Response(status, text=text,
                                             headers={"content-type": "text/event-stream"})
    else:
        handler = lambda req: httpx.Response(status, json=response)
    kwargs = {"api_key": key, "model": "model", "transport": httpx.MockTransport(handler)}
    if cls is not QwenLLM:
        kwargs["base_url"] = "https://fake.invalid/v1"
    return cls(**kwargs)


def event(delta):
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n\n"


@pytest.mark.parametrize("cls", [OpenAICompatLLM, DeepSeekLLM, QwenLLM])
@pytest.mark.parametrize("body", ["<html>gateway unavailable</html>", '{"error":{"message":"bad key"}}', ""])
def test_a_stream_that_is_not_a_stream_is_an_error(cls, body):
    reply = backend(cls, text=body).call_stream("hello", lambda *a, **k: None)
    assert isinstance(reply, LLMError)
    if "bad key" in body:
        assert "bad key" in reply


def test_a_whole_reply_sent_to_a_stream_request_is_accepted():
    whole = json.dumps({"choices": [{"message": {"content": "hello"}}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 1}})
    llm = backend(text=whole)
    seen = []
    assert llm.call_stream("hi", lambda text, kind="content": seen.append(text)) == "hello"
    assert seen == ["hello"] and (llm.total_prompt_tokens, llm.total_completion_tokens) == (3, 1)


def test_one_event_over_several_data_lines():
    text = 'data: {"choices": [\ndata: {"delta": {"content": "hello"}}]}\n\ndata: [DONE]\n\n'
    assert backend(text=text).call_stream("hello", lambda *a, **k: None) == "hello"


def test_events_without_blank_lines_between_them():
    text = ('data: {"choices": [{"delta": {"content": "he"}}]}\n'
            'data: {"choices": [{"delta": {"content": "llo"}}]}\n'
            "data: [DONE]\n")
    assert backend(text=text).call_stream("hello", lambda *a, **k: None) == "hello"


def test_streamed_tool_name_pieces_are_joined():
    text = event({"tool_calls": [{"index": 0, "id": "call_a", "function": {"name": "foo", "arguments": "{"}}]})
    text += event({"tool_calls": [{"index": 0, "function": {"name": "foo", "arguments": "}"}}]})
    text += "data: [DONE]\n\n"
    reply = backend(text=text).chat("p", [], on_chunk=lambda *a, **k: None)
    assert [c.name for c in reply.tool_calls] == ["foofoo"]


def test_a_repeated_tool_call_header_is_not_doubled():
    head = {"index": 0, "id": "call_a", "function": {"name": "calc", "arguments": ""}}
    text = event({"tool_calls": [head]}) + event({"tool_calls": [head]})
    text += event({"tool_calls": [{"index": 0, "function": {"arguments": '{"expr": "1+1"}'}}]})
    text += "data: [DONE]\n\n"
    reply = backend(text=text).chat("p", [], on_chunk=lambda *a, **k: None)
    assert [(c.name, c.args) for c in reply.tool_calls] == [("calc", {"expr": "1+1"})]


@pytest.mark.parametrize("stream,payload", [
    (True, "data: null\n\n"),
    (True, 'data: {"choices":[null]}\n\n'),
    (True, 'data: {"choices":[{"delta": "text"}]}\n\n'),
    (True, 'data: {"choices":[{"delta": {"tool_calls": ["x"]}}]}\n\n'),
    (False, {"choices": [None]}),
    (False, {"choices": [{"message": "not an object"}]}),
    (False, {"choices": "none"}),
])
def test_a_malformed_reply_is_an_error_not_a_crash(stream, payload):
    llm = backend(text=payload) if stream else backend(response=payload)
    reply = llm.call_stream("hello", lambda *a, **k: None) if stream else llm("hello")
    assert isinstance(reply, LLMError)


@pytest.mark.parametrize("stream", [False, True])
def test_a_key_echoed_by_the_server_is_masked(stream):
    def server(request):
        return httpx.Response(401, json={"error": {"message": "Rejected credential " + SECRET}})
    llm = OpenAICompatLLM(api_key=SECRET, model="m", base_url="https://fake.invalid/v1",
                          transport=httpx.MockTransport(server))
    reply = llm.call_stream("hi", lambda *a, **k: None) if stream else llm("hi")
    assert isinstance(reply, LLMError) and SECRET not in reply and mask(SECRET) in reply


def test_redact_keeps_the_type():
    assert isinstance(redact(LLMError("Error: " + SECRET), SECRET), LLMError)
    assert redact("nothing secret", SECRET) == "nothing secret"
    assert redact("short key abc", "abc") == "short key abc"     # too short to be a key


# ---- the order of the settings sources ----

@pytest.mark.parametrize("source", ["environment", "cwd", "user"])
@pytest.mark.parametrize("key", ["LLM_MODEL", "DEEPSEEK_MODEL", "OLLAMA_HOST"])
def test_any_model_setting_hides_the_package_env(isolated, monkeypatch, source, key):
    factory.PACKAGE_ENV.write_text("LLM_BACKEND=offline\nLLM_API_KEY=PACKAGE_SECRET\n")
    if source == "environment":
        monkeypatch.setenv(key, "chosen")
    else:
        path = isolated / "work" / ".env" if source == "cwd" else isolated / "home" / "config.env"
        path.write_text(f"{key}=chosen\n")
    assert "package .env" not in [name for name, _, _ in factory.config_layers()]
    assert "LLM_API_KEY" not in factory.load_config()


@pytest.mark.parametrize("backend_name,key", [
    ("deepseek", "DEEPSEEK_MODEL"), ("qwen", "QWEN_MODEL"),
    ("claude", "CLAUDE_MODEL"), ("ollama", "OLLAMA_MODEL")])
def test_llm_model_from_a_higher_layer_wins(isolated, monkeypatch, backend_name, key):
    (isolated / "home" / "config.env").write_text(f"LLM_BACKEND={backend_name}\n{key}=old-model\n")
    monkeypatch.setenv("LLM_MODEL", "chosen-model")        # what --model does
    assert factory.create_llm().model == "chosen-model"


def test_in_the_same_layer_the_backend_model_wins(isolated):
    (isolated / "home" / "config.env").write_text(
        "LLM_BACKEND=deepseek\nLLM_MODEL=generic\nDEEPSEEK_MODEL=specific\n")
    assert factory.create_llm().model == "specific"


def test_a_backend_model_from_a_higher_layer_wins(isolated, monkeypatch):
    (isolated / "home" / "config.env").write_text("LLM_BACKEND=deepseek\nLLM_MODEL=generic\n")
    monkeypatch.setenv("DEEPSEEK_MODEL", "specific")
    assert factory.create_llm().model == "specific"


@pytest.mark.parametrize("hint", ["anthropic", "claude", "Anthropic"])
def test_llm_provider_anthropic_uses_the_claude_backend(hint):
    llm = factory.create_llm({"LLM_API_KEY": "sk-ambiguous", "LLM_PROVIDER": hint, "LLM_MODEL": "m"})
    assert isinstance(llm, ClaudeLLM) and llm.model == "m"


def test_a_recognised_key_prefix_wins_over_llm_provider():
    llm = factory.create_llm({"LLM_API_KEY": "sk-or-v1-fake", "LLM_PROVIDER": "groq", "LLM_MODEL": "m"})
    assert llm.provider == "openrouter"
    llm = factory.create_llm({"LLM_API_KEY": "sk-ant-fake", "LLM_PROVIDER": "groq", "LLM_MODEL": "m"})
    assert isinstance(llm, ClaudeLLM)


def test_llm_provider_names_an_unrecognised_key():
    llm = factory.create_llm({"LLM_API_KEY": "sk-plain", "LLM_PROVIDER": "deepseek", "LLM_MODEL": "m"})
    assert llm.provider == "deepseek"


def test_llm_base_url_wins_over_key_and_provider():
    llm = factory.create_llm({"LLM_BASE_URL": "http://proxy.invalid/v1", "LLM_API_KEY": "sk-or-v1-x",
                              "LLM_PROVIDER": "groq", "LLM_MODEL": "m"})
    assert llm.base_url == "http://proxy.invalid/v1"


def test_a_damaged_settings_file_does_not_stop_the_program(isolated):
    (isolated / "home" / "config.env").write_bytes(b"LLM_BACKEND=offline\nLLM_MODEL=\xff\n")
    assert factory.load_config()["LLM_BACKEND"] == "offline"


# ---- setup and doctor ----

def key_rejected_by_server(monkeypatch):
    real = httpx.Client

    def server(request):     # a gateway that repeats the credential in its error message
        return httpx.Response(401, json={"error": {"message": "Rejected credential " + SECRET}})
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: real(*a, **dict(kw, transport=httpx.MockTransport(server))))


def setup_with_key(monkeypatch, *extra):
    monkeypatch.setattr(sys, "stdin", io.StringIO(SECRET + "\n"))
    return setup.main(["--provider", "compat", "--base-url", "https://fake.invalid/v1",
                       "--model", "model", "--key-stdin", *extra])


def test_setup_never_prints_the_key(isolated, monkeypatch, capsys):
    key_rejected_by_server(monkeypatch)
    assert setup_with_key(monkeypatch) == 1
    out = capsys.readouterr()
    assert SECRET not in out.out + out.err and "Rejected credential" in out.out


def test_doctor_never_prints_the_key(isolated, monkeypatch, capsys):
    key_rejected_by_server(monkeypatch)
    assert setup_with_key(monkeypatch, "--no-test") == 0
    capsys.readouterr()
    doctor.main(["--test"])
    out = capsys.readouterr()
    assert SECRET not in out.out + out.err and "Rejected credential" in out.out


def test_setup_keeps_no_backup_copy(isolated, monkeypatch):
    setup_with_key(monkeypatch, "--no-test")
    assert setup.main(["--provider", "offline", "--no-test"]) == 0
    home = isolated / "home"
    assert sorted(p.name for p in home.iterdir()) == ["config.env"]
    assert SECRET not in (home / "config.env").read_text()


def test_setup_replaces_a_symlink_instead_of_writing_through_it(isolated, monkeypatch):
    other = isolated / "unrelated-file"
    other.write_text("ORIGINAL")
    link = isolated / "home" / "config.env"
    link.symlink_to(other)
    assert setup_with_key(monkeypatch, "--no-test") == 0
    assert other.read_text() == "ORIGINAL"
    assert not link.is_symlink() and SECRET in link.read_text()
    if os.name != "nt":
        assert (link.stat().st_mode & 0o777) == 0o600


def test_setup_with_a_directory_in_the_way(isolated, capsys):
    (isolated / "home" / "config.env").mkdir()
    assert setup.main(["--provider", "offline"]) == 1
    err = capsys.readouterr().err
    assert "cannot save the settings" in err and "is a directory" in err
    assert sorted(p.name for p in (isolated / "home").iterdir()) == ["config.env"]   # no temp file left


def test_doctor_reports_a_damaged_settings_file(isolated, capsys):
    (isolated / "home" / "config.env").write_bytes(b"LLM_MODEL=\xff\n")
    assert doctor.main([]) == 1
    assert "not valid UTF-8" in capsys.readouterr().out


def test_setup_stops_cleanly_when_the_terminal_closes(isolated, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", type("Terminal", (), {"isatty": lambda self: True})())

    def closed(prompt=""):
        raise EOFError
    monkeypatch.setattr("builtins.input", closed)
    assert setup.main([]) == 1
    assert "nothing was saved" in capsys.readouterr().err
    assert not (isolated / "home" / "config.env").exists()


# ---- argument schemas and docstrings ----

def test_optional_allows_null():
    @tool
    def f(x: Optional[int] = None):
        return x
    assert f.__tool__.parameters["properties"]["x"] == {
        "anyOf": [{"type": "integer"}, {"type": "null"}], "default": None}


def test_a_none_default_never_contradicts_the_type():
    @tool
    def f(x: int = None):
        return x
    schema = f.__tool__.parameters["properties"]["x"]
    # Python 3.10 reads "x: int = None" as Optional[int] (null allowed, default kept);
    # 3.11 and later read it as int (no null, so the default is left out)
    assert schema in ({"type": "integer"},
                      {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None})
    assert f.__tool__.parameters["required"] == []


def test_fixed_tuple_and_typed_dict():
    @tool
    def f(pair: tuple[int, str], many: tuple[int, ...], values: dict[str, int], anything: dict):
        return pair
    p = f.__tool__.parameters["properties"]
    assert p["pair"] == {"type": "array", "prefixItems": [{"type": "integer"}, {"type": "string"}],
                         "minItems": 2, "maxItems": 2}
    assert p["many"] == {"type": "array", "items": {"type": "integer"}}
    assert p["values"] == {"type": "object", "additionalProperties": {"type": "integer"}}
    assert p["anything"] == {"type": "object"}


def test_sphinx_continuation_lines():
    @tool
    def f(x: str, y: int):
        """A tool.

        :param x: first line
            essential continuation
        :type x: str
        :param y: the y
        :returns: nothing
        """
    p = f.__tool__.parameters["properties"]
    assert p["x"]["description"] == "first line essential continuation"
    assert p["y"]["description"] == "the y"


def test_numpy_grouped_parameters():
    @tool
    def f(x: int, y: int, z: int):
        """A tool.

        Parameters
        ----------
        x, y : int
            Shared description.
        z : int
            Its own.
        """
    p = f.__tool__.parameters["properties"]
    assert [p[k].get("description") for k in "xyz"] == ["Shared description.", "Shared description.", "Its own."]


def test_positional_only_arguments_can_be_named():
    @tool
    def f(x: int, y: int = 2, /, *, z: int = 3):
        return f"{x} {y} {z}"
    funcs = ToolSet.build([f]).funcs
    assert execute("f", {"x": 7}, funcs) == "7 2 3"
    assert execute("f", {"x": 7, "y": 8, "z": 9}, funcs) == "7 8 9"
    assert f(1, 2, z=3) == "1 2 3"                  # the function itself is unchanged


def test_a_builtin_without_a_signature_still_works_in_a_dict():
    table = ToolSet.build({"clock": time.time}, "- clock(): the current time")
    assert not execute("clock", {}, table.funcs).startswith("Error:")
    assert table.desc == "- clock(): the current time"
