"""stdin_is_terminal: a question is asked only where a person can answer it."""
import os
import sys

import pytest

from mini_agent.terminal import stdin_is_terminal


def test_the_null_device_is_not_a_terminal():
    # on Windows NUL.isatty() is True; it is still not somewhere a person can answer
    with open(os.devnull, encoding="utf-8") as null:
        assert stdin_is_terminal(null) is False


def test_no_stdin_is_not_a_terminal(monkeypatch):
    monkeypatch.setattr(sys, "stdin", None)
    assert stdin_is_terminal() is False


def test_a_closed_stream_is_not_a_terminal():
    stream = open(os.devnull, encoding="utf-8")
    stream.close()
    assert stdin_is_terminal(stream) is False


@pytest.mark.skipif(os.name == "nt", reason="pseudo-terminals are POSIX")
def test_a_pseudo_terminal_is_a_terminal():
    leader, follower = os.openpty()
    try:
        with os.fdopen(follower, closefd=False, encoding="utf-8") as tty:
            assert stdin_is_terminal(tty) is True
    finally:
        os.close(leader)
        os.close(follower)
