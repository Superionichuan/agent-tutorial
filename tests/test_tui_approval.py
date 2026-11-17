"""The full-screen interface asks in a dialog while the agent waits on its worker thread."""
import asyncio
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("textual")


def test_dialog_answers_reach_the_worker(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_BACKEND", "offline")
    monkeypatch.setenv("MINI_AGENT_HOME", str(tmp_path / ".mini_agent"))
    from mini_agent.core.policy import Approval
    from mini_agent.ui import tui
    monkeypatch.setattr(tui, "SESS_DIR", tmp_path / "sessions")

    async def scenario():
        app = tui.MiniAgentApp(agent_options={"approve": "ask"})
        async with app.run_test() as pilot:
            assert isinstance(app.agent.approve, Approval)
            for key, expected in (("y", "yes"), ("n", "no"), ("a", "always")):
                got = {}
                worker = threading.Thread(
                    target=lambda: got.setdefault("answer", app._ask_approval("bash", {"cmd": "ls -la"})))
                worker.start()
                for _ in range(100):
                    await pilot.pause(0.02)
                    if isinstance(app.screen, tui.ApproveScreen):
                        break
                assert isinstance(app.screen, tui.ApproveScreen)
                assert app.screen.summary == "ls -la"
                await pilot.press(key)
                for _ in range(100):
                    await pilot.pause(0.02)
                    if "answer" in got:
                        break
                worker.join(5)
                assert got.get("answer") == expected
                assert not isinstance(app.screen, tui.ApproveScreen)

    asyncio.run(scenario())
