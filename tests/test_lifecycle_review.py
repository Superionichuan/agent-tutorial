"""Lifecycle: six failures found in an independent code review, kept as regression tests.

Each test failed on the code before the fixes: a user message containing a line like a
user.md header, a large result among the last steps dropped by format_history's cap, a
.mini_agent symlink leading outside the workspace, a continued turn deciding differently
than the run that was cut off (compaction state and handoff lists), and a summary that
used up the budget followed by another model call.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mini_agent.core.agent.run import run_agent  # noqa: E402
from mini_agent.core.state import State  # noqa: E402
from mini_agent.lifecycle import Lifecycle, restore_steps  # noqa: E402
from mini_agent.lifecycle import compact as C  # noqa: E402
from mini_agent.lifecycle import notes as N  # noqa: E402
from mini_agent.session import SessionLog  # noqa: E402


def test_user_message_with_entry_header_is_shown_verbatim(tmp_path):
    message = 'Keep this quotation:\n## 7 · 2026-11-17 12:00:00\n猫 stays here.'
    lifecycle = Lifecycle(tmp_path)
    prompts = []

    def model(prompt):
        prompts.append(prompt)
        return '{"action":"done","args":{"answer":"ok"}}'

    for goal in (message, 'What was my earlier message?'):
        run_agent(model, {}, '', goal, lifecycle=lifecycle, verbose=False)
    assert message in prompts[-1], 'earlier user message was split at its literal Markdown header'
    entries = N.user_messages(tmp_path / '.mini_agent' / 'user.md')
    assert [text for _, _, text in entries] == [message, 'What was my earlier message?']


def test_protected_recent_result_is_in_next_prompt(tmp_path):
    result = 'RECENT_BEGIN\n' + 'x' * 160001 + '\nRECENT_END'
    prompts = []

    def model(prompt):
        prompts.append(prompt)
        action = 'large_result' if len(prompts) == 1 else 'done'
        return json.dumps({'action': action, 'args': {}})

    model.context_limit = 1000000
    lifecycle = Lifecycle(tmp_path, llm=model, keep_steps=1)
    run_agent(model, {'large_result': lambda: result}, '', 'Read the result',
              lifecycle=lifecycle, verbose=False)
    assert lifecycle.state.steps[-1]['result'] == result
    assert result in prompts[-1], 'protected recent tool result vanished from the prompt'


def test_lifecycle_does_not_follow_workspace_directory_symlink(tmp_path):
    workspace, outside = tmp_path / 'workspace', tmp_path / 'outside'
    workspace.mkdir()
    outside.mkdir()
    try:
        (workspace / '.mini_agent').symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip('symbolic links are not available here')
    try:
        run_agent(lambda prompt: '{"action":"done","args":{}}', {}, '', 'private goal',
                  lifecycle=Lifecycle(workspace), verbose=False)
    except OSError:
        pass  # refusing an unsafe directory is acceptable
    assert not list(outside.iterdir()), 'lifecycle wrote through a symlink outside the workspace'


def test_first_restored_prompt_matches_uninterrupted_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(N, 'now', lambda: '2026-11-17 12:00:00')
    directory = tmp_path
    events, prompts, checkpoints = [], [], []

    def emit(kind, **data):
        events.append({'type': kind, 'data': data})

    def model(prompt):
        if prompt.startswith(C.SUMMARY_PROMPT):
            return 'Earlier work complete.'
        prompts.append(prompt)
        checkpoints.append(json.loads(json.dumps(events)))
        action = 'work' if len(prompts) < 4 else 'done'
        return json.dumps({'thought': 'continue', 'action': action, 'args': {}})

    def lifecycle(llm):
        return Lifecycle(directory, llm=llm, compact_at=300, compact_to=100, keep_steps=1)

    # A large fixed prompt means compaction cannot reach its target.
    goal = 'g' * 3000
    lc = lifecycle(model)
    saved_handoffs = []
    original_model = model

    def capture(prompt):
        if not prompt.startswith(C.SUMMARY_PROMPT):
            path = Path(directory) / '.mini_agent' / 'handoff.md'
            saved_handoffs.append(path.read_text(encoding='utf-8') if path.exists() else None)
        return original_model(prompt)

    lc.llm = capture
    run_agent(capture, {'work': lambda: 'work done'}, '', goal,
              lifecycle=lc, on_event=emit, verbose=False)
    # Restore the files and log prefix as they stood at the final model call.
    path = Path(directory) / '.mini_agent' / 'handoff.md'
    path.write_text(saved_handoffs[-1], encoding='utf-8')
    restored = restore_steps(checkpoints[-1])
    actual = []

    def resumed_model(prompt):
        if prompt.startswith(C.SUMMARY_PROMPT):
            return 'Earlier work complete.'
        actual.append(prompt)
        return '{"action":"done","args":{}}'

    run_agent(resumed_model, {'work': lambda: 'work done'}, '', goal,
              lifecycle=lifecycle(resumed_model), state=restored, verbose=False)
    assert actual[0] == prompts[-1], 'restore changed the first prompt despite identical files and log'


def test_resume_does_not_erase_compacted_plan_and_files(tmp_path, monkeypatch):
    monkeypatch.setattr(N, 'now', lambda: '2026-11-17 12:00:00')
    directory = tmp_path
    log = SessionLog(Path(directory) / 'session.jsonl')
    goal = 'Complete the release'
    steps = [
        dict(thought='write', action='write_file', args={'path': 'release.txt', 'content': 'release contents'}, result='ok'),
        dict(thought='plan', action='todo_write', args={'todos': [
            {'content': 'Verify the release signature', 'status': 'pending'}]}, result='ok'),
        dict(thought='continue', action='work', args={}, result='ok'),
    ]
    state = State(goal=goal, steps=steps)
    lc = Lifecycle(directory, compact_at=300, compact_to=100, keep_steps=1)
    lc.attach(log.append, state)
    lc.start(goal)
    log.append('turn/start', goal=goal)
    for number, step in enumerate(steps, 1):
        log.append('step/think', step=number, thought=step['thought'],
                   action=step['action'], args=step['args'])
        log.append('tool/call', step=number, name=step['action'],
                   args=step['args'], call_id=str(number))
        log.append('tool/result', step=number, content=step['result'], call_id=str(number))
    lc.before_think(4, lambda memory: memory + 'x' * 3000 + str(state.steps))
    handoff = Path(directory) / '.mini_agent' / 'handoff.md'
    before = handoff.read_text(encoding='utf-8')
    markers = ['written: release.txt', 'Verify the release signature']
    assert all(marker in before for marker in markers)
    assert state.steps[0].get('compacted')
    restored = restore_steps(log.unfinished_turn()['events'])
    # No new work: a continued segment immediately hits its step limit.
    run_agent(lambda prompt: '', {}, '', goal, max_steps=0, state=restored,
              lifecycle=Lifecycle(directory), verbose=False)
    after = handoff.read_text(encoding='utf-8')
    missing = [marker for marker in markers if marker not in after]
    assert not missing, f'resume erased handoff details: {missing}'


def test_summary_exhausting_budget_stops_before_another_model_call(tmp_path):
    class Model:
        context_limit = 100000
        total_prompt_tokens = 0
        total_completion_tokens = 0

        def __init__(self):
            self.calls = []

        def __call__(self, prompt):
            if prompt.startswith(C.SUMMARY_PROMPT):
                self.calls.append('summary')
                self.total_prompt_tokens += 1000
                return 'Work completed.'
            self.calls.append('agent')
            self.total_prompt_tokens += 1
            return json.dumps({'action': 'work', 'args': {}})

    model = Model()
    events = []
    run_agent(model, {'work': lambda: 'x' * 3000}, '', 'work',
              max_tokens=100, max_steps=10, verbose=False,
              lifecycle=Lifecycle(tmp_path, llm=model, compact_at=300,
                                  compact_to=100, keep_steps=1),
              on_event=lambda kind, **data: events.append((kind, data)))
    assert events[-1][1]['reason'] == 'token_limit'
    assert 'summary' in model.calls
    assert model.calls[-1] == 'summary', (
        f'another model call was sent after the summary exhausted the budget: {model.calls}')
