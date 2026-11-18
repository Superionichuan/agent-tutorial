"""A reply whose JSON is broken gets a specific error back, and one more try.

Modelled on two real replies (step 45 and 46 of a long task): each put a whole Python program
inside a bash "cmd" string, three levels of quoting deep. One ended `...'"}]}` (a stray `]`),
the other closed the cmd string early with an unescaped `"` before ` 2>&1'}}`. The generic
"no JSON object" message did not tell the model what was wrong, and the task stopped after two
such replies. The content here is neutral; the two breakages are the same.
"""
import json

from mini_agent.core.parser import parse_response
from mini_agent.core.parser.parse import CODE_HINT
from test_harness import J, end, results, run

NO_JSON = 'the reply contains no JSON object with a string "action" field'

PROGRAM = "\n".join(["import json", "", "results = []"]
                    + [f"results.append({{'case': {i}, 'ok': {i} % 3 != 0}})" for i in range(40)]
                    + ["print(json.dumps(results))"])
BODY = json.dumps(PROGRAM)[1:-1]          # the program as it appears inside a JSON string

STRAY_BRACKET = ('{"thought": "Write the check script, then run it.", "action": "bash", '
                 '"args": {"cmd": "cat > run_checks.py << \'PYEOF\'\\n' + BODY
                 + '\\nPYEOF\\necho \'Written run_checks.py\'"}]}')
BARE_QUOTE = ('{"thought": "Run the checks with the project\'s Python.", "action": "bash", '
              '"args": {"cmd": "cd project && python3 -c \\"\\n' + BODY + '\\n" 2>&1\'}}')


def error_position(reply: str) -> int:
    try:
        json.loads(reply)
    except json.JSONDecodeError as err:
        return err.pos
    raise AssertionError("the reply parses")


def test_the_two_broken_replies_get_the_json_error_and_where():
    for reply, near in ((STRAY_BRACKET, '"}]}'), (BARE_QUOTE, "2>&1'}}")):
        thought, action, args = parse_response(reply)
        assert action is None and args["json_syntax"] is True
        error = args["error"]
        assert error.startswith("the reply's JSON does not parse: Expecting ',' delimiter")
        assert f"at character {error_position(reply)}," in error
        assert near in error
        assert error.endswith(CODE_HINT)          # long code inside a cmd string
        assert error.endswith("Put code in write_file's content (its own argument), then run "
                              "the file; don't put multi-line programs inside bash quotes. "
                              'Inside a JSON string, write \\" for a double quote and \\n for '
                              "a newline")
        assert error != NO_JSON
        assert thought == reply


def test_a_reply_without_json_keeps_the_old_error_exactly():
    # The Tutorial 1 notebook prints this dict for its "no JSON" example.
    assert parse_response("The product is 817,261.") == (
        "The product is 817,261.", None, {"error": NO_JSON})


def test_a_short_broken_reply_gets_no_code_hint():
    reply = '{"thought": "t", "action": "calc", "args": {"expr": "1+1"}]}'
    _, action, args = parse_response(reply)
    assert action is None and args["json_syntax"] is True
    assert f"at character {error_position(reply)}," in args["error"]
    assert CODE_HINT not in args["error"]


def test_a_valid_object_after_a_broken_one_is_still_used():
    reply = STRAY_BRACKET + "\n" + J("calc", {"expr": "947*863"})
    assert parse_response(reply)[1:] == ("calc", {"expr": "947*863"})


def test_two_broken_replies_then_a_valid_one_continue_the_task():
    answer, ev, llm = run([STRAY_BRACKET, BARE_QUOTE, J("done", {"answer": "checked"})])
    assert answer == "checked" and end(ev)["reason"] == "done" and llm.calls == 3
    observed = results(ev)
    assert len(observed) == 2
    for text in observed:
        assert text.startswith("Error: the reply's JSON does not parse: Expecting ',' delimiter")
        assert CODE_HINT in text and text.endswith(". Reply with exactly one JSON object.")
        assert NO_JSON not in text


def test_three_broken_replies_stop():
    answer, ev, llm = run([STRAY_BRACKET, BARE_QUOTE, STRAY_BRACKET, J("done", {"answer": "x"})])
    assert end(ev)["reason"] == "invalid_reply" and llm.calls == 3
    assert answer.startswith("Stopped after 3 invalid replies. Last reply: ")


def test_replies_without_json_still_stop_after_two():
    answer, ev, llm = run(["No JSON here.", "Still none.", J("done", {"answer": "x"})])
    assert end(ev)["reason"] == "invalid_reply" and llm.calls == 2
    assert answer == "Stopped after 2 invalid replies. Last reply: Still none."
    assert results(ev) == [f"Error: {NO_JSON}. Reply with exactly one JSON object."]


def test_a_reply_without_json_after_a_broken_one_stops_at_two():
    _, ev, llm = run([STRAY_BRACKET, "No JSON here.", J("done", {"answer": "x"})])
    assert end(ev)["reason"] == "invalid_reply" and llm.calls == 2
