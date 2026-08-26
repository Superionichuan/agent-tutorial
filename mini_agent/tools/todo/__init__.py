"""Todo tool — task list (whole-list-replace semantics)

Design notes:
- todo_write(todos) always sends the complete list, no partial updates — snapshot semantics replay correctly by construction
- single-in_progress discipline (focus on one thing at a time)
- strict validation: empty/duplicate content, bad status, extra keys all rejected — what lands in the log matches what the model thinks it wrote
- the list itself goes into the session log (todo/write event emitted by the run loop); UIs render from events
"""

VALID_STATUS = ("pending", "in_progress", "completed")


def todo_write(todos: list) -> str:
    """Update the task list (whole-list replace). todos: [{content, status}], status in pending|in_progress|completed"""
    if not isinstance(todos, list) or not todos:
        return "Error: todos must be a non-empty list"
    seen = set()
    in_progress = 0
    for t in todos:
        if not isinstance(t, dict):
            return "Error: each item must be {content, status}"
        extra = set(t) - {"content", "status"}
        if extra:
            return f"Error: unexpected keys {sorted(extra)}"
        c = str(t.get("content", "")).strip()
        if not c:
            return "Error: content must not be empty"
        if c in seen:
            return f"Error: duplicate content: {c}"
        seen.add(c)
        if t.get("status") not in VALID_STATUS:
            return "Error: invalid status: " + str(t.get("status"))
        in_progress += t["status"] == "in_progress"
    if in_progress > 1:
        return f"Error: at most one task may be in_progress (got {in_progress})"
    n = {s: sum(1 for t in todos if t["status"] == s) for s in VALID_STATUS}
    return ("todo updated: " + str(n["completed"]) + " completed / " + str(n["in_progress"]) + " in progress / " + str(n["pending"]) + " pending")
