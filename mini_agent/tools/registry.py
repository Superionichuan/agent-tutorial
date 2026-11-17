"""Tool registry

TOOLS (name -> function) and TOOL_DESC (the [Tools] text) are the original
form and keep working. BUILTIN_SPECS holds the same tools as ToolSpecs, with
the argument schemas needed for native tool calling; rendered as text they
give exactly TOOL_DESC.
"""
from .math import calc
from .file import read_file, write_file, list_dir
from .shell import bash, run_python
from .memory import remember, log
from .todo import todo_write
from .skill import read_skill
from .spec import ToolSpec, spec_from_function

TOOLS = {
    "calc": calc,
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "bash": bash,
    "run_python": run_python,
    "remember": remember,
    "todo_write": todo_write,
    "log": log,
    "read_skill": read_skill,  # described in the skills section, not in TOOL_DESC
}

TOOL_DESC = """
- calc(expr): evaluate a math expression
- read_file(path): read a file
- write_file(path, content): write a file
- list_dir(path): list a directory
- bash(cmd): run a bash command
- run_python(code): run Python code
- todo_write(todos): update the task list (whole-list replace). todos=[{content, status}], status: pending|in_progress|completed. For multi-step tasks write the plan first, update statuses as you go; at most one in_progress at a time
- remember(content): save an important lesson to long-term memory
- log(content): append a note to today's log
"""

# Tools that change files or run code: the "ask" approval policy confirms them.
RISKY = frozenset({"bash", "run_python", "write_file"})

_TODO_ITEMS = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
        },
        "required": ["content", "status"],
    },
}


def _builtin(fn, description: str, listed: bool = True) -> ToolSpec:
    spec = spec_from_function(fn, description=description, risky=fn.__name__ in RISKY,
                              listed=listed)
    if fn is todo_write:
        spec.parameters["properties"]["todos"] = _TODO_ITEMS
    return spec


# Same order as TOOL_DESC, so that rendering the listed specs reproduces it.
BUILTIN_SPECS = {s.name: s for s in [
    _builtin(calc, "evaluate a math expression"),
    _builtin(read_file, "read a file"),
    _builtin(write_file, "write a file"),
    _builtin(list_dir, "list a directory"),
    _builtin(bash, "run a bash command"),
    _builtin(run_python, "run Python code"),
    _builtin(todo_write, "update the task list (whole-list replace). todos=[{content, status}], "
                         "status: pending|in_progress|completed. For multi-step tasks write the plan "
                         "first, update statuses as you go; at most one in_progress at a time"),
    _builtin(remember, "save an important lesson to long-term memory"),
    _builtin(log, "append a note to today's log"),
    _builtin(read_skill, "read the full instructions of a skill listed in the skills section",
             listed=False),
]}


def default_tools() -> list:
    """The built-in tools as a list of ToolSpecs, e.g. Agent(tools=default_tools() + [my_tool])."""
    return list(BUILTIN_SPECS.values())
