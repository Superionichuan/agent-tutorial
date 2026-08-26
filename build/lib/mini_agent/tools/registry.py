"""Tool registry"""
from .math import calc
from .file import read_file, write_file, list_dir
from .shell import bash, run_python
from .memory import remember, log
from .todo import todo_write

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
