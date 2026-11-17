"""Tool specifications: a function plus what the model is told about it.

A tool is a plain Python function. The model sees its name, a description,
and the JSON schema of its arguments. All three are derived from the
function's signature, type hints and docstring:

    from mini_agent import Agent, tool

    @tool
    def density(mass_g: float, volume_cm3: float) -> float:
        \"\"\"Density in g/cm^3.

        Args:
            mass_g: mass in grams
            volume_cm3: volume in cubic centimetres
        \"\"\"
        return mass_g / volume_cm3

    Agent(tools=[density]).run("A 7.874 g cube has a volume of 1 cm^3. Density?")

`@tool` returns the function itself, still callable as before, with its
ToolSpec attached as `fn.__tool__`. Undecorated functions work too: the spec
is derived when the agent is built. The old form, a dict of functions plus a
description string (TOOLS, TOOL_DESC), keeps working unchanged.
"""
from __future__ import annotations

import collections.abc
import enum
import functools
import inspect
import re
import types
import typing
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

_SIMPLE = {str: "string", int: "integer", float: "number", bool: "boolean"}
_SECTION = re.compile(r"^(Args|Arguments|Parameters|Params|Keyword Args|Returns?|Yields?|Raises|"
                      r"Examples?|Notes?|See Also|Attributes)\s*:?\s*$", re.I)
_GOOGLE_ARG = re.compile(r"^\*{0,2}(\w+)\s*(?:\(([^)]*)\))?\s*:\s*(.*)$")
_SPHINX_ARG = re.compile(r"^:param\s+(?:[\w\[\], |.]+\s+)?(\w+)\s*:\s*(.*)$")
_NUMPY_ARG = re.compile(r"^(\w+)\s*:\s*(.*)$")


@dataclass
class ToolSpec:
    """What the agent knows about one tool.

    name, description, parameters (a JSON schema object): what the model sees.
    fn: the function that runs. risky: needs approval under the "ask" policy.
    listed: shown in the [Tools] section of the prompt (read_skill is not:
    the skills section describes it).
    """
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]
    risky: bool = False
    listed: bool = True

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    @property
    def arg_names(self) -> List[str]:
        return list(self.parameters.get("properties", {}))

    def describe(self) -> str:
        """One line for the [Tools] section: - name(arg, ...): description"""
        return f"- {self.name}({', '.join(self.arg_names)}): {self.description}"

    def openai_schema(self) -> dict:
        """The tool in the OpenAI chat-completions `tools` format."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}

    def anthropic_schema(self) -> dict:
        """The tool in the Anthropic messages `tools` format."""
        return {"name": self.name, "description": self.description,
                "input_schema": self.parameters}


def _json_type(tp) -> dict:
    """JSON schema for a Python type hint (unknown types: any JSON value)."""
    if tp is Any:
        return {}
    if tp is type(None):
        return {"type": "null"}
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is typing.Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        options = [a for a in args if a is not type(None)]
        if len(options) == 1:
            return _json_type(options[0])
        return {"anyOf": [_json_type(a) for a in options]}
    if origin is typing.Literal:
        values = list(args)
        schema: dict = {"enum": values}
        kinds = {type(v) for v in values}
        if len(kinds) == 1 and next(iter(kinds)) in _SIMPLE:
            schema["type"] = _SIMPLE[next(iter(kinds))]
        return schema
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        values = [m.value for m in tp]
        schema = {"enum": values}
        kinds = {type(v) for v in values}
        if len(kinds) == 1 and next(iter(kinds)) in _SIMPLE:
            schema["type"] = _SIMPLE[next(iter(kinds))]
        return schema
    if tp in _SIMPLE:
        return {"type": _SIMPLE[tp]}
    sequence = (list, tuple, set, frozenset, collections.abc.Sequence, collections.abc.Iterable,
                collections.abc.Set)
    if tp in sequence or origin in sequence:
        schema = {"type": "array"}
        items = [a for a in args if a is not Ellipsis]
        if len(set(items)) == 1:
            schema["items"] = _json_type(items[0])
        elif items:
            schema["items"] = {"anyOf": [_json_type(a) for a in dict.fromkeys(items)]}
        return schema
    if tp in (dict, collections.abc.Mapping) or origin in (dict, collections.abc.Mapping):
        return {"type": "object"}
    return {}


def _parse_docstring(doc: Optional[str]):
    """(summary, {argument: description}) from a Google, Sphinx or NumPy docstring."""
    lines = inspect.cleandoc(doc or "").splitlines()
    summary: List[str] = []
    i = 0
    while i < len(lines) and lines[i].strip() and not _SECTION.match(lines[i].strip()) \
            and not lines[i].strip().startswith(":"):
        summary.append(lines[i].strip())
        i += 1
    args: Dict[str, str] = {}
    section, current, base_indent = None, None, 0
    for raw in lines:
        line = raw.strip()
        if not line:
            current = None if section != "numpy" else current
            continue
        if set(line) == {"-"} and section == "numpy-header":
            section = "numpy"
            continue
        head = _SECTION.match(line)
        if head:
            word = head.group(1).lower()
            is_args = word in ("args", "arguments", "parameters", "params", "keyword args")
            if not line.endswith(":") and is_args:      # NumPy: "Parameters" then a dashed line
                section = "numpy-header"
            else:
                section = "google" if is_args else None
            current = None
            continue
        indent = len(raw) - len(raw.lstrip())
        sphinx = _SPHINX_ARG.match(line)
        if sphinx:
            current, base_indent = sphinx.group(1), indent
            args[current] = sphinx.group(2).strip()
            continue
        if section == "google":
            m = _GOOGLE_ARG.match(line)
            if m and (current is None or indent <= base_indent):
                current, base_indent = m.group(1), indent
                args[current] = m.group(3).strip()
            elif current:
                args[current] = (args[current] + " " + line).strip()
        elif section == "numpy":
            m = _NUMPY_ARG.match(line)
            if m and (current is None or indent <= base_indent):
                current, base_indent = m.group(1), indent
                args[current] = ""
            elif current:
                args[current] = (args[current] + " " + line).strip()
    return " ".join(summary), args


def spec_from_function(fn: Callable, name: Optional[str] = None,
                       description: Optional[str] = None, risky: bool = False,
                       listed: bool = True) -> ToolSpec:
    """Derive a ToolSpec from a function's signature, type hints and docstring."""
    target = fn.func if isinstance(fn, functools.partial) else fn
    tool_name = name or getattr(target, "__name__", None)
    if not tool_name or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", tool_name):
        raise ValueError(f"cannot use {tool_name!r} as a tool name: give name=... "
                         "(letters, digits, _ or -, at most 64 characters)")
    summary, arg_docs = _parse_docstring(inspect.getdoc(target))
    try:
        hints = typing.get_type_hints(target)
    except Exception:            # unresolvable forward references: fall back to no hints
        hints = {}
    properties: Dict[str, dict] = {}
    required: List[str] = []
    extra = False
    for p in inspect.signature(fn).parameters.values():
        if p.kind is p.VAR_POSITIONAL:
            continue
        if p.kind is p.VAR_KEYWORD:
            extra = True
            continue
        hint = hints.get(p.name, inspect.Parameter.empty)
        if hint is not inspect.Parameter.empty:
            prop = _json_type(hint)
        elif type(p.default) in _SIMPLE:          # no hint: the type of the default, else text
            prop = {"type": _SIMPLE[type(p.default)]}
        else:
            prop = {"type": "string"}
        if arg_docs.get(p.name):
            prop["description"] = arg_docs[p.name]
        if p.default is inspect.Parameter.empty:
            required.append(p.name)
        elif p.default is None or isinstance(p.default, (str, int, float, bool)):
            prop["default"] = p.default
        properties[p.name] = prop
    parameters = {"type": "object", "properties": properties, "required": required}
    if not extra:
        parameters["additionalProperties"] = False
    return ToolSpec(name=tool_name, description=description or summary or tool_name,
                    parameters=parameters, fn=fn, risky=risky, listed=listed)


def tool(fn: Optional[Callable] = None, *, name: Optional[str] = None,
         description: Optional[str] = None, risky: bool = False):
    """Mark a function as a tool: @tool, or @tool(name=..., description=..., risky=True).

    risky=True: under the "ask" approval policy the user confirms each call.
    Returns the function itself with its ToolSpec as fn.__tool__ (or the
    ToolSpec, for callables that take no attributes).
    """
    def wrap(f: Callable):
        spec = spec_from_function(f, name=name, description=description, risky=risky)
        try:
            f.__tool__ = spec
            return f
        except (AttributeError, TypeError):
            return spec
    return wrap(fn) if fn is not None else wrap


def as_spec(obj: Any, name: Optional[str] = None) -> ToolSpec:
    """A ToolSpec for a ToolSpec, a decorated function, or a plain function."""
    if isinstance(obj, ToolSpec):
        return obj if name in (None, obj.name) else ToolSpec(
            name, obj.description, obj.parameters, obj.fn, obj.risky, obj.listed)
    spec = getattr(obj, "__tool__", None)
    if isinstance(spec, ToolSpec):
        return spec if name in (None, spec.name) else ToolSpec(
            name, spec.description, spec.parameters, spec.fn, spec.risky, spec.listed)
    if callable(obj):
        return spec_from_function(obj, name=name)
    raise TypeError(f"not a tool: {obj!r} (expected a function or a ToolSpec)")


class ToolSet:
    """The tools of one agent: name -> function, the [Tools] text, and the specs.

    Accepts a list of functions or ToolSpecs, or the old dict of name ->
    function (optionally with its own description string).
    """

    def __init__(self, specs: Dict[str, ToolSpec], desc: str):
        self.specs = specs
        self.desc = desc

    @property
    def funcs(self) -> Dict[str, Callable]:
        return {name: spec.fn for name, spec in self.specs.items()}

    @classmethod
    def build(cls, tools: Any = None, tool_desc: Optional[str] = None) -> "ToolSet":
        from .registry import BUILTIN_SPECS, TOOL_DESC, TOOLS
        if tools is None or tools is TOOLS:
            return cls(dict(BUILTIN_SPECS), TOOL_DESC if tool_desc is None else tool_desc)
        if isinstance(tools, ToolSet):
            return tools if tool_desc is None else cls(tools.specs, tool_desc)
        specs: Dict[str, ToolSpec] = {}
        if isinstance(tools, collections.abc.Mapping):
            for key, fn in tools.items():
                builtin = BUILTIN_SPECS.get(key)
                specs[key] = builtin if builtin is not None and builtin.fn is fn else as_spec(fn, name=key)
        elif isinstance(tools, Iterable) and not isinstance(tools, (str, bytes)):
            for item in tools:
                spec = as_spec(item)
                if spec.name in specs:
                    raise ValueError(f"two tools are named {spec.name!r}")
                specs[spec.name] = spec
        else:
            raise TypeError("tools must be a list of functions, or a dict of name -> function")
        if tool_desc is None:
            tool_desc = render_tool_desc(specs.values())
        return cls(specs, tool_desc)

    def __contains__(self, name: str) -> bool:
        return name in self.specs

    def __iter__(self):
        return iter(self.specs.values())

    def __len__(self) -> int:
        return len(self.specs)


def render_tool_desc(specs: Iterable[ToolSpec]) -> str:
    """The [Tools] text: one line per listed tool, framed like TOOL_DESC."""
    lines = [s.describe() for s in specs if s.listed]
    return "\n" + "\n".join(lines) + "\n"
