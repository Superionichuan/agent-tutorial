"""The agent loop, event-emitting version

Every node in the loop emits an event via on_event(type, **data) instead of
printing directly. The default (no on_event, verbose=True) uses the built-in
console renderer whose output matches the plain version exactly (fully
backward compatible).

Event protocol (same taxonomy as mini_agent.session):
    turn/start   {goal, name}
    step/start   {step}
    step/think   {step, thought, action, args, tokens_in, tokens_out}
    tool/call    {step, call_id, name, args}
    tool/result  {step, call_id, content, is_error}
    todo/write   {todos}                       (after a successful todo_write)
    turn/end     {answer, reason}   reason in: done | token_limit | max_steps | invalid_reply

One event stream, many consumers:
    - console renderer (default, plain output)
    - rich/textual UI (top-tier interface)
    - SessionLog (persist -> projection / fork / replay)
"""
from ..state import State, add_step, format_history
from ..executor import execute
from .think import think
from .check_done import is_done


def console_renderer(type: str, **d):
    """Default renderer: reproduces the plain print output."""
    if type == "step/think":
        total = d["tokens_in"] + d["tokens_out"]
        print(f"\n[{d['name']} Step {d['step']}]")
        print(f"  Tokens: {d['tokens_in']:,} in + {d['tokens_out']:,} out = {total:,} total")
        print(f"  Think: {d['thought'][:100]}")
        print(f"  Action: {d['action']}({d['args']})")
    elif type == "tool/result":
        content = d["content"]
        display = content[:500] + "..." if len(content) > 500 else content
        print(f"  Result: {display}")
    elif type == "turn/end":
        if d["reason"] == "done":
            print("  -> done")
        elif d["reason"] == "token_limit":
            print(f"  -> token limit ({d.get('detail', '')})")
        else:
            print(f"  -> stopped: {d['answer']}")


def run_agent(
    llm,
    tools: dict,
    tool_desc: str,
    goal: str,
    max_steps: int = 100,
    max_tokens: int = 130000,
    name: str = "agent",
    verbose: bool = True,
    skills_prompt: str = "",
    memory_prompt: str = "",
    on_event=None,
    conversation: str = "",
    prompt_name: str = "agent",
    approve=None,
) -> str:
    """Run the agent main loop.

    on_event(type, **data): event callback. None = default console rendering.
    conversation: prior dialogue of this session (projected from the session log).
    approve(action, args) -> bool: optional policy check before each tool call.
    """
    if on_event is None:
        emit = console_renderer if verbose else (lambda type, **d: None)
    else:
        emit = on_event

    state = State(goal=goal)

    # Reset LLM token counters
    if hasattr(llm, 'reset_usage'):
        llm.reset_usage()
    elif hasattr(llm, 'total_prompt_tokens'):
        llm.total_prompt_tokens = 0
        llm.total_completion_tokens = 0

    emit("turn/start", goal=goal, name=name)

    invalid = 0  # consecutive replies that did not parse

    for i in range(max_steps):
        step = i + 1
        emit("step/start", step=step)

        # 1. Think (streaming in UI mode: each LLM delta emits assistant/chunk)
        history = format_history(state)
        if conversation:
            history = f"[Earlier in this session]\n{conversation}\n[This turn]\n" + history
        on_chunk = None
        if on_event is not None:
            on_chunk = lambda text, kind="content": emit("assistant/chunk", step=step, text=text, kind=kind)
        thought, action, args = think(
            llm, goal, tool_desc, history,
            skills=skills_prompt,
            memory=memory_prompt,
            on_chunk=on_chunk,
            prompt_name=prompt_name
        )

        prompt_tokens = getattr(llm, 'total_prompt_tokens', 0)
        completion_tokens = getattr(llm, 'total_completion_tokens', 0)
        total_tokens = prompt_tokens + completion_tokens

        emit("step/think", step=step, name=name, thought=thought,
             action=action, args=args,
             tokens_in=prompt_tokens, tokens_out=completion_tokens)

        # 2. Done?
        if is_done(action):
            if isinstance(args, dict):
                key = next((k for k in ("answer", "result") if k in args), None)
                answer = args[key] if key else thought
            else:
                answer = args if args not in (None, "") else thought
            answer = str(answer)  # the LLM may hand back int/list etc.
            emit("turn/end", answer=answer, reason="done")
            return answer

        # 3. Token budget check (after the call: a run stops within one call)
        if total_tokens >= max_tokens:
            detail = f"{total_tokens:,}/{max_tokens:,}"
            emit("turn/end", answer=f"Token limit ({total_tokens:,} tokens)",
                 reason="token_limit", detail=detail)
            return f"Token limit ({total_tokens:,} tokens)"

        # 4. Invalid reply: nothing runs; the parse error is the observation
        if action is None:
            invalid += 1
            if invalid == 2:
                answer = f"Stopped after 2 invalid replies. Last reply: {thought[:200]}"
                emit("turn/end", answer=answer, reason="invalid_reply")
                return answer
            result = f"Error: {args['error']}. Reply with exactly one JSON object."
            emit("tool/result", step=step, call_id=f"s{step}", content=result, is_error=True)
            add_step(state, thought, "(invalid reply)", {}, result)
            continue
        invalid = 0

        # 5. Execute
        call_id = f"s{step}"
        emit("tool/call", step=step, call_id=call_id, name=action, args=args)
        result = execute(action, args, tools, approve)
        failed = result.startswith("Error:")  # tools report failures with this prefix
        emit("tool/result", step=step, call_id=call_id, content=result, is_error=failed)
        if action == "todo_write" and not failed:
            emit("todo/write", todos=args.get("todos", []))

        # 6. Update
        add_step(state, thought, action, args, result)

    emit("turn/end", answer=f"Unfinished ({max_steps} steps)", reason="max_steps")
    return f"Unfinished ({max_steps} steps)"
