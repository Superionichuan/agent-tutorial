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
    turn/end     {answer, reason}   reason in: done | token_limit | max_steps | invalid_reply |
                                    llm_error (the model call failed: bad key, network, ...) |
                                    no_recording (offline mode: no recorded reply matches) |
                                    context_full (lifecycle: even compacted, the prompt would
                                    not fit the model's window) |
                                    needs_approval (a risky call needs a yes and there is no
                                    terminal to ask; also carries tool)
    context/compact {step, before_tokens, after_tokens, method, at, replaced, steps}
                                    (with a lifecycle: the steps were slimmed before this call)

One event stream, many consumers:
    - console renderer (default, plain output)
    - rich/textual UI (top-tier interface)
    - SessionLog (persist -> projection / fork / replay)
"""
from ..state import State, add_step, format_history
from ..executor import execute
from ..prompt import build_prompt, load_prompt
from ...tools.spec import ToolSet
from ..policy import NeedsApproval, resolve_approval
from .think import think
from .check_done import is_done


def failed_call(d: dict) -> bool:
    """True for the step/think event of a model call that failed (no reply; the turn/end
    that follows says why): renderers skip it."""
    return d.get("action") is None and bool((d.get("args") or {}).get("llm_error"))


def console_renderer(type: str, /, **d):
    """Default renderer: reproduces the plain print output."""
    if type == "context/compact":
        print(f"\n  [context compacted before step {d.get('step')}: {d.get('before_tokens', 0):,} -> "
              f"{d.get('after_tokens', 0):,} tokens, {d.get('method', '?')}]")
    elif type == "step/think" and not failed_call(d):
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
    tools,
    tool_desc,
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
    lifecycle=None,
    state: State = None,
) -> str:
    """Run the agent main loop.

    tools: a list of functions / ToolSpecs, or a dict of name -> function.
    tool_desc: the [Tools] text; None = rendered from the tools.
    on_event(type, **data): event callback. None = default console rendering.
    conversation: prior dialogue of this session (projected from the session log).
    approve: None (allow all), a policy "ask" | "auto" | "deny", an Approval, or a
             function approve(action, args) -> bool checked before each tool call.
    lifecycle: None, or a mini_agent.lifecycle.Lifecycle: written memory, compaction before
             each model call, a handoff at every stop (the lines marked "lifecycle").
    state: an unfinished turn to continue (mini_agent.lifecycle.restore_steps); None = new.
    """
    toolset = ToolSet.build(tools, tool_desc)
    funcs, tool_desc = toolset.funcs, toolset.desc
    approve = resolve_approval(approve, [n for n, s in toolset.specs.items() if s.risky])
    if on_event is None:
        emit = console_renderer if verbose else (lambda type, **d: None)
    else:
        emit = on_event

    resumed = state is not None
    if state is None:
        state = State(goal=goal)

    # Reset LLM token counters
    if hasattr(llm, 'reset_usage'):
        llm.reset_usage()
    elif hasattr(llm, 'total_prompt_tokens'):
        llm.total_prompt_tokens = 0
        llm.total_completion_tokens = 0

    if lifecycle is not None:  # lifecycle: follow this turn (turn/end writes the handoff)
        emit = lifecycle.attach(emit, state, resumed=resumed)
        memory_prompt = lifecycle.start(goal, memory_prompt)

    def history_of():
        if lifecycle is None:
            history = format_history(state)
        else:  # lifecycle: it keeps the prompt under the window; no step is dropped silently
            history = format_history(state, max_tokens=lifecycle.history_tokens)
        if conversation:
            history = f"[Earlier in this session]\n{conversation}\n[This turn]\n" + history
        return history

    def prompt_of(memory):  # the prompt think() sends with this memory (lifecycle measures it)
        return build_prompt(goal, tool_desc, history_of(), skills=skills_prompt, memory=memory,
                            template=load_prompt(prompt_name))

    try:
        emit("turn/start", goal=goal, name=name, **({"resumed": True} if resumed else {}))

        invalid = 0  # consecutive replies that did not parse

        for _ in range(max_steps):
            step = state.steps_done + 1   # a continued or reused State goes on numbering
            state.steps_done = step
            emit("step/start", step=step)

            # 1. Think (streaming in UI mode: each LLM delta emits assistant/chunk)
            if lifecycle is not None:  # lifecycle: slim old steps if the next prompt is too big
                memory_prompt = lifecycle.before_think(step, prompt_of)
                if lifecycle.overflow:  # even compacted, the prompt would not fit the window
                    answer = (f"Context full: the next prompt needs about {lifecycle.overflow:,} "
                              f"tokens, more than the model's window ({lifecycle.window:,})")
                    emit("turn/end", answer=answer, reason="context_full")
                    return answer
                spent = getattr(llm, 'total_prompt_tokens', 0) + getattr(llm, 'total_completion_tokens', 0)
                if spent >= max_tokens:  # a summary used up the budget: no further call
                    emit("turn/end", answer=f"Token limit ({spent:,} tokens)",
                         reason="token_limit", detail=f"{spent:,}/{max_tokens:,}")
                    return f"Token limit ({spent:,} tokens)"
            history = history_of()
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

            # 4. The model call failed (key, credit, network): stop now, say why
            if action is None and args.get("llm_error"):
                emit("turn/end", answer=args["error"], reason=args.get("stop") or "llm_error")
                return args["error"]

            # 5. Invalid reply: nothing runs; the parse error is the observation
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

            # 6. Execute
            call_id = f"s{step}"
            emit("tool/call", step=step, call_id=call_id, name=action, args=args)
            try:
                result = execute(action, args, funcs, approve)
            except NeedsApproval:
                # nobody can say yes here: stop, rather than let the model try one risky
                # tool after another; -c (with --yes, or in a terminal) continues the task
                result = (f"Error: '{action}' did not run: it needs approval, "
                          "and there is no terminal to ask")
                emit("tool/result", step=step, call_id=call_id, content=result, is_error=True)
                add_step(state, thought, action, args, result)
                answer = f"Stopped: {action} needs approval, and there is no terminal to ask"
                emit("turn/end", answer=answer, reason="needs_approval", tool=action)
                return answer
            failed = result.startswith("Error:")  # tools report failures with this prefix
            emit("tool/result", step=step, call_id=call_id, content=result, is_error=failed)
            if action == "todo_write" and not failed:
                emit("todo/write", todos=args.get("todos", []))

            # 7. Update
            add_step(state, thought, action, args, result)

        emit("turn/end", answer=f"Unfinished ({max_steps} steps)", reason="max_steps")
        return f"Unfinished ({max_steps} steps)"
    except BaseException as e:
        if lifecycle is not None:  # lifecycle: a crash or Ctrl-C still leaves a handoff
            lifecycle.end("interrupted" if isinstance(e, KeyboardInterrupt) else "error",
                          f"{type(e).__name__}: {e}")
        raise
