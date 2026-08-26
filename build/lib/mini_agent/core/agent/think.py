"""The think step"""
from ..prompt import build_prompt, load_prompt
from ..parser import parse_response


def think(
    llm,
    goal: str,
    tool_desc: str,
    history: str,
    skills: str = "",
    memory: str = "",
    on_chunk=None,
    prompt_name: str = "agent"
):
    """Run the think step; returns (thought, action, args)

    Args:
        llm: LLM instance
        goal: user goal
        tool_desc: tool descriptions
        history: step history
        skills: skills prompt section
        memory: memory content
        on_chunk: streaming callback on_chunk(text). Used per-delta when the backend
                  supports call_stream; otherwise falls back to a one-shot call.
    """
    template = load_prompt(prompt_name)
    prompt = build_prompt(goal, tool_desc, history, skills=skills, memory=memory, template=template)
    if on_chunk is not None and hasattr(llm, "call_stream"):
        response = llm.call_stream(prompt, on_chunk)
    else:
        response = llm(prompt)
    return parse_response(response)
