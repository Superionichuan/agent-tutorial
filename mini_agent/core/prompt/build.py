"""Build the prompt"""
from .load import load_prompt


def build_prompt(
    goal: str,
    tools: str,
    history: str,
    skills: str = "",
    memory: str = "",
    template: str = None
) -> str:
    """Build the full prompt

    Args:
        goal: user goal
        tools: tool descriptions
        history: step history
        skills: skills prompt section
        memory: memory content
        template: custom template
    """
    template = template or load_prompt()
    return template.format(
        goal=goal,
        tools=tools,
        history=history,
        skills=skills,
        memory=memory
    )
