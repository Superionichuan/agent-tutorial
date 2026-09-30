"""Load a skill's instructions (progressive disclosure)"""
from ...skills import get_skills


def read_skill(name: str) -> str:
    """Return the full instructions of the named skill"""
    skills = get_skills()
    if name not in skills:
        return f"Error: no skill named '{name}'. Available: {', '.join(skills) or 'none'}"
    return skills[name].instructions
