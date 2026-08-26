"""Skills module — a minimal plugin system"""
from .loader import Skill, load_skills, get_skill, list_skills, format_skills_prompt

__all__ = ["Skill", "load_skills", "get_skill", "list_skills", "format_skills_prompt"]
