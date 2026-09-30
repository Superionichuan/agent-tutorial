"""Skills module: a minimal plugin system"""
from .loader import (Skill, load_skills, get_skill, get_skills, list_skills,
                     reload_skills, format_skills_prompt)

__all__ = ["Skill", "load_skills", "get_skill", "get_skills", "list_skills",
           "reload_skills", "format_skills_prompt"]
