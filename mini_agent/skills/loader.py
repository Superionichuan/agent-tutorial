"""Skill loader

Skill Layout:
  skills/
    skill_name/
      SKILL.md    # frontmatter + instructions

SKILL.md format:
  ---
  name: skill_name
  description: one-line description
  ---

  # details
  usage details...
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import re

# skill directory search order
SKILL_DIRS = [
    Path(__file__).parent.parent.parent / "skills",  # project skills/
    Path.home() / ".mini_agent" / "skills",          # user ~/.mini_agent/skills/
]


@dataclass
class Skill:
    """Skill definition"""
    name: str
    description: str
    instructions: str
    path: Path


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse frontmatter and body"""
    if not content.startswith("---"):
        return {}, content

    # find the second ---
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
    if not match:
        return {}, content

    fm_text, body = match.groups()

    # naive YAML-like frontmatter parsing
    fm = {}
    for line in fm_text.strip().split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            fm[key.strip()] = val.strip().strip('"\'')

    return fm, body.strip()


def load_skill(skill_dir: Path) -> Optional[Skill]:
    """Load one skill from a directory"""
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        return None

    content = skill_file.read_text()
    fm, body = parse_frontmatter(content)

    return Skill(
        name=fm.get("name", skill_dir.name),
        description=fm.get("description", ""),
        instructions=body,
        path=skill_dir,
    )


def load_skills() -> dict[str, Skill]:
    """Load all skills"""
    skills = {}

    for base_dir in SKILL_DIRS:
        if not base_dir.exists():
            continue

        for skill_dir in base_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            if skill_dir.name.startswith("."):
                continue

            skill = load_skill(skill_dir)
            if skill:
                # later loads override earlier ones (user first)
                skills[skill.name] = skill

    return skills


# cache
_skills_cache: Optional[dict[str, Skill]] = None


def get_skills() -> dict[str, Skill]:
    """Get skills (cached)"""
    global _skills_cache
    if _skills_cache is None:
        _skills_cache = load_skills()
    return _skills_cache


def get_skill(name: str) -> Optional[Skill]:
    """Get a skill by name"""
    return get_skills().get(name)


def list_skills() -> list[str]:
    """List all skill names"""
    return list(get_skills().keys())


def format_skills_prompt(skill_names: Optional[list[str]] = None) -> str:
    """Format skills as a prompt section"""
    skills = get_skills()

    if skill_names:
        skills = {k: v for k, v in skills.items() if k in skill_names}

    if not skills:
        return ""

    lines = []
    for name, skill in skills.items():
        desc = f" - {skill.description}" if skill.description else ""
        lines.append(f"〈{name}{desc}〉")
        lines.append(skill.instructions)
        lines.append("")

    return "\n".join(lines)
