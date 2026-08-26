"""Load prompt templates"""
from pathlib import Path

# prompts/ lives at the package root so users can edit templates
PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts"


def load_prompt(name: str = "agent") -> str:
    """Load a prompt template"""
    path = PROMPT_DIR / f"{name}.md"
    return path.read_text()
