"""List a directory"""
from pathlib import Path

MAX_ITEMS = 20


def list_dir(path: str = ".") -> str:
    """List directory content"""
    try:
        items = list(Path(path).iterdir())[:MAX_ITEMS]
        return "\n".join(str(p) for p in items)
    except Exception as e:
        return f"Error: {e}"
