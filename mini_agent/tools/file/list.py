"""List a directory"""
from ..workspace import resolve

MAX_ITEMS = 20


def list_dir(path: str = ".") -> str:
    """List directory content"""
    try:
        items = sorted(resolve(path).iterdir())[:MAX_ITEMS]
        return "\n".join(str(p) for p in items) or "(empty directory)"
    except Exception as e:
        return f"Error: {e}"
