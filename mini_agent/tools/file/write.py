"""Write a file"""
from ..workspace import resolve


def write_file(path: str, content: str) -> str:
    """Write a file (missing parent directories are created)"""
    try:
        target = resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"OK: {target}"
    except Exception as e:
        return f"Error: {e}"
