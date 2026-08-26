"""Write a file"""
from pathlib import Path


def write_file(path: str, content: str) -> str:
    """Write a file"""
    try:
        Path(path).write_text(content)
        return f"OK: {path}"
    except Exception as e:
        return f"Error: {e}"
