"""Read a file"""
from pathlib import Path


def read_file(path: str) -> str:
    """Read file content"""
    try:
        return Path(path).read_text()
    except Exception as e:
        return f"Error: {e}"
