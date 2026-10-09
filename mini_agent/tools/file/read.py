"""Read a file"""
from ..workspace import resolve


def read_file(path: str) -> str:
    """Read file content"""
    try:
        return resolve(path).read_text()
    except Exception as e:
        return f"Error: {e}"
