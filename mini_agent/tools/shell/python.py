"""Run Python code"""
import subprocess
import sys

from ..workspace import cwd
from .bash import decode, utf8_env


def run_python(code: str) -> str:
    """Run Python code (fresh process; sees newly installed packages)"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            timeout=120,
            cwd=cwd(),
            env=utf8_env(),
        )
        output = decode(result.stdout) + decode(result.stderr)
        return output if output else "OK"
    except subprocess.TimeoutExpired:
        return "Error: execution timeout"
    except Exception as e:
        return f"Error: {e}"
