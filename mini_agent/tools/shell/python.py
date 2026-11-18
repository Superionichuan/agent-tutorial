"""Run Python code"""
import subprocess
import sys

from .bash import decode, run_tool

TIMEOUT = 120


def run_python(code: str) -> str:
    """Run Python code (fresh process; sees newly installed packages)"""
    try:
        stdout, stderr = run_tool([sys.executable, "-c", code], TIMEOUT)
        output = decode(stdout) + decode(stderr)
        return output if output else "OK"
    except subprocess.TimeoutExpired:
        return "Error: execution timeout"
    except Exception as e:
        return f"Error: {e}"
