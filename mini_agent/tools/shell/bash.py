"""Run a bash command"""
import subprocess

from ..workspace import cwd

TIMEOUT = 120


def bash(cmd: str) -> str:
    """Run a bash command"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=TIMEOUT,
            cwd=cwd()
        )
        output = result.stdout + result.stderr
        return output if output else "OK"
    except subprocess.TimeoutExpired:
        return "Error: timeout"
    except Exception as e:
        return f"Error: {e}"
