"""Run Python code"""
import subprocess
import sys


def run_python(code: str) -> str:
    """Run Python code (fresh process; sees newly installed packages)"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120
        )
        output = result.stdout + result.stderr
        return output if output else "OK"
    except subprocess.TimeoutExpired:
        return "Error: execution timeout"
    except Exception as e:
        return f"Error: {e}"
