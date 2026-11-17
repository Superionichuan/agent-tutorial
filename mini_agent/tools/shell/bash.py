"""Run a shell command

Linux and macOS: bash (or /bin/sh where bash is missing).
Windows: Git Bash when Git for Windows is installed, otherwise PowerShell.
MINI_AGENT_SHELL=bash | sh | powershell | pwsh | <path to a shell> picks one.
Output is read as UTF-8, so text such as Chinese comes back intact.
"""
import base64
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from ..workspace import cwd

TIMEOUT = 120
WINDOWS = os.name == "nt"

# PowerShell writes in the console code page unless told otherwise
_PS_UTF8 = ("[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "$OutputEncoding = [System.Text.Encoding]::UTF8; ")


def _git_bash() -> Optional[str]:
    """bash.exe from Git for Windows (not the WSL launcher in System32)."""
    found = shutil.which("bash")
    if found and not any(part in found.lower() for part in ("\\system32\\", "\\windowsapps\\")):
        return found
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramW6432"),
                 os.environ.get("LOCALAPPDATA") and str(Path(os.environ["LOCALAPPDATA"]) / "Programs")):
        if base and (Path(base) / "Git" / "bin" / "bash.exe").is_file():
            return str(Path(base) / "Git" / "bin" / "bash.exe")
    return None


def shell() -> Tuple[str, str]:
    """(kind, executable) of the shell commands run in; kind is bash, sh or powershell."""
    choice = os.environ.get("MINI_AGENT_SHELL", "").strip()
    if choice:
        low = Path(choice).name.lower()
        exe = shutil.which(choice) or choice
        if low.startswith(("powershell", "pwsh")):
            return "powershell", exe
        return ("sh" if low.startswith("sh") else "bash"), exe
    if WINDOWS:
        git = _git_bash()
        if git:
            return "bash", git
        return "powershell", shutil.which("pwsh") or shutil.which("powershell") or "powershell"
    found = shutil.which("bash")
    return ("bash", found) if found else ("sh", "/bin/sh")


def describe_system() -> str:
    """One line for the prompt: the operating system and the shell the bash tool uses."""
    kind, exe = shell()
    if WINDOWS:
        if kind == "powershell":
            return "Windows; the bash tool runs PowerShell commands (use PowerShell syntax)"
        return "Windows; the bash tool runs Git Bash (Windows paths also work, e.g. C:/Users)"
    system = "macOS" if os.uname().sysname == "Darwin" else os.uname().sysname
    return f"{system}; the bash tool runs {'bash' if kind == 'bash' else 'sh'}"


def command_line(cmd: str) -> list:
    """The argument list that runs cmd in the shell."""
    kind, exe = shell()
    if kind == "powershell":
        script = base64.b64encode((_PS_UTF8 + cmd).encode("utf-16-le")).decode("ascii")
        return [exe, "-NoProfile", "-NonInteractive", "-EncodedCommand", script]
    return [exe, "-c", cmd]


def utf8_env() -> dict:
    """The environment for child processes: Python children print UTF-8 too."""
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    return env


def decode(data: bytes) -> str:
    return (data or b"").decode("utf-8", errors="replace")


def bash(cmd: str) -> str:
    """Run a shell command (bash; on Windows Git Bash, or PowerShell when Git Bash is missing)"""
    try:
        result = subprocess.run(command_line(cmd), capture_output=True, timeout=TIMEOUT,
                                cwd=cwd(), env=utf8_env())
        output = decode(result.stdout) + decode(result.stderr)
        return output if output else "OK"
    except subprocess.TimeoutExpired:
        return "Error: timeout"
    except Exception as e:
        return f"Error: {e}"
