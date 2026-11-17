"""Run a shell command

Linux and macOS: bash (or /bin/sh where bash is missing).
Windows: Git Bash when Git for Windows is installed, otherwise PowerShell.
MINI_AGENT_SHELL=bash | sh | powershell | pwsh | <path to a shell> picks one.
Output is read as UTF-8, so text such as Chinese comes back intact.
"""
import base64
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

from ..workspace import cwd

TIMEOUT = 120
WINDOWS = os.name == "nt"

# No progress records (e.g. "Preparing modules for first use"), no colour codes in messages
# (PowerShell 7); PowerShell writes in the console code page unless told otherwise
_PS_PREFIX = ("$ProgressPreference = 'SilentlyContinue'; "
              "if ($PSStyle) { $PSStyle.OutputRendering = 'PlainText' }; "
              "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
              "$OutputEncoding = [System.Text.Encoding]::UTF8; ")
_CLIXML = "#< CLIXML"
_PS_ESCAPE = re.compile(r"_x([0-9A-Fa-f]{4})_")
_PS_LABEL = {"warning": "WARNING: ", "verbose": "VERBOSE: ", "debug": "DEBUG: "}
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def from_clixml(err: str) -> str:
    """The text of PowerShell's error output.

    Started with -EncodedCommand, PowerShell writes its error, warning and progress streams
    to stderr as CLIXML. This keeps the text of the messages and drops progress records;
    anything else comes back unchanged.
    """
    text = err.lstrip("﻿ \t\r\n")
    if not text.startswith(_CLIXML):
        return err
    parts = []
    for block in text.split(_CLIXML):
        if not block.strip():
            continue
        try:
            root = ET.fromstring(block.strip())
        except ET.ParseError:
            return err
        for element in root:
            if element.tag.rsplit("}", 1)[-1] == "S" and element.text:
                piece = _PS_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), element.text)
                label = _PS_LABEL.get((element.get("S") or "").lower())
                if label:                    # one record per element, as the console shows it
                    piece = label + piece + ("" if piece.endswith("\n") else "\n")
                parts.append(piece)
    joined = "".join(parts).encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    return _ANSI.sub("", joined).replace("\r\n", "\n")


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
        script = base64.b64encode((_PS_PREFIX + cmd).encode("utf-16-le")).decode("ascii")
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
        output = decode(result.stdout) + from_clixml(decode(result.stderr))
        return output if output else "OK"
    except subprocess.TimeoutExpired:
        return "Error: timeout"
    except Exception as e:
        return f"Error: {e}"
