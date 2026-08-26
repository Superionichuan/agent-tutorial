"""Workspace helpers"""
from pathlib import Path


def get_workspace_str(workspace: Path) -> str:
    """Absolute workspace path as string"""
    return str(workspace.absolute())
