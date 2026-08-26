"""Agent config dataclass"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentConfig:
    name: str = "agent"
    workspace: Path = None
    model: str = None
    max_tokens: int = 130000  # 130K token budget
    max_steps: int = 100      # step cap (safety net)

    def __post_init__(self):
        if self.workspace is None:
            self.workspace = Path.cwd()
        elif isinstance(self.workspace, str):
            self.workspace = Path(self.workspace)
