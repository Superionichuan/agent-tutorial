"""State dataclass"""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class State:
    goal: str
    steps: List[Dict] = field(default_factory=list)
