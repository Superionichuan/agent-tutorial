"""State dataclass"""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class State:
    goal: str
    steps: List[Dict] = field(default_factory=list)
    steps_done: int = 0   # step numbers already used in this turn (a continued turn goes on after them)
