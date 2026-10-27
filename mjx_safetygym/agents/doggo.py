from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from ..bases.base_agent import BaseAgent


@dataclass
class Doggo(BaseAgent):
    name: str = 'agent'
    xml_name: str = 'doggo'

    keepout: float = 0.4
    action_dim: int = 12
    action_low: np.ndarray = None
    action_high: np.ndarray = None

    z_fall_threshold: float = 0.2

    def __post_init__(self):
        if self.action_low is None:
            self.action_low = -np.ones(self.action_dim, dtype=np.float64)
        if self.action_high is None:
            self.action_high = np.ones(self.action_dim, dtype=np.float64)

    def is_alive(self) -> bool:

        if self.engine is None:
            return True
        return float(self.pos[2]) > self.z_fall_threshold
