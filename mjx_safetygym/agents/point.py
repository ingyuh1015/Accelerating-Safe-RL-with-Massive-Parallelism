from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from ..bases.base_agent import BaseAgent


@dataclass
class Point(BaseAgent):
    name: str = 'agent'
    xml_name: str = 'point'

    keepout: float = 0.4
    action_dim: int = 2
    action_low: np.ndarray = None
    action_high: np.ndarray = None

    def __post_init__(self):
        if self.action_low is None:
            self.action_low = np.array([-1.0, -1.0], dtype=np.float64)
        if self.action_high is None:
            self.action_high = np.array([1.0, 1.0], dtype=np.float64)

    def is_alive(self) -> bool:

        return True
