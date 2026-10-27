from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from ..color import COLOR
from ..group import GROUP
from ...bases.base_object import Geom


@dataclass
class Goal(Geom):
    name: str = 'goal'
    size: float = 0.3
    placements: list = None
    locations: list = field(default_factory=list)
    keepout: float = 0.4

    reward_goal: float = 1.0
    reward_distance: float = 1.0

    color: np.ndarray = field(default_factory=lambda: np.asarray(COLOR['goal']))
    alpha: float = 0.25
    group: int = GROUP['goal']
    is_lidar_observed: bool = True
    is_comp_observed: bool = False
    is_constrained: bool = False

    def get_config(self, xy_pos, rot):
        return {
            'name': self.name,
            'pos': np.r_[xy_pos, self.size / 2 + 1e-2],
            'rot': rot,
            'geoms': [{
                'name': self.name,
                'size': [self.size, self.size / 2],
                'type': 'cylinder',
                'contype': 0,
                'conaffinity': 0,
                'group': self.group,
                'rgba': np.asarray(self.color) * np.array([1, 1, 1, self.alpha]),
            }],
        }

    @property
    def pos(self) -> np.ndarray:
        return np.asarray(self.engine.body_pos(self.name))
