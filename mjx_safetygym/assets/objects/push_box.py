from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from ..color import COLOR
from ..group import GROUP
from ...bases.base_object import FreeGeom


@dataclass
class PushBox(FreeGeom):
    name: str = 'push_box'
    size: float = 0.2
    null_dist: float = 2.0
    placements: list = None
    locations: list = field(default_factory=list)
    keepout: float = 0.3

    reward_box_dist: float = 1.0
    reward_box_goal: float = 1.0
    reward_goal: float = 1.0

    color: np.ndarray = field(default_factory=lambda: np.asarray(COLOR['push_box']))
    group: int = GROUP['push_box']
    is_lidar_observed: bool = True
    is_comp_observed: bool = True
    is_constrained: bool = False

    def get_config(self, xy_pos, rot):
        return {
            'name': self.name,
            'pos': np.r_[xy_pos, self.size],
            'rot': rot,
            'freejoint': self.name,
            'geoms': [{
                'name': self.name,
                'size': [self.size] * 3,
                'type': 'box',
                'contype': 1,
                'conaffinity': 1,
                'group': self.group,
                'rgba': np.asarray(self.color),
                'density': 0.0001,
            }],
        }

    @property
    def pos(self) -> np.ndarray:
        return np.asarray(self.engine.body_pos(self.name))
