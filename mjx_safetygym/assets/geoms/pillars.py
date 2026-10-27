from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from ..color import COLOR
from ..group import GROUP
from ...bases.base_object import Geom


@dataclass
class Pillars(Geom):
    name: str = 'pillars'
    num: int = 0
    size: float = 0.2
    height: float = 0.5
    placements: list = None
    locations: list = field(default_factory=list)
    keepout: float = 0.3

    cost: float = 1.0

    color: np.ndarray = field(default_factory=lambda: np.asarray(COLOR['pillar']))
    group: int = GROUP['pillar']
    is_lidar_observed: bool = True
    is_constrained: bool = True

    def get_config(self, xy_pos, rot):
        return {
            'name': self.name,
            'pos': np.r_[xy_pos, self.height / 2],
            'rot': rot,
            'geoms': [{
                'name': self.name,
                'size': [self.size, self.height / 2],
                'type': 'cylinder',
                'contype': 1,
                'conaffinity': 1,
                'group': self.group,
                'rgba': np.asarray(self.color),
            }],
        }

    def cal_cost(self) -> dict:

        contacts = self.engine.contact_with('agent', self.name)
        return {'cost_pillars': float(self.cost) if contacts else 0.0}

    @property
    def pos(self) -> list[np.ndarray]:
        return [np.asarray(self.engine.body_pos(f'{self.name}_{i}')) for i in range(self.num)]
