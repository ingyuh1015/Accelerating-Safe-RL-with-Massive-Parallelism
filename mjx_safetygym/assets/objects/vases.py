from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from ..color import COLOR
from ..group import GROUP
from ...bases.base_object import FreeGeom


@dataclass
class Vases(FreeGeom):
    name: str = 'vases'
    num: int = 0
    size: float = 0.1
    placements: list = None
    locations: list = field(default_factory=list)
    keepout: float = 0.15

    cost: float = 1.0
    contact_cost: float = 1.0
    displace_cost: float = 0.0
    velocity_cost: float = 0.0

    color: np.ndarray = field(default_factory=lambda: np.asarray(COLOR['vase']))
    group: int = GROUP['vase']
    is_lidar_observed: bool = True
    is_constrained: bool = True

    def get_config(self, xy_pos, rot):
        return {
            'name': self.name,
            'pos': np.r_[xy_pos, self.size - 1e-3],
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
                'density': 0.001,
            }],
        }

    def cal_cost(self) -> dict:

        cost = 0.0
        for i in range(self.num):
            if self.contact_cost and self.engine.contact_with('agent', f'{self.name}_{i}'):
                cost += self.contact_cost
        return {'cost_vases': float(cost)}

    @property
    def pos(self) -> list[np.ndarray]:
        return [np.asarray(self.engine.body_pos(f'{self.name}_{i}')) for i in range(self.num)]
