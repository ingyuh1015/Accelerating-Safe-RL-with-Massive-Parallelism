from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from ..color import COLOR
from ..group import GROUP
from ...bases.base_object import Geom


@dataclass
class Sigwalls(Geom):
    name: str = 'sigwalls'
    num: int = 0
    size: float = 3.5
    placements: list = None
    locations: list = field(default_factory=list)
    keepout: float = 0.0

    cost: float = 1.0

    color: np.ndarray = field(default_factory=lambda: np.asarray(COLOR['sigwall']))
    alpha: float = 0.1
    group: int = GROUP['sigwall']
    is_lidar_observed: bool = False
    is_constrained: bool = True

    def get_config(self, xy_pos, rot):
        return {
            'name': self.name,
            'pos': np.r_[xy_pos, 0.25],
            'rot': rot,
            'geoms': [{
                'name': self.name,
                'size': [0.05, self.size, 0.3],
                'type': 'box',
                'contype': 0,
                'conaffinity': 0,
                'group': self.group,
                'rgba': np.asarray(self.color) * np.array([1, 1, 1, self.alpha]),
            }],
        }

    def cal_cost(self) -> dict:

        agent_xy = np.asarray(self.agent.pos)[:2]
        hits = 0
        for i in range(self.num):
            sw_xy = np.asarray(self.engine.body_pos(f'{self.name}_{i}'))[:2]

            if abs(agent_xy[0]) > abs(sw_xy[0]):
                hits += 1
        return {'cost_sigwalls': float(self.cost * hits)}

    @property
    def pos(self) -> list[np.ndarray]:
        return [np.asarray(self.engine.body_pos(f'{self.name}_{i}')) for i in range(self.num)]
