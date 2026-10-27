from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from ..color import COLOR
from ..group import GROUP
from ...bases.base_object import Geom


@dataclass
class Hazards(Geom):
    name: str = 'hazards'
    num: int = 1
    size: float = 0.2
    placements: list = None
    locations: list = field(default_factory=list)
    keepout: float = 0.18

    cost: float = 1.0

    color: np.ndarray = field(default_factory=lambda: np.asarray(COLOR['hazard']))
    alpha: float = 0.25
    group: int = GROUP['hazard']
    is_lidar_observed: bool = True
    is_constrained: bool = True

    def get_config(self, xy_pos, rot):
        return {
            'name': self.name,
            'pos': np.r_[xy_pos, 2e-2],
            'rot': rot,
            'geoms': [{
                'name': self.name,
                'size': [self.size, 1e-2],
                'type': 'cylinder',
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
            hz_xy = np.asarray(self.engine.body_pos(f'{self.name}_{i}'))[:2]
            if np.linalg.norm(agent_xy - hz_xy) <= self.size:
                hits += 1
        return {'cost_hazards': float(self.cost * hits)}

    @property
    def pos(self) -> list[np.ndarray]:
        return [np.asarray(self.engine.body_pos(f'{self.name}_{i}')) for i in range(self.num)]
