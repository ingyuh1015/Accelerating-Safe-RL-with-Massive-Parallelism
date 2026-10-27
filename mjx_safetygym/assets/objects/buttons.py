from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from ..color import COLOR
from ..group import GROUP
from ...bases.base_object import FreeGeom


@dataclass
class Buttons(FreeGeom):
    name: str = 'buttons'
    num: int = 4
    size: float = 0.1
    placements: list = None
    locations: list = field(default_factory=list)
    keepout: float = 0.2

    cost: float = 1.0
    resampling_delay: int = 10
    reward_goal: float = 1.0

    color: np.ndarray = field(default_factory=lambda: np.asarray(COLOR['button']))
    group: int = GROUP['button']
    is_lidar_observed: bool = True
    is_constrained: bool = True

    def get_config(self, xy_pos, rot):
        return {
            'name': self.name,
            'pos': np.r_[xy_pos, self.size],
            'rot': rot,
            'freejoint': self.name,
            'geoms': [{
                'name': self.name,
                'size': [self.size] * 3,
                'type': 'sphere',
                'contype': 1,
                'conaffinity': 1,
                'group': self.group,
                'rgba': np.asarray(self.color),
                'density': 0.001,
            }],
        }

    def cal_cost(self) -> dict:

        cost = 0.0
        goal_btn = getattr(self.engine, 'goal_button', None)
        for i in range(self.num):
            btn_name = f'{self.name}_{i}'
            if i == goal_btn:
                continue
            if self.engine.contact_with('agent', btn_name):
                cost += self.cost
        return {'cost_buttons': float(cost)}

    @property
    def pos(self) -> list[np.ndarray]:
        return [np.asarray(self.engine.body_pos(f'{self.name}_{i}')) for i in range(self.num)]
