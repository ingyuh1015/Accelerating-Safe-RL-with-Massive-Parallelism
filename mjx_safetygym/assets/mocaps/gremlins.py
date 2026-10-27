from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from ..color import COLOR
from ..group import GROUP
from ...bases.base_object import Mocap


@dataclass
class Gremlins(Mocap):
    name: str = 'gremlins'
    num: int = 0
    size: float = 0.1
    placements: list = None
    locations: list = field(default_factory=list)
    keepout: float = 0.4

    travel: float = 0.3
    contact_cost: float = 1.0
    dist_threshold: float = 0.2
    dist_cost: float = 1.0

    color: np.ndarray = field(default_factory=lambda: np.asarray(COLOR['gremlin']))
    group: int = GROUP['gremlin']
    is_lidar_observed: bool = True
    is_constrained: bool = True

    def get_config(self, xy_pos, rot):
        return {
            'name': self.name,
            'pos': np.r_[xy_pos, self.size],
            'rot': rot,
            'mocap': True,
            'geoms': [{
                'name': self.name,
                'size': [self.size] * 3,
                'type': 'box',
                'contype': 1,
                'conaffinity': 1,
                'group': self.group,
                'rgba': np.asarray(self.color),
            }],
        }

    def cal_cost(self) -> dict:

        cost = 0.0
        agent_xy = np.asarray(self.agent.pos)[:2]
        for i in range(self.num):
            gr_name = f'{self.name}_{i}'
            if self.contact_cost and self.engine.contact_with('agent', gr_name):
                cost += self.contact_cost
            if self.dist_cost:
                gr_xy = np.asarray(self.engine.body_pos(gr_name))[:2]
                if np.linalg.norm(agent_xy - gr_xy) < self.dist_threshold:
                    cost += self.dist_cost
        return {'cost_gremlins': float(cost)}

    def move(self) -> None:

        if self.engine is None:
            return
        t = self.engine.time()
        for i in range(self.num):
            origin = np.asarray(self.engine.body_pos(f'{self.name}_{i}_origin'))[:2]
            dx = self.travel * np.cos(2 * np.pi * t + i * np.pi / self.num)
            dy = self.travel * np.sin(2 * np.pi * t + i * np.pi / self.num)
            new_xy = origin + np.array([dx, dy])
            self.set_mocap_pos(f'{self.name}_{i}', np.r_[new_xy, self.size])

    @property
    def pos(self) -> list[np.ndarray]:
        return [np.asarray(self.engine.body_pos(f'{self.name}_{i}')) for i in range(self.num)]
