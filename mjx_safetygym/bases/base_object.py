from __future__ import annotations
import abc
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BaseObject(abc.ABC):


    name: str = 'object'
    placements: list = None
    locations: list = field(default_factory=list)
    keepout: float = 0.3
    is_lidar_observed: bool = False
    is_comp_observed: bool = False
    is_constrained: bool = False


    agent: object = None
    engine: object = None
    random_generator: object = None


    def set_agent(self, agent):
        self.agent = agent

    def set_engine(self, engine):

        self.engine = engine

    def set_random_generator(self, rng):
        self.random_generator = rng

    def cal_cost(self) -> dict:

        return {}

    def process_config(self, config: dict, layout: dict, rots) -> None:


        body = self.get_config(layout[self.name], rots)
        config['bodies'][self.name] = body

    def _specific_agent_config(self) -> None:
        pass

    @property
    def pos(self) -> np.ndarray:

        raise NotImplementedError

    @abc.abstractmethod
    def get_config(self, xy_pos, rot):
        pass


@dataclass
class Geom(BaseObject):
    pass


@dataclass
class FreeGeom(BaseObject):
    pass


@dataclass
class Mocap(BaseObject):


    def set_mocap_pos(self, name: str, value: np.ndarray) -> None:

        if self.engine is not None:
            self.engine.set_mocap_pos(name, value)

    def move(self) -> None:
        pass
