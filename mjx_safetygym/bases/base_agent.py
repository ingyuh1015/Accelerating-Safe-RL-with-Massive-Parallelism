from __future__ import annotations
import abc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium


_AGENT_XMLS_DIR = Path(__file__).parent.parent / 'agents' / 'xmls'


@dataclass
class BaseAgent(abc.ABC):


    name: str = 'agent'
    xml_name: str = ''
    placements: list = None
    locations: list = field(default_factory=list)
    keepout: float = 0.4
    rot: Optional[float] = None


    random_generator: object = None
    engine: object = None

    action_dim: int = 0
    action_low: np.ndarray = None
    action_high: np.ndarray = None


    def set_engine(self, engine):
        self.engine = engine

    def set_random_generator(self, rng):
        self.random_generator = rng

    @property
    def xml_path(self) -> str:
        if not self.xml_name:
            raise ValueError(f'{self.__class__.__name__}: xml_name not set')
        path = _AGENT_XMLS_DIR / f'{self.xml_name}.xml'
        if not path.exists():
            raise FileNotFoundError(f'Agent XML not found: {path}')
        return str(path)

    def load_xml_text(self) -> str:

        with open(self.xml_path, 'r') as f:
            return f.read()

    @property
    def action_space(self) -> gymnasium.spaces.Box:
        if self.action_dim <= 0:
            raise ValueError(f'{self.__class__.__name__}: action_dim unset')
        if self.action_low is None or self.action_high is None:
            return gymnasium.spaces.Box(
                low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float64,
            )
        return gymnasium.spaces.Box(
            low=self.action_low, high=self.action_high, shape=(self.action_dim,), dtype=np.float64,
        )

    @property
    def pos(self) -> np.ndarray:

        if self.engine is None:
            raise RuntimeError(f'{self.__class__.__name__}: engine not bound')
        return np.asarray(self.engine.body_pos(self.name))

    @property
    def mat(self) -> np.ndarray:

        return np.asarray(self.engine.body_mat(self.name))

    @abc.abstractmethod
    def is_alive(self) -> bool:
        pass
