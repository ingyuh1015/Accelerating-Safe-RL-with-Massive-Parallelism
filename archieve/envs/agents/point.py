from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import jax
import jax.numpy as jnp
from archieve import get_dtype
from archieve.envs.agents import BaseAgentConfig

@dataclass(frozen=True)
class PointAgent(BaseAgentConfig):
    name: str = 'Point'
    action_size: int = 2
    agent_obs_size: int = 12
    keepout: float = 0.4
    xml_path: str = str(Path(__file__).with_name('point.xml'))
    geom_size: float = 0.1
    max_speed: float = 1.5

    def compute_agent_obs(self, data: Any) -> jnp.ndarray:
        return jnp.asarray(data.sensordata, dtype=get_dtype())

    def compute_heading(self, data: Any) -> jnp.ndarray:
        return jnp.asarray(data.qpos[2], dtype=get_dtype())

    def randomize_init_qpos(self, data: Any, rng: Any) -> Any:
        if not self.randomize_heading:
            return data
        theta = jax.random.uniform(rng, (), minval=0.0, maxval=2.0 * jnp.pi, dtype=get_dtype())
        return data.replace(qpos=data.qpos.at[2].set(theta))
