from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import jax
import jax.numpy as jnp
from archieve import get_dtype
from archieve.envs.agents import BaseAgentConfig

@dataclass(frozen=True)
class CarAgent(BaseAgentConfig):
    name: str = 'Car'
    action_size: int = 2
    agent_obs_size: int = 24
    keepout: float = 0.4
    xml_path: str = str(Path(__file__).with_name('car.xml'))
    geom_size: float = 0.14
    collision_geom_count: int = 4
    max_speed: float = 1.5

    def compute_agent_obs(self, data: Any) -> jnp.ndarray:
        dtype = get_dtype()
        raw = jnp.asarray(data.sensordata, dtype=dtype)
        (qw, qx, qy, qz) = (raw[0], raw[1], raw[2], raw[3])
        mat = jnp.array([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy), 2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx), 2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)], dtype=dtype)
        return jnp.concatenate([mat, raw[4:]])

    def compute_heading(self, data: Any) -> jnp.ndarray:
        dtype = get_dtype()
        qw, qx, qy, qz = jnp.asarray(data.qpos[3:7], dtype=dtype)
        return jnp.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    def randomize_init_qpos(self, data: Any, rng: Any) -> Any:
        if not self.randomize_heading:
            return data
        theta = jax.random.uniform(rng, (), minval=0.0, maxval=2.0 * jnp.pi, dtype=get_dtype())
        half = 0.5 * theta
        quat = jnp.array([jnp.cos(half), 0.0, 0.0, jnp.sin(half)], dtype=get_dtype())
        return data.replace(qpos=data.qpos.at[3:7].set(quat))
