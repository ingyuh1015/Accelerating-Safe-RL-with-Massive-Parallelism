from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import jax.numpy as jnp
from archieve import get_dtype

@dataclass(frozen=True)
class BaseAgentConfig:
    name: str
    action_size: int
    agent_obs_size: int = 0
    n_substeps: int = 10
    action_scale: float = 1.0
    keepout: float = 0.4
    geom_size: float = 0.1
    collision_geom_count: int = 1
    max_speed: float = 1.0
    randomize_heading: bool = True

    def compute_agent_obs(self, data: Any) -> jnp.ndarray:
        raise NotImplementedError

    def compute_heading(self, data: Any) -> jnp.ndarray:
        raise NotImplementedError

    def randomize_init_qpos(self, data: Any, rng: Any) -> Any:
        return data

    def compute_state_info(self, data: Any, info: dict[str, Any], *, ctrl_dt: float) -> None:
        dtype = get_dtype()
        agent_position = jnp.asarray(data.qpos[:2], dtype=get_dtype())
        delta = agent_position - info['agent_pos']
        move_dist = jnp.linalg.norm(delta)
        info['agent_pos'] = agent_position
        info['agent_vel_xy'] = data.qvel[:2].astype(dtype)
        info['move_dist'] = move_dist
        info['agent_speed'] = move_dist / jnp.maximum(jnp.asarray(ctrl_dt, dtype=dtype), jnp.asarray(1e-06, dtype=dtype))
