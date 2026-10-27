from __future__ import annotations
from typing import Any
import jax.numpy as jnp
from archieve import get_dtype
from archieve.envs.tasks import NavEnv
_dtype = get_dtype

class CircleEnv(NavEnv):

    def __init__(self, *, reward_factor: float=0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._circle_radius = float(self.objects['circle'].radius)
        self._reward_factor = float(reward_factor)

    def reset_objects(self, data: Any, info: dict, rng: Any) -> Any:
        agent_pos = info['agent_pos']
        agent_dist = jnp.linalg.norm(agent_pos)
        safe_dist = jnp.maximum(agent_dist, 1e-06)
        info['target_pos'] = agent_pos * (self._circle_radius / safe_dist)
        info['goal_distance'] = jnp.abs(agent_dist - self._circle_radius)
        return super().reset_objects(data, info, rng)

    def compute_reward(self, data: Any, info: dict) -> Any:
        dtype = _dtype()
        pos = jnp.asarray(info['agent_pos'], dtype=dtype)
        vel = jnp.asarray(info['agent_vel_xy'], dtype=dtype)
        r = jnp.asarray(self._circle_radius, dtype=dtype)
        dist = jnp.linalg.norm(pos)
        safe_dist = jnp.maximum(dist, jnp.asarray(1e-06, dtype=dtype))
        sweep = -vel[0] * pos[1] + vel[1] * pos[0]
        radial_dev = jnp.abs(dist - r)
        info['target_pos'] = pos * (r / safe_dist)
        info['goal_distance'] = radial_dev
        info['reward'] = sweep / (jnp.asarray(1.0, dtype=dtype) + radial_dev) * jnp.asarray(self._reward_factor, dtype=dtype)
        return data
