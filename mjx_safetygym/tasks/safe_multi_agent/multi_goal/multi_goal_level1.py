from __future__ import annotations
import jax.numpy as jnp

from .multi_goal_level0 import MultiGoalLevel0


class MultiGoalLevel1(MultiGoalLevel0):
    n_rewards = 2
    n_costs = 2
    n_hazards = 8

    def __init__(self, world, **kwargs):
        kwargs.setdefault('extents', (-1.5, -1.5, 1.5, 1.5))
        super().__init__(world, **kwargs)

    def cost_vector(self, data, info) -> jnp.ndarray:
        haz = info['hazards_xy']
        a0 = self.agent0_xy(data)
        a1 = self.agent1_xy(data)
        d0 = jnp.linalg.norm(haz - a0[None, :], axis=-1)
        d1 = jnp.linalg.norm(haz - a1[None, :], axis=-1)
        c0 = jnp.sum((d0 <= self.hazard_size).astype(jnp.float32))
        c1 = jnp.sum((d1 <= self.hazard_size).astype(jnp.float32))
        return jnp.stack([c0, c1])
