from __future__ import annotations
import jax.numpy as jnp

from ....bases.underlying import agent_xy
from .goal_level0 import GoalLevel0


class GoalLevel1(GoalLevel0):
    n_rewards = 1
    n_costs = 1
    n_hazards = 8
    n_vases = 1

    def __init__(self, world, **kwargs):
        kwargs.setdefault('extents', (-1.5, -1.5, 1.5, 1.5))
        super().__init__(world, **kwargs)

    def cost_vector(self, data, info) -> jnp.ndarray:


        a_xy = agent_xy(data, self._xy_adr)
        d = jnp.linalg.norm(info['hazards_xy'] - a_xy[None, :], axis=-1)
        raw = jnp.sum(jnp.where(d <= self.hazard_size, self.hazard_size - d, 0.0))
        cost = (raw > 0.0).astype(jnp.float32)
        return jnp.array([cost])
