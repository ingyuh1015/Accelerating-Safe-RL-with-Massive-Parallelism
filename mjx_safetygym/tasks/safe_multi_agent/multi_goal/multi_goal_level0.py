from __future__ import annotations
import jax.numpy as jnp

from ....bases.base_multi_task import BaseMultiTask


class MultiGoalLevel0(BaseMultiTask):
    n_rewards = 2
    n_costs = 2
    n_hazards = 0

    reward_distance: float = 1.0
    reward_goal: float = 1.0

    def __init__(self, world, **kwargs):
        kwargs.setdefault('extents', (-1.0, -1.0, 1.0, 1.0))
        super().__init__(world, **kwargs)

    def cost_vector(self, data, info) -> jnp.ndarray:
        return jnp.zeros((self.n_costs,), dtype=jnp.float32)
