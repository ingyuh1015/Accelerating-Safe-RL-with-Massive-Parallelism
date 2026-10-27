from __future__ import annotations
import jax.numpy as jnp

from ....bases.base_task import BaseTask
from ....bases.underlying import agent_xy


class GoalLevel0(BaseTask):
    n_rewards = 1
    n_costs = 1
    n_hazards = 0

    reward_distance: float = 1.0
    reward_goal: float = 1.0

    def __init__(self, world, **kwargs):
        kwargs.setdefault('extents', (-1.0, -1.0, 1.0, 1.0))
        super().__init__(world, **kwargs)

    def reward_vector(self, prev_dist, data, action, info) -> jnp.ndarray:
        cur_dist = self.dist_goal(data, info)
        progress = (prev_dist - cur_dist) * self.reward_distance
        reached = self.is_goal_reached(data, info)
        r_goal = progress + reached * self.reward_goal
        return jnp.stack([r_goal])

    def cost_vector(self, data, info) -> jnp.ndarray:
        return jnp.zeros((self.n_costs,), dtype=jnp.float32)
