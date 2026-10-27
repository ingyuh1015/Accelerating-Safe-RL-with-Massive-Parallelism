from __future__ import annotations
import jax.numpy as jnp

from ....bases.underlying import agent_xy
from .push_level0 import PushLevel0


class PushLevel1(PushLevel0):
    n_rewards = 1
    n_costs = 1
    n_hazards = 2
    n_pillars = 1

    def __init__(self, world, **kwargs):
        kwargs.setdefault('extents', (-1.5, -1.5, 1.5, 1.5))
        kwargs.setdefault('hazard_size', 0.3)
        super().__init__(world, **kwargs)

    def cost_vector(self, data, info) -> jnp.ndarray:
        a_xy = agent_xy(data, self._xy_adr)
        d_haz = jnp.linalg.norm(info['hazards_xy'] - a_xy[None, :], axis=-1)
        c_haz = jnp.any(d_haz <= self.hazard_size).astype(jnp.float32)
        return jnp.array([c_haz])
