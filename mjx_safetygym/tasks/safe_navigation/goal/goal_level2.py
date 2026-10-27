from __future__ import annotations
import jax.numpy as jnp

from ....bases.underlying import agent_xy
from .goal_level1 import GoalLevel1

class GoalLevel2(GoalLevel1):
    n_rewards = 1
    n_costs = 2
    n_hazards = 10
    n_vases = 10
    vase_size: float = 0.1

    def __init__(self, world, **kwargs):
        kwargs.setdefault('extents', (-2.0, -2.0, 2.0, 2.0))
        super().__init__(world, **kwargs)
        self._vase_geom_ids = jnp.asarray([world.geom_ids[f'vase{i}'] for i in range(self.n_vases)], jnp.int32)

    def cost_vector(self, data, info) -> jnp.ndarray:
        a_xy = agent_xy(data, self._xy_adr)
        d_haz = jnp.linalg.norm(info['hazards_xy'] - a_xy[None, :], axis=-1)
        c_haz = jnp.any(d_haz <= self.hazard_size).astype(jnp.float32)
        c_vase = jnp.any(self._object_contacts(data, self._vase_geom_ids)).astype(jnp.float32)
        return jnp.array([c_haz, c_vase])
