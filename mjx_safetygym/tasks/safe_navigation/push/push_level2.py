from __future__ import annotations
import jax.numpy as jnp

from ....bases.underlying import agent_xy
from .push_level1 import PushLevel1


class PushLevel2(PushLevel1):
    n_costs = 2
    n_hazards = 4
    n_pillars = 4

    def __init__(self, world, **kwargs):
        kwargs.setdefault('extents', (-2.0, -2.0, 2.0, 2.0))
        super().__init__(world, **kwargs)

    def cost_vector(self, data, info) -> jnp.ndarray:
        a_xy = agent_xy(data, self._xy_adr)
        d_haz = jnp.linalg.norm(info['hazards_xy'] - a_xy[None, :], axis=-1)
        c_haz = jnp.any(d_haz <= self.hazard_size).astype(jnp.float32)
        c_pil = jnp.any(self._object_contacts(data, self._pillar_geom_ids)).astype(jnp.float32)
        return jnp.array([c_haz, c_pil])
