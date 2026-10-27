from __future__ import annotations
import jax.numpy as jnp

from ....bases.underlying import agent_xy
from .circle_level0 import CircleLevel0


class CircleLevel1(CircleLevel0):
    n_costs = 1
    n_sigwalls = 2

    def cost_vector(self, data, info) -> jnp.ndarray:
        xy = agent_xy(data, self._xy_adr)
        out = jnp.abs(xy[0]) > self.sigwall_locate
        if self.n_sigwalls == 4:
            out = out | (jnp.abs(xy[1]) > self.sigwall_locate)
        return jnp.array([out.astype(jnp.float32)])
