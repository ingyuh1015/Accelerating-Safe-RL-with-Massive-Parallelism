from __future__ import annotations
import jax.numpy as jnp

from ....bases.underlying import agent_xy
from .button_level0 import ButtonLevel0


class ButtonLevel1(ButtonLevel0):
    n_rewards = 1
    n_costs = 3
    n_hazards = 4
    n_buttons = 4
    n_gremlins = 4

    def __init__(self, world, **kwargs):
        kwargs.setdefault('extents', (-1.5, -1.5, 1.5, 1.5))
        super().__init__(world, **kwargs)

    def cost_vector(self, data, info) -> jnp.ndarray:
        a_xy = agent_xy(data, self._xy_adr)

        wrong = self._object_contacts(data, self._button_geom_ids) & (info['goal_onehot'] < 0.5)
        c_btn = (jnp.any(wrong) & (info['btn_timer'] == 0.0)).astype(jnp.float32)
        d_haz = jnp.linalg.norm(info['hazards_xy'] - a_xy[None, :], axis=-1)
        c_haz = jnp.any(d_haz <= self.hazard_size).astype(jnp.float32)
        c_gr = jnp.any(self._object_contacts(data, self._gremlin_geom_ids)).astype(jnp.float32)
        return jnp.array([c_btn, c_haz, c_gr])
