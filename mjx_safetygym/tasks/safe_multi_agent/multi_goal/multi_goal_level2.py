from __future__ import annotations
import jax
import jax.numpy as jnp

from ....utils.lidar import pseudo_lidar
from ....bases.base_multi_task import (
    AGENT0_XY_ADR, AGENT0_HEADING_ADR, AGENT1_XY_ADR, AGENT1_HEADING_ADR,
)
from .multi_goal_level1 import MultiGoalLevel1


class MultiGoalLevel2(MultiGoalLevel1):
    n_rewards = 2
    n_costs = 2
    n_hazards = 10
    n_vases = 10
    vase_size: float = 0.1

    def __init__(self, world, **kwargs):
        kwargs.setdefault('extents', (-2.0, -2.0, 2.0, 2.0))
        super().__init__(world, **kwargs)

    def sample_layout(self, key) -> dict:
        k_g0, k_g1, k_haz, k_vase = jax.random.split(key, 4)
        goal0_xy = self._sample_xy(k_g0)
        goal1_xy = self._sample_xy(k_g1)
        hazards_xy = jax.vmap(self._sample_xy)(jax.random.split(k_haz, self.n_hazards))
        vases_xy = jax.vmap(self._sample_xy)(jax.random.split(k_vase, self.n_vases))
        return {'goal0_xy': goal0_xy, 'goal1_xy': goal1_xy,
                'hazards_xy': hazards_xy, 'vases_xy': vases_xy}

    def cost_vector(self, data, info) -> jnp.ndarray:
        haz = info['hazards_xy']
        vase = info['vases_xy']
        a0 = self.agent0_xy(data)
        a1 = self.agent1_xy(data)
        c0 = jnp.sum((jnp.linalg.norm(haz - a0[None, :], axis=-1) <= self.hazard_size)
                     .astype(jnp.float32)) \
            + jnp.sum((jnp.linalg.norm(vase - a0[None, :], axis=-1) <= self.vase_size)
                      .astype(jnp.float32))
        c1 = jnp.sum((jnp.linalg.norm(haz - a1[None, :], axis=-1) <= self.hazard_size)
                     .astype(jnp.float32)) \
            + jnp.sum((jnp.linalg.norm(vase - a1[None, :], axis=-1) <= self.vase_size)
                      .astype(jnp.float32))
        return jnp.stack([c0, c1])

    def _agent_vase_lidar(self, data, xy_adr, heading_adr, info) -> jnp.ndarray:
        a_xy = self._xy(data, xy_adr)
        heading = data.qpos[heading_adr]
        return pseudo_lidar(a_xy, heading, info['vases_xy'],
                            n_bins=self.n_bins, max_dist=self.lidar_max_dist)

    def observation(self, data, info) -> jnp.ndarray:

        o0 = self._agent_obs(data, AGENT0_XY_ADR, AGENT0_HEADING_ADR,
                             info['goal0_xy'], info)
        o1 = self._agent_obs(data, AGENT1_XY_ADR, AGENT1_HEADING_ADR,
                             info['goal1_xy'], info)
        v0 = self._agent_vase_lidar(data, AGENT0_XY_ADR, AGENT0_HEADING_ADR, info)
        v1 = self._agent_vase_lidar(data, AGENT1_XY_ADR, AGENT1_HEADING_ADR, info)

        return jnp.concatenate([o0, v0, o1, v1])

    @property
    def per_agent_obs_dim(self) -> int:
        return super().per_agent_obs_dim + self.n_bins

    @property
    def obs_dim(self) -> int:
        return 2 * self.per_agent_obs_dim
