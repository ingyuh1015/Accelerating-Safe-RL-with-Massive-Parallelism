from __future__ import annotations
import jax
import jax.numpy as jnp

from ....bases.base_task import BaseTask
from ....bases.underlying import agent_xy, agent_heading
from ....utils.lidar import pseudo_lidar


class CircleLevel0(BaseTask):
    n_rewards = 1
    n_costs = 1
    n_hazards = 0
    n_bins = 16
    lidar_max_dist: float = 6.0

    num_steps: int = 500
    circle_radius: float = 1.5
    reward_factor: float = 1e-1
    sigwall_locate: float = 1.125
    n_sigwalls: int = 0

    def __init__(self, world, **kwargs):
        kwargs.setdefault('extents', (-0.8, -0.8, 0.8, 0.8))
        super().__init__(world, **kwargs)


    def sample_layout(self, key) -> dict:
        return {'agent_xy': self._sample_xy(key, shrink=0.0),
                'goal_xy': jnp.zeros((2,), jnp.float32)}

    def is_goal_reached(self, data, info) -> jax.Array:
        return jnp.float32(0.0)

    def _on_goal_reached(self, key, data, info, reached) -> dict:
        return info

    def reward_vector(self, prev_dist, data, action, info) -> jnp.ndarray:
        xy = agent_xy(data, self._xy_adr)
        heading = self._heading(data)

        va = self._velocimeter_adr
        v_body = data.sensordata[va:va + 2]
        ch, sh = jnp.cos(heading), jnp.sin(heading)
        u = ch * v_body[0] - sh * v_body[1]
        v = sh * v_body[0] + ch * v_body[1]
        x, y = xy[0], xy[1]
        radius = jnp.sqrt(x * x + y * y)
        r_circle = ((-u * y + v * x)
                    / (1.0 + jnp.abs(radius - self.circle_radius))) * self.reward_factor
        return jnp.stack([r_circle])

    def cost_vector(self, data, info) -> jnp.ndarray:
        return jnp.zeros((self.n_costs,), dtype=jnp.float32)


    def observation(self, data, info) -> jnp.ndarray:
        a_xy = agent_xy(data, self._xy_adr)
        heading = self._heading(data)
        sensors = self._sensor_obs(data)
        circle_lidar = pseudo_lidar(a_xy, heading, jnp.zeros((1, 2), jnp.float32),
                                    n_bins=self.n_bins, max_dist=self.lidar_max_dist)
        return jnp.concatenate([sensors, circle_lidar])

    @property
    def obs_dim(self) -> int:
        return self._sensor_obs_dim + self.n_bins
