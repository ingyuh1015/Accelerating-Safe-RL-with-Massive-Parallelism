from __future__ import annotations
import jax.numpy as jnp

from ...bases.base_velocity_task import BaseVelocityTask

THRESHOLDS = {0: 2.5745, 1: 2.6222}


class AntVelocity(BaseVelocityTask):
    xml_name = 'ant'
    frame_skip = 5
    velocity_threshold = THRESHOLDS[0]

    forward_reward_weight = 1.0
    ctrl_cost_weight = 0.5
    healthy_reward = 1.0
    terminate_when_unhealthy = True
    reset_noise_scale = 5e-3
    exclude_x_in_obs = True
    use_planar_speed = True


    n_exclude_qpos = 2


    healthy_z_min = 0.2
    healthy_z_max = 1.0

    def _position(self, data) -> jnp.ndarray:
        return data.xpos[self.world.body_ids['torso'], :2]

    def observation(self, data) -> jnp.ndarray:
        return jnp.concatenate(
            [data.qpos[self.n_exclude_qpos:], jnp.clip(data.qvel, -10.0, 10.0)])

    @property
    def obs_dim(self) -> int:
        return (self._nq - self.n_exclude_qpos) + self._nv

    def is_healthy(self, data) -> jnp.ndarray:
        z = data.qpos[2]
        z_ok = (z >= self.healthy_z_min) & (z <= self.healthy_z_max)
        finite = jnp.all(jnp.isfinite(data.qpos)) & jnp.all(jnp.isfinite(data.qvel))
        return (z_ok & finite).astype(jnp.float32)
