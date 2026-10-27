from __future__ import annotations
import jax.numpy as jnp

from ...bases.base_velocity_task import BaseVelocityTask

THRESHOLDS = {0: 1.7075, 1: 2.3415}


class Walker2dVelocity(BaseVelocityTask):
    xml_name = 'walker2d'
    frame_skip = 4
    velocity_threshold = THRESHOLDS[0]

    forward_reward_weight = 1.0
    ctrl_cost_weight = 1e-3
    healthy_reward = 1.0
    terminate_when_unhealthy = True
    reset_noise_scale = 5e-3
    exclude_x_in_obs = True


    healthy_z_min = 0.8
    healthy_z_max = 2.0
    healthy_angle = 1.0

    def is_healthy(self, data) -> jnp.ndarray:
        z = data.qpos[1]
        angle = data.qpos[2]
        z_ok = (z > self.healthy_z_min) & (z < self.healthy_z_max)
        angle_ok = (jnp.abs(angle) < self.healthy_angle)
        finite = jnp.all(jnp.isfinite(data.qpos)) & jnp.all(jnp.isfinite(data.qvel))
        return (z_ok & angle_ok & finite).astype(jnp.float32)
