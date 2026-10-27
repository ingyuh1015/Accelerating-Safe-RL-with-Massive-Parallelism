from __future__ import annotations
import jax.numpy as jnp

from ...bases.base_velocity_task import BaseVelocityTask

THRESHOLDS = {0: 0.37315, 1: 0.7402}


class HopperVelocity(BaseVelocityTask):
    xml_name = 'hopper'
    frame_skip = 4
    velocity_threshold = THRESHOLDS[0]

    forward_reward_weight = 1.0
    ctrl_cost_weight = 1e-3
    healthy_reward = 1.0
    terminate_when_unhealthy = True
    reset_noise_scale = 5e-3
    exclude_x_in_obs = True


    healthy_z_min = 0.7
    healthy_z_max = float('inf')
    healthy_angle = 0.2

    def is_healthy(self, data) -> jnp.ndarray:
        z = data.qpos[1]
        angle = data.qpos[2]
        state = jnp.concatenate([data.qpos[2:], data.qvel])
        state_ok = jnp.all(jnp.abs(state) < 100.0)
        z_ok = (z > self.healthy_z_min)
        angle_ok = (jnp.abs(angle) < self.healthy_angle)
        finite = jnp.all(jnp.isfinite(data.qpos)) & jnp.all(jnp.isfinite(data.qvel))
        return (state_ok & z_ok & angle_ok & finite).astype(jnp.float32)
