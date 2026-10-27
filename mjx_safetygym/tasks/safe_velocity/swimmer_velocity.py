from __future__ import annotations

from ...bases.base_velocity_task import BaseVelocityTask

THRESHOLDS = {0: 0.04845, 1: 0.2282}


class SwimmerVelocity(BaseVelocityTask):
    xml_name = 'swimmer'
    frame_skip = 4
    velocity_threshold = THRESHOLDS[0]

    forward_reward_weight = 1.0
    ctrl_cost_weight = 1e-4
    healthy_reward = 0.0
    terminate_when_unhealthy = False
    reset_noise_scale = 5e-3
    exclude_x_in_obs = True
    use_planar_speed = True
