from __future__ import annotations

from ...bases.base_velocity_task import BaseVelocityTask

THRESHOLDS = {0: 2.8795, 1: 3.2096}


class HalfCheetahVelocity(BaseVelocityTask):
    xml_name = 'half_cheetah'
    frame_skip = 5
    velocity_threshold = THRESHOLDS[0]

    forward_reward_weight = 1.0
    ctrl_cost_weight = 0.1
    healthy_reward = 0.0
    terminate_when_unhealthy = False
    reset_noise_scale = 5e-3
    exclude_x_in_obs = True
