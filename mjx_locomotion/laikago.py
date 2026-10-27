from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .base_locomotion_task import BaseLocomotionTask
from .generator import LaikagoGenerator


class LaikagoTask(BaseLocomotionTask):
    n_rewards = 2
    n_costs = 3
    num_legs = 4
    num_history = 3

    def _configure(self):
        w = self.world
        nl = self.num_legs


        self.sim_dt = float(w.timestep)
        self.n_substeps = int(w.n_substeps)
        self.dt = self.sim_dt * self.n_substeps
        self.control_freq = 1.0 / self.dt


        self.lower_limits = jnp.asarray(
            np.array([-1.0, -3.0, -2.0] * nl, dtype=np.float32))
        self.upper_limits = jnp.asarray(
            np.array([1.0, 1.5, 2.5] * nl, dtype=np.float32))


        self.action_weight = 0.75


        self._joint_qposadr = jnp.arange(7, 7 + self._nu)
        self._joint_dofadr = jnp.arange(6, 6 + self._nu)


        self.init_base_pos = jnp.array([0.0, 0.0, 0.45])
        self.init_base_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        self.init_base_quat_noise = 0.1
        self.init_joint_noise = 0.1
        self.gravity = jnp.array([0.0, 0.0, -9.8])


        self.lin_vel_cmd_range = (-1.0, 2.0)
        self.ang_vel_cmd_range = (-0.5, 0.5)


        self.tilt_angle_deg = 15.0
        self.height_threshold = 0.35


        self.generator = LaikagoGenerator(
            foot_height=0.0, bottom=-0.4, abduct_length=0.037,
            thigh_length=0.25, knee_length=0.25, num_leg=nl, default_freq=10.0)


        self.site_id = int(w.site_ids['robot'])
        self.site_bodyid = int(w.mj_model.site_bodyid[self.site_id])
        self.body_rootid = int(w.mj_model.body_rootid[self.site_bodyid])
        toe_names = ['fr_toe', 'fl_toe', 'hr_toe', 'hl_toe']
        self.toe_geom_ids = jnp.asarray(
            np.array([w.geom_ids[n] for n in toe_names], dtype=np.int32))
        self.floor_geom_id = int(w.geom_ids['floor'])

        assert self._nq == 19 and self._nv == 18 and self._nu == 12, \
            f"unexpected model dims nq={self._nq} nv={self._nv} nu={self._nu}"
