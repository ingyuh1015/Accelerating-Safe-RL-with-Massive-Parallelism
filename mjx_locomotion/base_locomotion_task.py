from __future__ import annotations

import abc

import numpy as np
import jax
import jax.numpy as jnp

from .physics import (physics_step, mjx_forward,
                      site_world_velocity)
from .state import State


def _yaw_from_mat(mat):

    return jnp.arctan2(-mat[0, 1], mat[0, 0])


def _rz(theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class BaseLocomotionTask(abc.ABC):

    n_rewards: int = 2
    n_costs: int = 3
    num_legs: int = 4
    num_history: int = 3
    reward_names = ('command', 'power')
    cost_names = ('body_angle', 'com_height', 'foot_timing')

    def __init__(self, world, max_episode_steps: int = 1000,
                 is_earlystop: bool = False,
                 lin_vel_cmd_range=None, ang_vel_cmd_range=None):

        self.world = world
        self.mjx_model = world.mjx_model
        self._nq = world.nq
        self._nv = world.nv
        self._nu = world.action_dim
        from .physics import make_init_data
        self._init_data = (world.init_data if hasattr(world, 'init_data')
                           else make_init_data(world.mjx_model))

        self.max_episode_steps = int(max_episode_steps)
        self.num_steps = self.max_episode_steps
        self.is_earlystop = bool(is_earlystop)
        self.is_variable_episode = self.is_earlystop
        self._configure()
        self.action_dim = self._nu
        self.frame_skip = self.n_substeps


        if lin_vel_cmd_range is not None:
            self.lin_vel_cmd_range = (float(lin_vel_cmd_range[0]),
                                      float(lin_vel_cmd_range[1]))
        if ang_vel_cmd_range is not None:
            self.ang_vel_cmd_range = (float(ang_vel_cmd_range[0]),
                                      float(ang_vel_cmd_range[1]))
        assert self.lin_vel_cmd_range[0] <= self.lin_vel_cmd_range[1]
        assert self.ang_vel_cmd_range[0] <= self.ang_vel_cmd_range[1]


        gen0 = np.asarray(self.generator.joint_targets(0.0))
        lo = np.asarray(self.lower_limits)
        hi = np.asarray(self.upper_limits)
        self.action_bounds = np.maximum(
            np.abs(lo - gen0), np.abs(hi - gen0)).astype(np.float32)


    @abc.abstractmethod
    def _configure(self):

        raise NotImplementedError

    def _joint_pos(self, data):
        return data.qpos[self._joint_qposadr]

    def _joint_vel(self, data):
        return data.qvel[self._joint_dofadr]


    def _heading(self, data):
        base_mat = data.site_xmat[self.site_id].reshape(3, 3)
        yaw = _yaw_from_mat(base_mat)
        rot_mat = _rz(yaw)
        return base_mat, rot_mat

    def _gravity_vec(self, data, base_mat, rot_mat):
        g_world = base_mat @ self.gravity
        return rot_mat.T @ g_world


    def _contacts(self, data):

        contact_query = getattr(self.world, 'contact_query', None)
        if contact_query is not None:
            return contact_query(data, self.toe_geom_ids, use_margin=True).astype(jnp.float32)
        contact = data._impl.contact
        g = contact.geom
        dist = contact.dist
        active = dist < contact.includemargin
        toe_ids = self.toe_geom_ids

        def per_toe(toe_id):
            involves = (g[:, 0] == toe_id) | (g[:, 1] == toe_id)
            differ = g[:, 0] != g[:, 1]
            hit = jnp.any(involves & differ & active)
            return hit.astype(jnp.float32)

        return jax.vmap(per_toe)(toe_ids)


    def _raw_fields(self, data, info):

        base_mat, rot_mat = self._heading(data)
        gravity_vec = self._gravity_vec(data, base_mat, rot_mat)

        lin_w, ang_w = site_world_velocity(
            data, self.site_id, self.site_bodyid, self.body_rootid)
        base_lin_vel = rot_mat.T @ lin_w
        base_ang_vel = rot_mat.T @ ang_w

        joint_pos = self._joint_pos(data)
        joint_vel = self._joint_vel(data)
        base_height = data.site_xpos[self.site_id, 2]

        phase = self.generator.phase_list(info['gen_t'])
        contact = self._contacts(data)
        return dict(
            gravity_vec=gravity_vec, base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel, joint_pos=joint_pos, joint_vel=joint_vel,
            base_height=base_height, phase=phase, contact=contact,
        )

    def _observation(self, raw, info):

        hist_pos = info['hist_pos'].reshape(-1)
        hist_vel = info['hist_vel'].reshape(-1)
        hist_tgt = info['hist_tgt'].reshape(-1)
        return jnp.concatenate([
            info['cmd_lin_vel'],
            info['cmd_ang_vel'],
            raw['gravity_vec'],
            raw['base_lin_vel'],
            raw['base_ang_vel'],
            raw['joint_pos'],
            raw['joint_vel'],
            raw['phase'],
            hist_pos,
            hist_vel,
            hist_tgt,
            raw['contact'],
            jnp.atleast_1d(raw['base_height']),
        ])

    @property
    def obs_dim(self) -> int:
        nu = self._nu
        return (3 + 3 + 3 + 3 + 3 + nu + nu + 2 * self.num_legs
                + 3 * self.num_history * nu + self.num_legs + 1)


    def _rewards(self, raw, info, power):
        ang_vel_error = (info['cmd_ang_vel'][2] - raw['base_ang_vel'][2]) ** 2
        lin_vel_error = jnp.sum(
            jnp.square(raw['base_lin_vel'][:2] - info['cmd_lin_vel'][:2]))
        return jnp.stack([-(ang_vel_error + lin_vel_error), -power])

    def _costs(self, raw):

        a = -jnp.cos(jnp.deg2rad(self.tilt_angle_deg))
        gz = raw['gravity_vec'][2] / jnp.linalg.norm(raw['gravity_vec'])
        tilt = (gz > a).astype(jnp.float32)


        height = (raw['base_height'] < self.height_threshold).astype(jnp.float32)


        phase = raw['phase']
        sin_phase = phase[1::2]
        contact = raw['contact']
        swing = sin_phase < 0.0
        swing_cost = jnp.where(contact > 0.5, 1.0, 0.0)
        stance_cost = jnp.where(contact > 0.5, 0.0, 1.0)
        per_leg = jnp.where(swing, swing_cost, stance_cost)
        gait = jnp.sum(per_leg) / self.num_legs
        return jnp.stack([tilt, height, gait])


    def _ema_target(self, action, info, gen_t):

        gen_targets = self.generator.joint_targets(gen_t)
        joint_targets = jnp.clip(action + gen_targets,
                                 self.lower_limits, self.upper_limits)
        ema = info['ema_action'] * self.action_weight \
            + joint_targets * (1.0 - self.action_weight)
        return ema

    def _advance(self, data, ema):
        data = physics_step(self.mjx_model, data, ema, self.n_substeps)
        data = mjx_forward(self.mjx_model, data)
        power = jnp.sum(jnp.abs(data.actuator_force * self._joint_vel(data)))
        return data, power


    @staticmethod
    def _push(buf, item):

        return jnp.concatenate([buf[1:], item[None]], axis=0)


    def reset(self, key) -> State:
        k_quat, k_joint, k_lin, k_ang, k_state = jax.random.split(key, 5)


        init_base_quat = jnp.asarray(self.init_base_quat) + \
            jax.random.uniform(k_quat, (4,), minval=-1.0, maxval=1.0) \
            * self.init_base_quat_noise
        init_base_quat = init_base_quat / jnp.linalg.norm(init_base_quat)

        qpos = self._init_data.qpos
        qpos = qpos.at[0:3].set(jnp.asarray(self.init_base_pos))
        qpos = qpos.at[3:7].set(init_base_quat)

        nominal = self.generator.joint_targets(0.0)
        joint_init = nominal + jax.random.uniform(
            k_joint, (self._nu,), minval=-1.0, maxval=1.0) * self.init_joint_noise
        joint_init = jnp.clip(joint_init, self.lower_limits, self.upper_limits)
        qpos = qpos.at[7:].set(joint_init)

        qvel = jnp.zeros((self._nv,))
        data = self._init_data.replace(qpos=qpos, qvel=qvel)
        data = mjx_forward(self.mjx_model, data)

        cmd_lin_vel = jnp.array([
            jax.random.uniform(k_lin, (), minval=self.lin_vel_cmd_range[0],
                               maxval=self.lin_vel_cmd_range[1]), 0.0, 0.0])
        cmd_ang_vel = jnp.array([
            0.0, 0.0,
            jax.random.uniform(k_ang, (), minval=self.ang_vel_cmd_range[0],
                               maxval=self.ang_vel_cmd_range[1])])

        joint_pos_now = self._joint_pos(data)
        joint_vel_now = self._joint_vel(data)
        hist_pos = jnp.broadcast_to(joint_init - joint_pos_now, (self.num_history, self._nu))
        hist_vel = jnp.broadcast_to(joint_vel_now, (self.num_history, self._nu))
        hist_tgt = jnp.broadcast_to(joint_init, (self.num_history, self._nu))

        info = {
            'step': jnp.array(0.0),
            'gen_t': jnp.array(0.0),
            'ema_action': jnp.zeros((self._nu,)),
            'cmd_lin_vel': cmd_lin_vel,
            'cmd_ang_vel': cmd_ang_vel,
            'hist_pos': hist_pos,
            'hist_vel': hist_vel,
            'hist_tgt': hist_tgt,
            'is_terminated': jnp.array(0.0),
        }
        raw = self._raw_fields(data, info)
        obs = self._observation(raw, info)
        info['term_obs'] = obs
        info['term_reward'] = jnp.zeros((self.n_rewards,))
        info['term_cost'] = jnp.zeros((self.n_costs,))

        return State(
            data=data, obs=obs,
            reward=jnp.zeros((self.n_rewards,)),
            cost=jnp.zeros((self.n_costs,)),
            done=jnp.array(0.0), key=k_state, info=info,
        )


    def step(self, state: State, action: jax.Array) -> State:
        info = dict(state.info)
        was_terminated = info['is_terminated'] > 0.5


        new_step = info['step'] + 1.0
        gen_t = new_step * self.dt


        ema = self._ema_target(action, info, gen_t)
        data_live, power = self._advance(state.data, ema)

        live_info = {**info, 'gen_t': gen_t, 'ema_action': ema}
        raw_live = self._raw_fields(data_live, live_info)


        jp = self._joint_pos(data_live)
        jv = self._joint_vel(data_live)
        hist_pos = self._push(info['hist_pos'], ema - jp)
        hist_vel = self._push(info['hist_vel'], jv)
        hist_tgt = self._push(info['hist_tgt'], ema)
        live_info = {**live_info, 'hist_pos': hist_pos,
                     'hist_vel': hist_vel, 'hist_tgt': hist_tgt}

        obs_live = self._observation(raw_live, live_info)
        reward_live = self._rewards(raw_live, live_info, power)
        cost_live = self._costs(raw_live)


        gz = raw_live['gravity_vec'][2] / jnp.linalg.norm(raw_live['gravity_vec'])
        fell_now = gz >= 0.0


        select_data = getattr(self.world, 'select_data', None)
        if select_data is not None:
            data_next = select_data(was_terminated, state.data, data_live)
        else:
            data_next = jax.tree.map(
                lambda froz, liv: jnp.where(was_terminated, froz, liv),
                state.data, data_live)

        obs = jnp.where(was_terminated, info['term_obs'], obs_live)
        reward = jnp.where(was_terminated, info['term_reward'], reward_live)
        cost = jnp.where(was_terminated, info['term_cost'], cost_live)


        newly = (~was_terminated) & fell_now
        is_terminated = jnp.where(was_terminated | fell_now, 1.0, 0.0)


        def freeze(froz, liv):
            return jnp.where(was_terminated, froz, liv)

        info_out = {
            'step': new_step,
            'gen_t': freeze(info['gen_t'], gen_t),
            'ema_action': freeze(info['ema_action'], ema),
            'cmd_lin_vel': info['cmd_lin_vel'],
            'cmd_ang_vel': info['cmd_ang_vel'],
            'hist_pos': jax.tree.map(freeze, info['hist_pos'], hist_pos),
            'hist_vel': jax.tree.map(freeze, info['hist_vel'], hist_vel),
            'hist_tgt': jax.tree.map(freeze, info['hist_tgt'], hist_tgt),
            'is_terminated': is_terminated,
            'term_obs': jnp.where(newly, obs_live,
                                  jnp.where(was_terminated, info['term_obs'], obs_live)),
            'term_reward': jnp.where(newly, reward_live,
                                     jnp.where(was_terminated, info['term_reward'], reward_live)),
            'term_cost': jnp.where(newly, cost_live,
                                   jnp.where(was_terminated, info['term_cost'], cost_live)),
        }

        done = is_terminated if self.is_earlystop else jnp.array(0.0)

        return State(data=data_next, obs=obs, reward=reward, cost=cost,
                     done=done, key=state.key, info=info_out)
