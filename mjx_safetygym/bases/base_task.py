from __future__ import annotations
import abc
from typing import Tuple

import jax
import jax.numpy as jnp
from mujoco import mjx

from .underlying import (State, physics_step, agent_xy, agent_heading,
                         agent_vel_xy, make_init_data, quat_to_mat)
from ..utils.lidar import pseudo_lidar, compass


class BaseTask(abc.ABC):

    n_rewards: int = 1
    n_costs: int = 1
    n_hazards: int = 0
    n_vases: int = 0
    n_bins: int = 16
    lidar_max_dist: float = 3.0

    def __init__(self, world, *, extents=(-1.0, -1.0, 1.0, 1.0),
                 goal_size: float = 0.3, hazard_size: float = 0.2,
                 ctrl_cost_weight: float = 0.01, action_scale: float = 1.0,
                 continue_goal: bool = True):
        self.world = world
        self.mjx_model = world.mjx_model

        if self.n_vases > 0 and len(getattr(world, 'vase_qpos_adr', ())) != self.n_vases:
            raise ValueError(
                f'{type(self).__name__} needs {self.n_vases} vase bodies in the world '
                f'model but it has {len(getattr(world, "vase_qpos_adr", ()))} — build '
                f'the world with build_world(..., n_vases={self.n_vases}).')
        self._vase_qpos_adr = tuple(getattr(world, 'vase_qpos_adr', ()))
        self._vase_body_ids = jnp.asarray(getattr(world, 'vase_body_ids', ()), dtype=jnp.int32) \
            if self.n_vases > 0 else None
        self.extents = jnp.asarray(extents, dtype=jnp.float32)
        self.goal_size = float(goal_size)
        self.hazard_size = float(hazard_size)
        self.ctrl_cost_weight = float(ctrl_cost_weight)
        self.action_scale = float(action_scale)
        self.continue_goal = bool(continue_goal)


        self.agent_keepout = 0.4
        self.goal_keepout = 0.305
        self.hazard_keepout = 0.18
        self.vase_keepout = 0.15
        self.placements_margin = 0.0
        self.frame_skip = int(world.frame_skip)
        self._xy_adr = world.agent_layout.xy_qpos_adr
        self._heading_adr = world.agent_layout.heading_qpos_adr
        self._heading_mode = world.agent_layout.heading_mode


        self._obs_sensors = world.obs_sensors
        self._sensor_obs_dim = sum(out for *_unused, out in self._obs_sensors)
        self._velocimeter_adr = next(
            (adr for n, adr, *_ in self._obs_sensors if n == 'velocimeter'), 3)
        self._init_data = (world.init_data if hasattr(world, 'init_data')
                           else make_init_data(world.mjx_model))
        model = world.mj_model
        root = world.body_ids['agent']
        bodies = set()
        for body in range(model.nbody):
            parent = body
            while parent > 0 and parent != root:
                parent = int(model.body_parentid[parent])
            if parent == root:
                bodies.add(body)
        self._agent_geom_ids = jnp.asarray(
            [i for i, body in enumerate(model.geom_bodyid) if body in bodies], dtype=jnp.int32)

    def _object_contacts(self, data, geom_ids) -> jax.Array:

        contact_query = getattr(self.world, 'contact_query', None)
        if contact_query is not None:
            return contact_query(data, geom_ids, self._agent_geom_ids)
        contact = data.contact
        geoms = contact.geom
        active = (jnp.arange(contact.dist.shape[0]) < data.ncon) & (contact.dist <= 0.0)
        agent0 = jnp.any(geoms[:, 0, None] == self._agent_geom_ids, axis=-1)
        agent1 = jnp.any(geoms[:, 1, None] == self._agent_geom_ids, axis=-1)
        pairs = ((agent0[:, None] & (geoms[:, 1, None] == geom_ids))
                 | (agent1[:, None] & (geoms[:, 0, None] == geom_ids)))
        return jnp.any(pairs & active[:, None], axis=0)

    def _sample_xy(self, key, shrink: float = 0.0) -> jax.Array:


        lo = self.extents[:2] + shrink
        hi = self.extents[2:] - shrink
        return jax.random.uniform(key, shape=(2,), minval=lo, maxval=hi)

    def _sample_xy_k(self, key, k: int, shrink: float = 0.0) -> jax.Array:
        lo = self.extents[:2] + shrink
        hi = self.extents[2:] - shrink
        return jax.random.uniform(key, shape=(k, 2), minval=lo, maxval=hi)

    @staticmethod
    def _first_valid(cands, existing_xy, min_seps) -> jax.Array:


        d = jnp.linalg.norm(cands[:, None, :] - existing_xy[None, :, :], axis=-1)
        valid = jnp.all(d >= min_seps[None, :], axis=1)
        margin = jnp.min(d - min_seps[None, :], axis=1)
        idx = jnp.where(jnp.any(valid), jnp.argmax(valid), jnp.argmax(margin))
        return cands[idx]

    def _place_objects(self, key, n: int, keepout, fixed_xy, fixed_seps) -> jax.Array:


        if n <= 0:
            return jnp.zeros((max(n, 0), 2), dtype=jnp.float32)
        K = 64
        hh = 2.0 * keepout + self.placements_margin
        keys = jax.random.split(key, n)

        def place(carry, k):
            placed, i = carry
            cands = self._sample_xy_k(k, K, shrink=keepout)
            df = jnp.linalg.norm(cands[:, None, :] - fixed_xy[None, :, :], axis=-1)
            ok_f = jnp.all(df >= fixed_seps[None, :], axis=1)
            dp = jnp.linalg.norm(cands[:, None, :] - placed[None, :, :], axis=-1)
            live = (jnp.arange(n) < i)[None, :]
            ok_p = jnp.all((dp >= hh) | (~live), axis=1)
            valid = ok_f & ok_p

            margin = jnp.minimum(jnp.min(df - fixed_seps[None, :], axis=1),
                                 jnp.min(jnp.where(live, dp - hh, jnp.inf), axis=1))
            idx = jnp.where(jnp.any(valid), jnp.argmax(valid), jnp.argmax(margin))
            chosen = cands[idx]
            return (placed.at[i].set(chosen), i + 1), chosen

        init = (jnp.full((n, 2), 1e3, dtype=jnp.float32), 0)
        _, xy = jax.lax.scan(place, init, keys)
        return xy

    def sample_layout(self, key) -> dict:

        m = self.placements_margin
        ka, kg, kh, kv = jax.random.split(key, 4)
        agent_s = self._sample_xy(ka, shrink=self.agent_keepout)
        goal_xy = self._first_valid(
            self._sample_xy_k(kg, 64, shrink=self.goal_keepout), agent_s[None, :],
            jnp.array([self.agent_keepout + self.goal_keepout + m]))
        ag_gl = jnp.stack([agent_s, goal_xy])
        hazards_xy = self._place_objects(
            kh, self.n_hazards, self.hazard_keepout, ag_gl,
            jnp.array([self.agent_keepout + self.hazard_keepout + m,
                       self.goal_keepout + self.hazard_keepout + m]))
        if self.n_vases > 0:
            vase_fixed = jnp.concatenate([ag_gl, hazards_xy], axis=0)
            vase_seps = jnp.concatenate([
                jnp.array([self.agent_keepout + self.vase_keepout + m,
                           self.goal_keepout + self.vase_keepout + m]),
                jnp.full((self.n_hazards,), self.hazard_keepout + self.vase_keepout + m)])
            vases_xy = self._place_objects(kv, self.n_vases, self.vase_keepout, vase_fixed, vase_seps)
        else:
            vases_xy = jnp.zeros((max(self.n_vases, 0), 2), dtype=jnp.float32)
        return {'agent_xy': agent_s, 'goal_xy': goal_xy,
                'hazards_xy': hazards_xy, 'vases_xy': vases_xy}

    def _resample_goal_keepout(self, key, data, info) -> jax.Array:

        vase_indices = jnp.asarray(self._vase_qpos_adr, dtype=jnp.int32)[:, None] + jnp.arange(2)
        existing = jnp.concatenate([agent_xy(data, self._xy_adr)[None, :],
                                    info['hazards_xy'], data.qpos[vase_indices]], axis=0)
        min_seps = jnp.concatenate([
            jnp.array([self.agent_keepout + self.goal_keepout + self.placements_margin]),
            jnp.full((self.n_hazards,), self.hazard_keepout + self.goal_keepout + self.placements_margin),
            jnp.full((self.n_vases,), self.vase_keepout + self.goal_keepout + self.placements_margin),
        ])
        return self._first_valid(self._sample_xy_k(key, 64, shrink=self.goal_keepout), existing, min_seps)


    def dist_goal(self, data, info) -> jax.Array:
        return jnp.linalg.norm(agent_xy(data, self._xy_adr) - info['goal_xy'])

    def is_goal_reached(self, data, info) -> jax.Array:
        return (self.dist_goal(data, info) <= self.goal_size).astype(jnp.float32)

    def _heading(self, data) -> jax.Array:

        return agent_heading(data, self._heading_adr, self._heading_mode)

    def _sensor_obs(self, data) -> jax.Array:


        parts = []
        for _name, adr, dim, kind, _out in self._obs_sensors:
            raw = data.sensordata[adr:adr + dim]
            if kind == 'plain':
                parts.append(raw)
            elif kind == 'hinge_pos':
                parts.append(jnp.stack([jnp.sin(raw[0]), jnp.cos(raw[0])]))
            elif kind == 'ballquat':
                parts.append(quat_to_mat(raw).reshape(-1))
        return jnp.concatenate(parts)

    def _write_heading(self, qpos, head) -> jax.Array:


        if self._heading_mode == 'free':
            a = self._heading_adr
            return (qpos.at[a + 0].set(jnp.cos(head / 2.0))
                        .at[a + 1].set(0.0)
                        .at[a + 2].set(0.0)
                        .at[a + 3].set(jnp.sin(head / 2.0)))
        return qpos.at[self._heading_adr].set(head)

    def observation(self, data, info) -> jax.Array:


        a_xy = agent_xy(data, self._xy_adr)
        heading = self._heading(data)
        sensors = self._sensor_obs(data)

        def lidar(xys):
            return pseudo_lidar(a_xy, heading, xys, n_bins=self.n_bins, max_dist=self.lidar_max_dist)


        parts = [sensors, lidar(info['goal_xy'][None, :])]
        if self.n_hazards > 0:
            parts.append(lidar(info['hazards_xy']))
        if self.n_vases > 0:
            parts.append(lidar(info['vases_xy']))
        return jnp.concatenate(parts)

    @property
    def obs_dim(self) -> int:

        return self._sensor_obs_dim + self.n_bins * (
            1 + int(self.n_hazards > 0) + int(self.n_vases > 0))


    @abc.abstractmethod
    def reward_vector(self, prev_dist, data, action, info) -> jax.Array:
        pass

    @abc.abstractmethod
    def cost_vector(self, data, info) -> jax.Array:
        pass


    def reset(self, key) -> State:
        k_layout, k_agent, k_head, k_vrot, k_state = jax.random.split(key, 5)
        layout = self.sample_layout(k_layout)


        agent_xy0 = layout.get('agent_xy')
        if agent_xy0 is None:
            agent_xy0 = self._sample_xy(k_agent, shrink=self.agent_keepout)


        head = jax.random.uniform(k_head, (), minval=0.0, maxval=2.0 * jnp.pi)
        qpos = (self._init_data.qpos
                .at[self._xy_adr[0]].set(agent_xy0[0])
                .at[self._xy_adr[1]].set(agent_xy0[1]))
        qpos = self._write_heading(qpos, head)

        zeros2 = lambda n: jnp.zeros((max(n, 0), 2), dtype=jnp.float32)
        vases_xy0 = layout.get('vases_xy', zeros2(self.n_vases))


        if self.n_vases > 0:
            vrot = jax.random.uniform(k_vrot, (self.n_vases,), minval=0.0,
                                      maxval=2.0 * jnp.pi)
            for i, adr in enumerate(self._vase_qpos_adr):
                qpos = (qpos
                        .at[adr + 0].set(vases_xy0[i, 0])
                        .at[adr + 1].set(vases_xy0[i, 1])
                        .at[adr + 2].set(0.1 - 4e-5)
                        .at[adr + 3].set(jnp.cos(vrot[i] / 2.0))
                        .at[adr + 4].set(0.0)
                        .at[adr + 5].set(0.0)
                        .at[adr + 6].set(jnp.sin(vrot[i] / 2.0)))
        data0 = self._apply_layout_extras(self._init_data.replace(qpos=qpos), layout)
        data = mjx.forward(self.mjx_model, data0)


        info = {k: v for k, v in layout.items() if k != 'agent_xy'}
        info['hazards_xy'] = layout.get('hazards_xy', zeros2(self.n_hazards))
        info['vases_xy'] = vases_xy0

        info['agent_xy0'] = agent_xy0
        info['last_dist_goal'] = self.dist_goal(data, info)
        info['step'] = jnp.array(0.0)

        obs = self.observation(data, info)
        return State(
            data=data,
            obs=obs,
            reward=jnp.zeros((self.n_rewards,), dtype=jnp.float32),
            cost=jnp.zeros((self.n_costs,), dtype=jnp.float32),
            done=jnp.array(0.0),
            key=k_state,
            info=info,
        )


    def _apply_layout_extras(self, data0, layout):


        return data0

    def _pre_physics(self, data, info):

        return data

    def _post_physics_info(self, data, info) -> dict:

        return info

    def _on_goal_reached(self, key, data, info, reached) -> dict:


        new_goal = self._resample_goal_keepout(key, data, info)
        cont = jnp.float32(self.continue_goal)
        info['goal_xy'] = jnp.where((reached * cont)[None].astype(bool),
                                    new_goal, info['goal_xy'])
        return info

    def _update_trackers(self, data, info) -> dict:

        info['last_dist_goal'] = jnp.linalg.norm(
            agent_xy(data, self._xy_adr) - info['goal_xy'])
        return info

    def step(self, state: State, action: jax.Array) -> State:
        ctrl = jnp.clip(action, -1.0, 1.0) * self.action_scale
        data = self._pre_physics(state.data, state.info)
        data = physics_step(self.mjx_model, data, ctrl, self.frame_skip)
        info = dict(state.info)


        if self.n_vases > 0:
            info['vases_xy'] = data.xpos[self._vase_body_ids, :2]
        info = self._post_physics_info(data, info)

        reward = self.reward_vector(info['last_dist_goal'], data, action, info)
        cost = self.cost_vector(data, info)

        reached = self.is_goal_reached(data, info)

        key, sub = jax.random.split(state.key)
        info = self._on_goal_reached(sub, data, info, reached)
        info = self._update_trackers(data, info)
        info['step'] = info['step'] + 1.0


        cont = jnp.float32(self.continue_goal)
        done = reached * (1.0 - cont)

        obs = self.observation(data, info)
        return State(data=data, obs=obs, reward=reward, cost=cost,
                     done=done, key=key, info=info)
