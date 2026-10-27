from __future__ import annotations
import abc

import jax
import jax.numpy as jnp

from .underlying import State, physics_step, make_init_data
from ..utils.lidar import pseudo_lidar, compass


AGENT0_XY_ADR = (0, 1)
AGENT0_HEADING_ADR = 2
AGENT1_XY_ADR = (3, 4)
AGENT1_HEADING_ADR = 5


class BaseMultiTask(abc.ABC):

    n_rewards: int = 2
    n_costs: int = 2
    n_hazards: int = 0
    n_bins: int = 16
    lidar_max_dist: float = 3.0

    reward_distance: float = 1.0
    reward_goal: float = 1.0

    def __init__(self, world, *, extents=(-1.0, -1.0, 1.0, 1.0),
                 goal_size: float = 0.3, hazard_size: float = 0.2,
                 ctrl_cost_weight: float = 0.01, action_scale: float = 1.0,
                 continue_goal: bool = True):
        self.world = world
        self.mjx_model = world.mjx_model
        self.extents = jnp.asarray(extents, dtype=jnp.float32)
        self.goal_size = float(goal_size)
        self.hazard_size = float(hazard_size)
        self.ctrl_cost_weight = float(ctrl_cost_weight)
        self.action_scale = float(action_scale)
        self.continue_goal = bool(continue_goal)
        self.frame_skip = int(world.frame_skip)
        self._init_data = (world.init_data if hasattr(world, 'init_data')
                           else make_init_data(world.mjx_model))


    @staticmethod
    def _xy(data, adr) -> jax.Array:
        return jnp.stack([data.qpos[adr[0]], data.qpos[adr[1]]])

    @staticmethod
    def _vel_xy(data, adr) -> jax.Array:
        return jnp.stack([data.qvel[adr[0]], data.qvel[adr[1]]])

    def agent0_xy(self, data) -> jax.Array:
        return self._xy(data, AGENT0_XY_ADR)

    def agent1_xy(self, data) -> jax.Array:
        return self._xy(data, AGENT1_XY_ADR)


    def _sample_xy(self, key) -> jax.Array:
        lo = self.extents[:2]
        hi = self.extents[2:]
        return jax.random.uniform(key, shape=(2,), minval=lo, maxval=hi)

    def sample_layout(self, key) -> dict:

        k_g0, k_g1, k_haz = jax.random.split(key, 3)
        goal0_xy = self._sample_xy(k_g0)
        goal1_xy = self._sample_xy(k_g1)
        if self.n_hazards > 0:
            haz_keys = jax.random.split(k_haz, self.n_hazards)
            hazards_xy = jax.vmap(self._sample_xy)(haz_keys)
        else:
            hazards_xy = jnp.zeros((max(self.n_hazards, 0), 2), dtype=jnp.float32)
        return {'goal0_xy': goal0_xy, 'goal1_xy': goal1_xy, 'hazards_xy': hazards_xy}


    def dist_goal(self, data, info, agent_idx: int) -> jax.Array:
        if agent_idx == 0:
            return jnp.linalg.norm(self.agent0_xy(data) - info['goal0_xy'])
        return jnp.linalg.norm(self.agent1_xy(data) - info['goal1_xy'])

    def _reached(self, cur_dist) -> jax.Array:
        return (cur_dist <= self.goal_size).astype(jnp.float32)


    def reward_vector(self, state: State, data, action, info) -> jax.Array:

        cur0 = self.dist_goal(data, info, 0)
        cur1 = self.dist_goal(data, info, 1)
        r0 = (info['last_dist_0'] - cur0) * self.reward_distance \
            + self._reached(cur0) * self.reward_goal
        r1 = (info['last_dist_1'] - cur1) * self.reward_distance \
            + self._reached(cur1) * self.reward_goal
        return jnp.stack([r0, r1])


    @abc.abstractmethod
    def cost_vector(self, data, info) -> jax.Array:
        pass


    def _agent_obs(self, data, xy_adr, heading_adr, goal_xy, info) -> jax.Array:
        a_xy = self._xy(data, xy_adr)
        heading = data.qpos[heading_adr]
        vel = self._vel_xy(data, xy_adr)
        ego = jnp.array([
            jnp.cos(heading), jnp.sin(heading), vel[0], vel[1],
            data.qvel[heading_adr],
        ])
        goal_comp = compass(a_xy, heading, goal_xy)
        goal_lidar = pseudo_lidar(a_xy, heading, goal_xy[None, :],
                                  n_bins=self.n_bins, max_dist=self.lidar_max_dist)
        if self.n_hazards > 0:
            haz_lidar = pseudo_lidar(a_xy, heading, info['hazards_xy'],
                                     n_bins=self.n_bins, max_dist=self.lidar_max_dist)
        else:
            haz_lidar = jnp.zeros((self.n_bins,), dtype=jnp.float32)
        return jnp.concatenate([ego, goal_comp, goal_lidar, haz_lidar])

    def observation(self, data, info) -> jax.Array:

        o0 = self._agent_obs(data, AGENT0_XY_ADR, AGENT0_HEADING_ADR,
                             info['goal0_xy'], info)
        o1 = self._agent_obs(data, AGENT1_XY_ADR, AGENT1_HEADING_ADR,
                             info['goal1_xy'], info)
        return jnp.concatenate([o0, o1])

    @property
    def per_agent_obs_dim(self) -> int:

        return 5 + 2 + self.n_bins + self.n_bins

    @property
    def obs_dim(self) -> int:
        return 2 * self.per_agent_obs_dim


    def reset(self, key) -> State:
        k_layout, k_state = jax.random.split(key)
        layout = self.sample_layout(k_layout)

        data = self._init_data
        info = dict(layout)
        info['last_dist_0'] = self.dist_goal(data, info, 0)
        info['last_dist_1'] = self.dist_goal(data, info, 1)
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

    def step(self, state: State, action: jax.Array) -> State:
        ctrl = jnp.clip(action, -1.0, 1.0) * self.action_scale
        data = physics_step(self.mjx_model, state.data, ctrl, self.frame_skip)
        info = dict(state.info)

        reward = self.reward_vector(state, data, action, info)
        cost = self.cost_vector(data, info)

        cur0 = self.dist_goal(data, info, 0)
        cur1 = self.dist_goal(data, info, 1)
        reached0 = self._reached(cur0)
        reached1 = self._reached(cur1)


        key, sub0, sub1 = jax.random.split(state.key, 3)
        cont = jnp.float32(self.continue_goal)
        new_g0 = self._sample_xy(sub0)
        new_g1 = self._sample_xy(sub1)
        goal0_xy = jnp.where((reached0 * cont)[None].astype(bool), new_g0, info['goal0_xy'])
        goal1_xy = jnp.where((reached1 * cont)[None].astype(bool), new_g1, info['goal1_xy'])
        info['goal0_xy'] = goal0_xy
        info['goal1_xy'] = goal1_xy


        info['last_dist_0'] = jnp.linalg.norm(self.agent0_xy(data) - goal0_xy)
        info['last_dist_1'] = jnp.linalg.norm(self.agent1_xy(data) - goal1_xy)
        info['step'] = info['step'] + 1.0


        done = jnp.array(0.0)

        obs = self.observation(data, info)
        return State(data=data, obs=obs, reward=reward, cost=cost,
                     done=done, key=key, info=info)
