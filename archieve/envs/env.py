from __future__ import annotations
from typing import TYPE_CHECKING, Any
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from archieve import get_dtype, ConstraintConfig, EnvState
if TYPE_CHECKING:
    from archieve.envs.agents import BaseAgentConfig
    from archieve.envs.tasks import BaseObjConfig

def sample_point(rng: jax.Array, radius: float) -> jnp.ndarray:
    return jax.random.uniform(rng, (2,), minval=-radius, maxval=radius)

class MjxEnv:

    def __init__(self, *, mj_model: mujoco.MjModel, mjx_model: mjx.Model, agent: BaseAgentConfig, objects: dict[str, BaseObjConfig], placements_extents: tuple, constraints: tuple[ConstraintConfig, ...], action_size: int, ctrl_dt: float, n_substeps: int, agent_geom_ids: jnp.ndarray) -> None:
        self._dtype = get_dtype()
        self._mj_model = mj_model
        self._mjx_model = mjx_model
        self.agent = agent
        self.objects = objects
        self.placements_extents = tuple(placements_extents)
        self.constraints = tuple(constraints)
        self._action_size_val = action_size
        self._num_constraints = len(self.constraints)
        self._ctrl_dt = float(ctrl_dt)
        self._n_substeps = int(n_substeps)
        self._zero_action = jnp.zeros((action_size,), dtype=self._dtype)
        self._agent_geom_ids = agent_geom_ids
        self._default_spawn_radius = (self.placements_extents[2] - self.placements_extents[0]) / 2.0
        self._agent_keepout = float(agent.keepout)
        goal_cfg = objects.get('goal')
        self._goal_keepout = float(goal_cfg.keepout) if goal_cfg is not None else 0.0
        self._placement_objects = tuple(((name, cfg) for (name, cfg) in objects.items() if name not in ('goal', 'circle')))
        self._mocap_descriptors = tuple(({'name': name, 'body_mocapid': cfg.model_info.body_mocapid, 'travel': float(cfg.travel), 'size': float(cfg.size), 'base_xy_key': 'gremlin_base_xy' if name == 'gremlins' else f'{name}_base_xy'} for (name, cfg) in objects.items() if cfg.obj_type == 'mocap' and cfg.num > 0))
        self._obs_size = int(self.agent.agent_obs_size)

    def _compute_obs_size(self) -> int:
        return int(self.agent.agent_obs_size)

    def compute_step_info(self, data, info: dict, action) -> None:
        del action
        self.agent.compute_state_info(data, info, ctrl_dt=self._ctrl_dt)

    def compute_reward(self, data, info: dict):
        info['reward'] = jnp.zeros((), dtype=self._dtype)
        return data

    def compute_obs(self, data, info: dict) -> jnp.ndarray:
        return self.agent.compute_agent_obs(data).astype(self._dtype)

    def compute_cost_vector(self, data, info: dict) -> jnp.ndarray:
        cost_vector = jnp.zeros((self._num_constraints,), dtype=self._dtype)
        for (i, c) in enumerate(self.constraints):
            c.compute(data, info)
            cost_vector = cost_vector.at[i].set(info[f'cost_{c.name}'])
        return cost_vector

    def mjx_step(self, data, action, info: dict):
        objects = tuple(({**d, 'base_xy': info[d['base_xy_key']]} for d in self._mocap_descriptors))

        def substep(d, _):
            dtype = get_dtype()
            t_sin = jnp.sin(d.time)
            t_cos = jnp.cos(d.time)
            for obj in objects:
                time_xy = jnp.array([t_sin, t_cos], dtype=dtype) * jnp.asarray(obj['travel'], dtype=dtype)
                mocap_xy = obj['base_xy'] + time_xy[None, :]
                z = jnp.ones((mocap_xy.shape[0], 1), dtype=dtype) * jnp.asarray(obj['size'], dtype=dtype)
                mocap_pos = jnp.concatenate([mocap_xy, z], axis=1)
                d = d.replace(mocap_pos=d.mocap_pos.at[obj['body_mocapid']].set(mocap_pos))
            d = d.replace(ctrl=jnp.asarray(action, dtype=d.ctrl.dtype))
            return (mjx.step(self._mjx_model, d), None)
        return jax.lax.scan(substep, data, None, length=self._n_substeps)[0]

    @property
    def n_substeps(self) -> int:
        return self._n_substeps

    @property
    def action_size(self) -> int:
        return self._action_size_val

    @property
    def observation_size(self) -> int:
        return self._obs_size

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def unwrapped(self) -> 'MjxEnv':
        return self

    def _sample_obstacles(self, rng: jax.Array, agent_pos: jnp.ndarray, goal: jnp.ndarray, count: int, keepout: float=0.2, max_attempts: int=20) -> jnp.ndarray:
        spawn_radius = self._default_spawn_radius
        agent_keepout = self._agent_keepout
        goal_keepout = self._goal_keepout
        safe_count = max(int(count), 1)
        dtype = get_dtype()
        placed_init = jnp.ones((safe_count, 2), dtype=dtype) * 100.0
        min_agent_dist_sq = jnp.asarray(keepout + agent_keepout, dtype=dtype) ** 2
        min_goal_dist_sq = jnp.asarray(keepout + goal_keepout, dtype=dtype) ** 2
        min_pair_dist_sq = jnp.asarray(keepout * 2.0, dtype=dtype) ** 2

        def sample_one(carry: tuple[jax.Array, jnp.ndarray], idx: jax.Array) -> tuple[tuple[jax.Array, jnp.ndarray], jnp.ndarray]:
            (key, placed) = carry
            (key, init_key) = jax.random.split(key)
            init_pos = sample_point(init_key, spawn_radius)

            def attempt(sub_carry: tuple[jax.Array, jnp.ndarray], _unused: None) -> tuple[tuple[jax.Array, jnp.ndarray], jax.Array]:
                (key_in, best) = sub_carry
                (key_in, subkey) = jax.random.split(key_in)
                cand = sample_point(subkey, spawn_radius)
                cand_agent_delta = cand - agent_pos
                cand_goal_delta = cand - goal
                placed_delta = placed - cand
                valid = (jnp.sum(cand_agent_delta * cand_agent_delta) > min_agent_dist_sq) & (jnp.sum(cand_goal_delta * cand_goal_delta) > min_goal_dist_sq) & jnp.all(jnp.sum(placed_delta * placed_delta, axis=-1) > min_pair_dist_sq)
                new_best = jnp.where(valid, cand, best)
                return ((key_in, new_best), valid)
            ((key, final_pos), _) = jax.lax.scan(attempt, (key, init_pos), None, length=max_attempts)
            placed = placed.at[idx].set(final_pos)
            return ((key, placed), final_pos)
        ((_, _), result) = jax.lax.scan(sample_one, (rng, placed_init), jnp.arange(safe_count))
        return result[:count]

    def reset_objects(self, data, info: dict, rng) -> Any:
        needs_forward = False
        target_ref = info.get('target_pos', info['agent_pos'])
        for (name, cfg) in self._placement_objects:
            obj_rng = jax.random.fold_in(rng, int.from_bytes(name.encode(), 'little') % 2 ** 31)
            positions = self._sample_obstacles(obj_rng, info['agent_pos'], target_ref, cfg.num, float(cfg.keepout))
            info[f'{name}_positions'] = positions
            if cfg.obj_type in ('freejoint', 'mocap'):
                body_count = positions.shape[0]
                pose = jnp.concatenate([positions, jnp.ones((body_count, 1), dtype=get_dtype()) * jnp.asarray(float(cfg.size), dtype=get_dtype()), jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=get_dtype()), (body_count, 1))], axis=1)
                offsets = jnp.arange(7, dtype=jnp.int8)[None, :]
                indices = cfg.model_info.jnt_qposadr[:, None] + offsets
                data = data.replace(qpos=data.qpos.at[indices.reshape(-1)].set(pose.reshape(-1)))
                needs_forward = True
            elif cfg.obj_type == 'static_mocap':
                dtype = get_dtype()
                mocap_xy = jnp.asarray(positions, dtype=dtype)
                center_z = getattr(cfg, 'height', 0.0) / 2.0 if getattr(cfg, 'height', 0.0) > 0 else float(cfg.size)
                z = jnp.ones((mocap_xy.shape[0], 1), dtype=dtype) * jnp.asarray(center_z, dtype=dtype)
                mocap_pose = jnp.concatenate([mocap_xy, z], axis=1)
                data = data.replace(mocap_pos=data.mocap_pos.at[cfg.model_info.body_mocapid].set(mocap_pose))
                needs_forward = True
            if name == 'gremlins':
                info['gremlin_base_xy'] = positions
                dtype = get_dtype()
                travel = jnp.asarray(cfg.travel, dtype=dtype)
                size = jnp.asarray(cfg.size, dtype=dtype)
                t_sin = jnp.sin(jnp.asarray(data.time, dtype=dtype))
                t_cos = jnp.cos(jnp.asarray(data.time, dtype=dtype))
                time_xy = jnp.array([t_sin, t_cos], dtype=dtype) * travel
                mocap_xy = positions.astype(dtype) + time_xy[None, :]
                z = jnp.ones((mocap_xy.shape[0], 1), dtype=dtype) * size
                mocap_pos = jnp.concatenate([mocap_xy, z], axis=1)
                data = data.replace(mocap_pos=data.mocap_pos.at[cfg.model_info.body_mocapid].set(mocap_pos))
        if needs_forward:
            data = mjx.forward(self._mjx_model, data)
        return data

    def reset(self, rng: jax.Array) -> EnvState:
        data = mjx.make_data(self._mjx_model, device=jax.local_devices()[0])
        data = mjx.forward(self._mjx_model, data)
        (rng, sub_rng) = jax.random.split(rng)
        (rng, agent_rng) = jax.random.split(rng)
        ext = self.placements_extents
        xy = jax.random.uniform(agent_rng, (2,), minval=jnp.asarray([ext[0], ext[1]], dtype=self._dtype), maxval=jnp.asarray([ext[2], ext[3]], dtype=self._dtype))
        data = data.replace(qpos=data.qpos.at[0:2].set(xy))
        (rng, theta_rng) = jax.random.split(rng)
        data = self.agent.randomize_init_qpos(data, theta_rng)
        data = mjx.forward(self._mjx_model, data)
        info: dict[str, Any] = {'step_count': jnp.asarray(0, dtype=jnp.int32), 'goal_rng': sub_rng, 'button_rng': sub_rng, 'agent_pos': data.qpos[:2].astype(self._dtype)}
        data = self.reset_objects(data, info, sub_rng)
        info['rng'] = rng
        self.compute_step_info(data, info, self._zero_action)
        data = self.compute_reward(data, info)
        cost_vector = self.compute_cost_vector(data, info)
        obs = self.compute_obs(data, info)
        state = EnvState.build_zero(data=data, obs=obs, num_constraints=self._num_constraints, info=info)
        return state.replace(reward=info['reward'], cost_vector=cost_vector)

    def step(self, state: EnvState, action: jax.Array) -> EnvState:
        rng = state.info.get('rng', jax.random.PRNGKey(0))
        (rng, _) = jax.random.split(rng)
        info = dict(state.info)
        data = self.mjx_step(state.data, action, info)
        self.compute_step_info(data, info, action)
        data = self.compute_reward(data, info)
        cost_vector = self.compute_cost_vector(data, info)
        obs = self.compute_obs(data, info)
        info['rng'] = rng
        terminated = jnp.asarray(info.get('terminated', 0.0), dtype=self._dtype)
        return EnvState(data=data, obs=obs, reward=info['reward'], cost_vector=cost_vector, done=terminated, terminated=terminated, truncated=jnp.zeros((), dtype=self._dtype), info=info)
