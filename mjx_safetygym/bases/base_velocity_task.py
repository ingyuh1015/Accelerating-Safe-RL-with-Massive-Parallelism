from __future__ import annotations
import abc

import jax
import jax.numpy as jnp

from .underlying import State, physics_step, make_init_data


class BaseVelocityTask(abc.ABC):
    n_rewards: int = 2
    n_costs: int = 1


    xml_name: str = ''
    frame_skip: int = 4
    velocity_threshold: float = 1.0
    forward_reward_weight: float = 1.0
    ctrl_cost_weight: float = 1e-3
    healthy_reward: float = 1.0
    terminate_when_unhealthy: bool = True
    healthy_z_min: float = -jnp.inf
    healthy_z_max: float = jnp.inf
    reset_noise_scale: float = 5e-3
    exclude_x_in_obs: bool = True
    use_planar_speed: bool = False

    def __init__(self, world, *, velocity_threshold: float | None = None):
        self.world = world
        self.mjx_model = world.mjx_model
        self.dt = float(world.mj_model.opt.timestep) * self.frame_skip
        if velocity_threshold is not None:
            self.velocity_threshold = float(velocity_threshold)
        self._init_data = (world.init_data if hasattr(world, 'init_data')
                           else make_init_data(world.mjx_model))
        self._init_qpos = jnp.asarray(world.mj_model.qpos0)
        self._nq = world.mj_model.nq
        self._nv = world.mj_model.nv


    def observation(self, data) -> jax.Array:
        qpos = data.qpos
        if self.exclude_x_in_obs:
            qpos = qpos[1:]
        return jnp.concatenate([qpos, jnp.clip(data.qvel, -10.0, 10.0)])

    @property
    def obs_dim(self) -> int:
        return (self._nq - (1 if self.exclude_x_in_obs else 0)) + self._nv


    def is_healthy(self, data) -> jax.Array:
        return jnp.array(1.0, dtype=jnp.float32)

    def _z_healthy(self, data, z_idx: int) -> jax.Array:

        z = data.qpos[z_idx]
        ok = (z >= self.healthy_z_min) & (z <= self.healthy_z_max)
        return ok.astype(jnp.float32)

    def _position(self, data) -> jax.Array:
        return data.qpos[:2]


    def reset(self, key) -> State:
        k_pos, k_vel, k_state = jax.random.split(key, 3)
        s = self.reset_noise_scale
        qpos = self._init_qpos + jax.random.uniform(
            k_pos, (self._nq,), minval=-s, maxval=s)
        qvel = jax.random.uniform(k_vel, (self._nv,), minval=-s, maxval=s)
        data = self._init_data.replace(qpos=qpos, qvel=qvel)
        data = self._forward(data)

        obs = self.observation(data)
        info = {
            'x_prev': self._position(data)[0],
            'step': jnp.array(0.0),
        }
        return State(
            data=data, obs=obs,
            reward=jnp.zeros((self.n_rewards,), dtype=jnp.float32),
            cost=jnp.zeros((self.n_costs,), dtype=jnp.float32),
            done=jnp.array(0.0), key=k_state, info=info,
        )

    def _forward(self, data):

        from mujoco import mjx
        return mjx.forward(self.mjx_model, data)

    def step(self, state: State, action: jax.Array) -> State:
        position_before = self._position(state.data)
        ctrl = jnp.clip(action, -1.0, 1.0)
        data = physics_step(self.mjx_model, state.data, ctrl, self.frame_skip)
        position_after = self._position(data)
        velocity = (position_after - position_before) / self.dt
        x_vel = velocity[0]

        healthy = self.is_healthy(data)
        alive = jnp.maximum(healthy, jnp.float32(self.terminate_when_unhealthy))
        forward_reward = self.forward_reward_weight * x_vel + self.healthy_reward * alive
        ctrl_penalty = -self.ctrl_cost_weight * jnp.sum(jnp.square(action))
        reward = jnp.stack([forward_reward, ctrl_penalty])

        speed = jnp.linalg.norm(velocity) if self.use_planar_speed else x_vel
        vel_cost = (speed > self.velocity_threshold).astype(jnp.float32)
        cost = jnp.array([vel_cost])

        info = dict(state.info)
        info['x_prev'] = position_after[0]
        info['step'] = info['step'] + 1.0

        unhealthy = (1.0 - healthy) * jnp.float32(self.terminate_when_unhealthy)
        done = unhealthy

        obs = self.observation(data)
        return State(data=data, obs=obs, reward=reward, cost=cost,
                     done=done, key=state.key, info=info)
