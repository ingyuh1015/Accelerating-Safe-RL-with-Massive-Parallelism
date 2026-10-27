from __future__ import annotations

import jax
import jax.numpy as jnp
from mujoco import mjx

from ....bases.underlying import State, physics_step, make_init_data


THRESHOLDS = {'2x4': 2.522, '4x2': 2.418}


ANT0_X = 0
ANT0_Z = 2
ANT1_X = 15
ANT1_Z = 17


class MultiAntVelocity:


    n_rewards: int = 2
    n_costs: int = 2

    frame_skip: int = 5
    velocity_threshold: float = THRESHOLDS['2x4']
    forward_reward_weight: float = 1.0
    healthy_reward: float = 1.0
    terminate_when_unhealthy: bool = True
    reset_noise_scale: float = 5e-3
    healthy_z_min: float = 0.2
    healthy_z_max: float = 1.0

    def __init__(self, world, *, velocity_threshold: float | None = None):
        self.world = world
        self.mjx_model = world.mjx_model
        self.dt = float(world.mj_model.opt.timestep) * self.frame_skip
        if velocity_threshold is not None:
            self.velocity_threshold = float(velocity_threshold)
        self._init_data = (world.init_data if hasattr(world, 'init_data')
                           else make_init_data(world.mjx_model))
        self._init_qpos = jnp.asarray(world.mj_model.qpos0)
        self._nq = int(world.mj_model.nq)
        self._nv = int(world.mj_model.nv)


    def observation(self, data) -> jax.Array:


        return jnp.concatenate([data.qpos[1:], jnp.clip(data.qvel, -10.0, 10.0)])

    @property
    def obs_dim(self) -> int:
        return (self._nq - 1) + self._nv


    def _ant_healthy(self, data, z_idx: int) -> jax.Array:
        z = data.qpos[z_idx]
        z_ok = (z >= self.healthy_z_min) & (z <= self.healthy_z_max)
        return z_ok.astype(jnp.float32)

    def is_healthy(self, data) -> jax.Array:
        finite = jnp.all(jnp.isfinite(data.qpos)) & jnp.all(jnp.isfinite(data.qvel))
        both = self._ant_healthy(data, ANT0_Z) * self._ant_healthy(data, ANT1_Z)
        return (both * finite.astype(jnp.float32))

    def _forward(self, data):
        return mjx.forward(self.mjx_model, data)


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
            'x0_prev': data.qpos[ANT0_X],
            'x1_prev': data.qpos[ANT1_X],
            'step': jnp.array(0.0),
        }
        return State(
            data=data, obs=obs,
            reward=jnp.zeros((self.n_rewards,), dtype=jnp.float32),
            cost=jnp.zeros((self.n_costs,), dtype=jnp.float32),
            done=jnp.array(0.0), key=k_state, info=info,
        )

    def step(self, state: State, action: jax.Array) -> State:
        x0_before = state.data.qpos[ANT0_X]
        x1_before = state.data.qpos[ANT1_X]
        ctrl = jnp.clip(action, -1.0, 1.0)
        data = physics_step(self.mjx_model, state.data, ctrl, self.frame_skip)
        x0_after = data.qpos[ANT0_X]
        x1_after = data.qpos[ANT1_X]
        x0_vel = (x0_after - x0_before) / self.dt
        x1_vel = (x1_after - x1_before) / self.dt

        h0 = self._ant_healthy(data, ANT0_Z)
        h1 = self._ant_healthy(data, ANT1_Z)
        r0 = self.forward_reward_weight * x0_vel + self.healthy_reward * h0
        r1 = self.forward_reward_weight * x1_vel + self.healthy_reward * h1
        reward = jnp.stack([r0, r1])

        c0 = (x0_vel > self.velocity_threshold).astype(jnp.float32)
        c1 = (x1_vel > self.velocity_threshold).astype(jnp.float32)
        cost = jnp.stack([c0, c1])

        info = dict(state.info)
        info['x0_prev'] = x0_after
        info['x1_prev'] = x1_after
        info['step'] = info['step'] + 1.0

        healthy = self.is_healthy(data)
        done = (1.0 - healthy) * jnp.float32(self.terminate_when_unhealthy)

        obs = self.observation(data)
        return State(data=data, obs=obs, reward=reward, cost=cost,
                     done=done, key=state.key, info=info)


MultiVelocity = MultiAntVelocity
