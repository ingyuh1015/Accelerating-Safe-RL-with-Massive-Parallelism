from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from ..bases.underlying import State


class MJXVectorEnv:


    def __init__(self, task, num_envs: int, max_episode_steps: int = 1000, seed: int = 0):
        self.task = task
        self.num_envs = int(num_envs)
        self.max_episode_steps = int(max_episode_steps)
        self.n_rewards = task.n_rewards
        self.n_costs = task.n_costs
        self.obs_dim = task.obs_dim
        self.action_dim = task.world.action_dim

        self._key = jax.random.PRNGKey(seed)
        self._state = None

        self._reset_batch = jax.jit(jax.vmap(task.reset))
        self._step_batch = jax.jit(self._make_step_with_autoreset())

    def _make_step_with_autoreset(self):
        task = self.task
        max_steps = self.max_episode_steps

        def step_one(state: State, action, reset_key):
            nxt = task.step(state, action)
            truncated = (nxt.info['step'] >= max_steps).astype(jnp.float32)
            terminated = nxt.done
            done = jnp.maximum(terminated, truncated)
            fresh = task.reset(reset_key)

            next_state = jax.tree.map(
                lambda f, n: jnp.where(done.astype(bool), f, n), fresh, nxt,
            )

            return next_state, nxt.obs, nxt.reward, nxt.cost, terminated, truncated

        def step_batch(state, actions, reset_keys):
            return jax.vmap(step_one)(state, actions, reset_keys)

        return step_batch


    def reset(self, seed=None):
        if seed is not None:
            self._key = jax.random.PRNGKey(int(seed))
        self._key, sub = jax.random.split(self._key)
        keys = jax.random.split(sub, self.num_envs)
        self._state = self._reset_batch(keys)
        return np.asarray(self._state.obs)

    def step(self, actions):
        actions = jnp.asarray(actions, dtype=jnp.float32)
        self._key, sub = jax.random.split(self._key)
        reset_keys = jax.random.split(sub, self.num_envs)
        next_state, final_obs, reward, cost, terminated, truncated = \
            self._step_batch(self._state, actions, reset_keys)
        self._state = next_state

        obs = np.asarray(next_state.obs)
        reward = np.asarray(reward)
        cost = np.asarray(cost)
        terminated = np.asarray(terminated)
        truncated = np.asarray(truncated)
        done = np.maximum(terminated, truncated)
        info = {
            'reward_vec': reward, 'cost_vec': cost,
            'final_observation': np.asarray(final_obs),
            'terminated': terminated, 'truncated': truncated,
        }
        return obs, reward, cost, done, info
