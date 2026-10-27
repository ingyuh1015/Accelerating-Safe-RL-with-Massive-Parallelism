from __future__ import annotations
from typing import Optional

import numpy as np
import jax
import gymnasium
from gymnasium import spaces

from .bases.underlying import State


class Builder(gymnasium.Env):
    metadata = {'render_modes': []}

    def __init__(self, task, max_episode_steps: int = 1000, seed: int = 0):
        self.task = task
        self.max_episode_steps = int(max_episode_steps)

        self._reset_fn = jax.jit(task.reset)
        self._step_fn = jax.jit(task.step)

        self._key = jax.random.PRNGKey(seed)
        self._state: Optional[State] = None
        self._elapsed = 0

        a_dim = task.world.action_dim
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(a_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(task.obs_dim,), dtype=np.float32)
        self.reward_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(task.n_rewards,), dtype=np.float32)
        self.cost_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(task.n_costs,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._key = jax.random.PRNGKey(int(seed))
        self._key, sub = jax.random.split(self._key)
        self._state = self._reset_fn(sub)
        self._elapsed = 0
        return np.asarray(self._state.obs), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self._state = self._step_fn(self._state, action)
        self._elapsed += 1

        obs = np.asarray(self._state.obs)
        reward_vec = np.asarray(self._state.reward)
        cost_vec = np.asarray(self._state.cost)
        terminated = bool(self._state.done > 0.5)
        truncated = self._elapsed >= self.max_episode_steps

        info = {
            'cost_vec': cost_vec,
            'reward_vec': reward_vec,
            'goal_xy': np.asarray(self._state.info['goal_xy']),
        }
        return obs, reward_vec, cost_vec, terminated, truncated, info
