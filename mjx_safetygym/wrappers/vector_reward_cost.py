from __future__ import annotations

import numpy as np


class ScalarizeReward:


    def __init__(self, env, weights=None):
        self.env = env
        if weights is None:
            weights = np.ones((env.n_rewards,), dtype=np.float32) / float(env.n_rewards)
        self.weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        if self.weights.shape[0] != env.n_rewards:
            raise ValueError(
                f'weights shape {self.weights.shape} != n_rewards {env.n_rewards}'
            )

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, actions):
        obs, reward, cost, done, info = self.env.step(actions)
        reward = np.asarray(reward)
        scalar = reward @ self.weights
        info = dict(info)
        info['reward_vec'] = reward
        return obs, scalar.astype(np.float32), cost, done, info


class LinearPreference:


    def __init__(self, env, preference):
        self.env = env
        self.set_preference(preference)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def set_preference(self, preference) -> None:
        omega = np.asarray(preference, dtype=np.float32).reshape(-1)
        if omega.shape[0] != self.env.n_rewards:
            raise ValueError(
                f'preference shape {omega.shape} != n_rewards {self.env.n_rewards}'
            )
        self.preference = omega

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, actions):
        obs, reward, cost, done, info = self.env.step(actions)
        reward = np.asarray(reward)
        scalar = reward @ self.preference
        info = dict(info)
        info['reward_vec'] = reward
        return obs, scalar.astype(np.float32), cost, done, info


class CostSumView:


    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, actions):
        obs, reward, cost, done, info = self.env.step(actions)
        cost = np.asarray(cost)
        info = dict(info)
        info['cost_sum'] = cost.sum(axis=-1).astype(np.float32)
        return obs, reward, cost, done, info
