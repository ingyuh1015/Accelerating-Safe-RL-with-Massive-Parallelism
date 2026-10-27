from __future__ import annotations

import numpy as np


class _RunningMeanStd:


    def __init__(self, shape, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:

        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var = m2 / tot_count
        self.count = tot_count


class NormalizeObservation:


    def __init__(self, env, epsilon: float = 1e-8, clip: float = 10.0):
        self.env = env
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self.training = True
        self.obs_rms = _RunningMeanStd(shape=(env.obs_dim,))

    def __getattr__(self, name):


        return getattr(self.env, name)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        if self.training:
            self.obs_rms.update(obs)
        norm = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        norm = np.clip(norm, -self.clip, self.clip)
        return norm.astype(np.float32)

    def reset(self, seed=None):
        obs = self.env.reset(seed=seed)
        return self.normalize(obs)

    def step(self, actions):
        obs, reward, cost, done, info = self.env.step(actions)
        return self.normalize(obs), reward, cost, done, info


    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self


    def save_stats(self) -> dict:
        return {
            'mean': np.asarray(self.obs_rms.mean).copy(),
            'var': np.asarray(self.obs_rms.var).copy(),
            'count': float(self.obs_rms.count),
        }

    def load_stats(self, stats: dict) -> None:
        self.obs_rms.mean = np.asarray(stats['mean'], dtype=np.float64).copy()
        self.obs_rms.var = np.asarray(stats['var'], dtype=np.float64).copy()
        self.obs_rms.count = float(stats['count'])


class NormalizeReward:


    def __init__(self, env, gamma: float = 0.99, epsilon: float = 1e-8, clip: float = 10.0):
        self.env = env
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self.training = True
        self.n_rewards = env.n_rewards
        self.num_envs = env.num_envs
        self.return_rms = _RunningMeanStd(shape=(self.n_rewards,))
        self._returns = np.zeros((self.num_envs, self.n_rewards), dtype=np.float64)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def normalize(self, reward: np.ndarray) -> np.ndarray:
        norm = reward / np.sqrt(self.return_rms.var + self.epsilon)
        norm = np.clip(norm, -self.clip, self.clip)
        return norm.astype(np.float32)

    def reset(self, seed=None):
        self._returns = np.zeros((self.num_envs, self.n_rewards), dtype=np.float64)
        return self.env.reset(seed=seed)

    def step(self, actions):
        obs, reward, cost, done, info = self.env.step(actions)
        reward = np.asarray(reward)
        if self.training:

            not_done = (1.0 - np.asarray(done, dtype=np.float64))[:, None]
            self._returns = self._returns * self.gamma * not_done + reward
            self.return_rms.update(self._returns)
        norm_reward = self.normalize(reward)

        info = dict(info)
        info['reward_vec'] = reward.astype(np.float32)
        return obs, norm_reward, cost, done, info

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def save_stats(self) -> dict:
        return {
            'mean': np.asarray(self.return_rms.mean).copy(),
            'var': np.asarray(self.return_rms.var).copy(),
            'count': float(self.return_rms.count),
        }

    def load_stats(self, stats: dict) -> None:
        self.return_rms.mean = np.asarray(stats['mean'], dtype=np.float64).copy()
        self.return_rms.var = np.asarray(stats['var'], dtype=np.float64).copy()
        self.return_rms.count = float(stats['count'])
