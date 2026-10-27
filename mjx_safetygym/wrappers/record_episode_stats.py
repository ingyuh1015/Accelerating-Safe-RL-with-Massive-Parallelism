from __future__ import annotations

from collections import deque

import numpy as np


class RecordEpisodeStatistics:


    def __init__(self, env, deque_size: int = 100):
        self.env = env
        self.num_envs = env.num_envs
        self.n_rewards = env.n_rewards
        self.n_costs = env.n_costs

        self._return = np.zeros((self.num_envs, self.n_rewards), dtype=np.float64)
        self._cost = np.zeros((self.num_envs, self.n_costs), dtype=np.float64)
        self._length = np.zeros((self.num_envs,), dtype=np.int64)


        self.return_queue = deque(maxlen=deque_size)
        self.cost_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, seed=None):
        self._return = np.zeros((self.num_envs, self.n_rewards), dtype=np.float64)
        self._cost = np.zeros((self.num_envs, self.n_costs), dtype=np.float64)
        self._length = np.zeros((self.num_envs,), dtype=np.int64)
        return self.env.reset(seed=seed)

    def step(self, actions):
        obs, reward, cost, done, info = self.env.step(actions)
        reward = np.asarray(reward)
        cost = np.asarray(cost)
        done_mask = np.asarray(done) > 0.5


        self._return += reward
        self._cost += cost
        self._length += 1

        info = dict(info)
        if done_mask.any():
            ret_done = self._return[done_mask].astype(np.float32)
            cost_done = self._cost[done_mask].astype(np.float32)
            len_done = self._length[done_mask].astype(np.int64)
            info['episode'] = {
                'r': ret_done,
                'c': cost_done,
                'l': len_done,
                'done_mask': done_mask,
            }

            self.return_queue.extend(ret_done.mean(axis=-1).tolist())
            self.cost_queue.extend(cost_done.mean(axis=-1).tolist())
            self.length_queue.extend(len_done.tolist())


            self._return[done_mask] = 0.0
            self._cost[done_mask] = 0.0
            self._length[done_mask] = 0

        return obs, reward, cost, done, info
