from __future__ import annotations
import itertools
from typing import Any
import jax
import jax.numpy as jnp
from archieve.envs import MjxEnv

class NavEnv(MjxEnv):

    def __init__(self, *, lidar_num_bins: int, lidar_max_dist: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lidar_num_bins = lidar_num_bins
        self._lidar_max_dist = lidar_max_dist
        self._lidar_cfgs = tuple((cfg for cfg in self.objects.values() if cfg.lidar_observable))
        self._obs_size = self._compute_obs_size()

    def _compute_obs_size(self) -> int:
        return self.agent.agent_obs_size + len(self._lidar_cfgs) * self._lidar_num_bins

    def compute_obs(self, data: Any, info: dict) -> jnp.ndarray:
        position_groups = tuple(cfg.get_positions(data, info)[:, :2] for cfg in self._lidar_cfgs)
        agent_pos = info['agent_pos']
        agent_theta = self.agent.compute_heading(data)
        dummy = jnp.array([[jnp.inf, jnp.inf]], dtype=self._dtype)
        padded_groups = [jnp.concatenate([g, dummy], axis=0) for g in position_groups]
        group_sizes = tuple((g.shape[0] for g in padded_groups))
        all_positions = jnp.concatenate(padded_groups, axis=0)
        num_bins = self._lidar_num_bins
        max_dist = self._lidar_max_dist
        rel = all_positions - agent_pos
        dists = jnp.linalg.norm(rel, axis=-1)
        angles = jnp.arctan2(rel[:, 1], rel[:, 0]) - agent_theta
        angles = jnp.mod(angles, 2 * jnp.pi)
        bin_size = 2 * jnp.pi / num_bins
        clamped_dists = jnp.clip(dists, 0.0, max_dist)
        sensor = 1.0 - clamped_dists / max_dist
        bin_main = jnp.floor(angles / bin_size).astype(jnp.int32)
        alias_frac = (angles - bin_main.astype(self._dtype) * bin_size) / bin_size
        bin_plus = jnp.mod(bin_main + 1, num_bins)
        bin_minus = jnp.mod(bin_main - 1, num_bins)
        main_oh = jax.nn.one_hot(bin_main, num_bins, dtype=self._dtype)
        plus_oh = jax.nn.one_hot(bin_plus, num_bins, dtype=self._dtype)
        minus_oh = jax.nn.one_hot(bin_minus, num_bins, dtype=self._dtype)
        main_vals = main_oh * sensor[:, None]
        plus_vals = plus_oh * (alias_frac * sensor)[:, None]
        minus_vals = minus_oh * ((1.0 - alias_frac) * sensor)[:, None]
        all_weighted = jnp.maximum(jnp.maximum(main_vals, plus_vals), minus_vals)
        offsets = tuple(itertools.accumulate((0,) + group_sizes[:-1]))
        per_group = tuple((jnp.max(all_weighted[o:o + s], axis=0) for (o, s) in zip(offsets, group_sizes)))
        lidar = jnp.concatenate(per_group, axis=0)
        agent_obs = self.agent.compute_agent_obs(data).astype(self._dtype)
        agent_size = self.agent.agent_obs_size
        obs = jnp.zeros((self._obs_size,), dtype=self._dtype)
        obs = obs.at[:agent_size].set(agent_obs)
        obs = obs.at[agent_size:].set(lidar.astype(self._dtype))
        return obs
