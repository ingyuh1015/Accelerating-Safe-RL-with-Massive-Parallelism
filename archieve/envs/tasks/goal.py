from __future__ import annotations
from typing import Any, cast
import jax
import jax.numpy as jnp
from archieve import get_dtype
from archieve.envs import sample_point
from archieve.envs.tasks import NavEnv
_dtype = get_dtype

class GoalEnv(NavEnv):

    def __init__(self, **kwargs: Any) -> None:
        obstacle_exclude = {'goal', 'circle', 'box', 'gremlins'}
        super().__init__(**kwargs)
        goal_cfg = self.objects['goal']
        self._goal_radius = float(goal_cfg.size)
        self._goal_keepout = float(goal_cfg.keepout)
        self._spawn_radius = float(goal_cfg.spawn_radius) or self._default_spawn_radius
        self._agent_keepout = float(self.agent.keepout)
        self._obstacle_cfgs = tuple((c for (n, c) in self.objects.items() if n not in obstacle_exclude))
        self._empty_obs = jnp.zeros((0, 2), dtype=_dtype())

    def _sample_goal(self, rng: jax.Array, agent_pos: jnp.ndarray, obstacles: jnp.ndarray, max_attempts: int=20) -> jnp.ndarray:
        spawn_radius = self._spawn_radius
        keepout = self._goal_keepout
        agent_keepout = self._agent_keepout
        (rng, sub) = jax.random.split(rng)
        pos0 = sample_point(sub, spawn_radius)
        min_agent_dist_sq = jnp.asarray(keepout + agent_keepout, dtype=get_dtype()) ** 2
        min_obstacle_dist_sq = jnp.asarray(keepout, dtype=get_dtype()) ** 2

        def attempt(carry: tuple[jax.Array, jnp.ndarray], _unused: None) -> tuple[tuple[jax.Array, jnp.ndarray], None]:
            (key, pos) = carry
            (key, subkey) = jax.random.split(key)
            cand = sample_point(subkey, spawn_radius)
            agent_delta = cand - agent_pos
            obstacle_delta = obstacles - cand
            valid = (jnp.sum(agent_delta * agent_delta) > min_agent_dist_sq) & jnp.all(jnp.sum(obstacle_delta * obstacle_delta, axis=-1) > min_obstacle_dist_sq)
            pos = jnp.where(valid, cand, pos)
            return ((key, pos), None)
        ((rng, pos), _) = jax.lax.scan(attempt, (rng, pos0), None, length=max_attempts)
        return cast(jnp.ndarray, pos)

    def reset_objects(self, data: Any, info: dict, rng: Any) -> Any:
        rng_goal = jax.random.split(rng, 2)[0]
        info['target_pos'] = self._sample_goal(rng_goal, info['agent_pos'], self._empty_obs)
        info['goal_distance'] = jnp.linalg.norm(info['target_pos'] - info['agent_pos'])
        info['goal_reached'] = jnp.zeros((), dtype=_dtype())
        info['prev_goal_distance'] = info['goal_distance']
        return super().reset_objects(data, info, rng)

    def compute_reward(self, data: Any, info: dict) -> Any:
        dtype = _dtype()
        target = jnp.asarray(info['target_pos'], dtype=dtype)
        agent = jnp.asarray(info['agent_pos'], dtype=dtype)
        goal_dist = jnp.linalg.norm(target - agent)
        reached = (goal_dist < jnp.asarray(self._goal_radius, dtype=dtype)).astype(dtype)
        progress = jnp.asarray(info['prev_goal_distance'], dtype=dtype) - goal_dist
        info['goal_distance'] = goal_dist
        info['goal_reached'] = reached
        info['goal_progress'] = progress
        info['reward'] = progress + reached
        parts = [c.get_positions(data, info) for c in self._obstacle_cfgs] or [self._empty_obs]
        obstacles = jnp.concatenate(parts, axis=0)
        (rng_next, sample_rng) = jax.random.split(jnp.asarray(info['goal_rng']))
        new_goal = self._sample_goal(sample_rng, info['agent_pos'], obstacles)
        target_pos = jnp.where(reached.astype(jnp.bool_)[..., None], new_goal, target)
        info['goal_rng'] = rng_next
        info['target_pos'] = target_pos
        info['goal_distance'] = jnp.linalg.norm(target_pos - agent)
        info['prev_goal_distance'] = info['goal_distance']
        return data
