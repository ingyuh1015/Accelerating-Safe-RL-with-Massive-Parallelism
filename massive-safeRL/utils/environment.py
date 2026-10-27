from __future__ import annotations

from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
from massive_saferl import EnvState, TaskEnvConfig, get_dtype

if TYPE_CHECKING:
    from mjx_safetygym.bases.underlying import State


class Environment:


    reset_conditional = staticmethod(jax.lax.cond)

    def __init__(self, task, *, episode_length: int, batch_size: int, gamma: float):
        if episode_length < 1 or batch_size < 1:
            raise ValueError('episode_length and batch_size must be positive')
        self.task = task
        self.variable_episode = bool(getattr(task, 'is_variable_episode', False))
        self.episode_length = int(episode_length)
        self.batch_size = int(batch_size)
        self.gamma = float(gamma)
        self.observation_size = int(task.obs_dim)
        self.action_size = int(task.world.action_dim)
        self.constraint_names = task.constraint_names
        self.cost_limits = task.cost_limits
        self.mj_model = task.world.mj_model
        self.mjx_model = task.mjx_model

    @staticmethod
    def select_state(mask, new, old):
        def select(a, b):
            shape = mask.shape + (1,) * (a.ndim - mask.ndim)
            return jnp.where(mask.reshape(shape), a, b)
        return jax.tree.map(select, new, old)

    def _from_task_state(self, state: State) -> EnvState:
        dtype = get_dtype()
        done = state.done.astype(dtype)
        return EnvState(
            data=state.data, obs=state.obs.astype(dtype),
            reward=jnp.sum(state.reward, axis=-1).astype(dtype), cost_vector=state.cost.astype(dtype),
            done=done, terminated=done, truncated=jnp.zeros_like(done),
            info={'task_info': state.info, 'task_rng': state.key, 'task_reward': state.reward},
        )

    def _reset_envs(self, rng: jax.Array) -> EnvState:
        keys = jax.random.split(rng, self.batch_size) if jax.random.key_data(rng).ndim == 1 else rng
        return self._from_task_state(jax.vmap(self.task.reset)(keys))

    def _step_envs(self, state: EnvState, action: jax.Array) -> EnvState:
        from mjx_safetygym.bases.underlying import State
        if action.ndim == 1:
            action = jnp.broadcast_to(action, (self.batch_size, self.action_size))
        task_state = State(
            data=state.data, obs=state.obs, reward=state.info['task_reward'],
            cost=state.cost_vector, done=state.done, key=state.info['task_rng'],
            info=state.info['task_info'],
        )
        return self._from_task_state(jax.vmap(self.task.step)(task_state, action))

    def reset(self, rng: jax.Array) -> EnvState:
        (rng_env, rng_autoreset) = jax.random.split(rng)
        state = self._reset_envs(rng_env)
        zeros = jnp.zeros_like(jnp.asarray(state.done, dtype=get_dtype()))
        state.info['steps'] = jnp.zeros_like(jnp.asarray(state.done, dtype=jnp.int32))
        state.info['truncation'] = zeros
        state.info['episode_done'] = zeros
        state.info['terminal_cause'] = jnp.zeros_like(jnp.asarray(state.done, dtype=jnp.int8))
        step_cost = jnp.asarray(state.cost_vector, dtype=get_dtype())
        state.info['step_cost'] = step_cost
        metric_zeros = jnp.zeros(state.done.shape, dtype=jnp.float32)
        cost_zeros = jnp.zeros(step_cost.shape, dtype=jnp.float32)
        state.info['discounted_rewards'] = zeros
        state.info['discounted_costs'] = jnp.zeros_like(step_cost)
        episode_metrics = {'sum_reward': metric_zeros, 'sum_d_reward': metric_zeros, 'length': metric_zeros, 'sum_r_costs': cost_zeros, 'sum_d_costs': cost_zeros}
        state.info['episode_metrics'] = episode_metrics
        state.info['terminal_episode_metrics'] = episode_metrics
        if self.variable_episode:
            state.info['completed_episode_sums'] = {**episode_metrics, 'count': jnp.zeros_like(state.info['steps'])}
        num_envs = state.done.shape[0]
        state.info['rng'] = jax.random.split(rng_autoreset, num_envs)
        state.info['terminal_obs'] = state.obs
        return state.replace(terminated=jnp.zeros_like(state.done), truncated=jnp.zeros_like(state.done), done=jnp.zeros_like(state.done))

    def _auto_reset_done_envs(self, *, state: EnvState, done: jax.Array, done_mask: jax.Array, terminated: jax.Array, truncated: jax.Array, rng_reset: jax.Array, rng_next: jax.Array) -> EnvState:

        def _reset_branch(args):
            (current_state, reset_keys, current_done_mask) = args
            reset_state = self._reset_envs(reset_keys)

            aligned_info = {key: reset_state.info[key] if key in reset_state.info else jax.tree.map(jnp.zeros_like, current_state.info[key]) for key in current_state.info}
            reset_state = reset_state.replace(info=aligned_info)
            return self.select_state(current_done_mask, reset_state, current_state)

        def _no_reset_branch(args):
            (current_state, _, _) = args
            return current_state
        auto_reset_state = self.reset_conditional(jnp.any(done_mask), _reset_branch, _no_reset_branch, (state, rng_reset, done_mask))
        auto_reset_state.info['terminal_obs'] = state.info['terminal_obs']
        auto_reset_state.info['terminal_episode_metrics'] = state.info['terminal_episode_metrics']
        if self.variable_episode:
            auto_reset_state.info['completed_episode_sums'] = state.info['completed_episode_sums']
        for name in ('step_cost', 'discounted_rewards', 'discounted_costs'):
            auto_reset_state.info[name] = state.info[name]
        auto_reset_state.info['rng'] = rng_next
        auto_reset_state.info['steps'] = jnp.where(done_mask, jnp.zeros_like(state.info['steps']), state.info['steps'])
        auto_reset_state.info['episode_done'] = jnp.where(done_mask, jnp.zeros_like(done), done)
        return auto_reset_state.replace(reward=state.reward, cost_vector=state.cost_vector, done=done, terminated=terminated, truncated=truncated)

    def step(self, state: EnvState, action: jax.Array, *, return_transition: bool = False):
        per_env_rng = state.info['rng']
        rng_pairs = jax.vmap(jax.random.split)(per_env_rng)
        rng_next = rng_pairs[:, 0]
        rng_reset = rng_pairs[:, 1]
        nstate = self._step_envs(state, action)
        merged_info = dict(state.info)
        merged_info.update(nstate.info)
        state = nstate.replace(info=merged_info)
        prev_steps = state.info['steps']
        steps = prev_steps + 1
        base_term = jnp.asarray(state.terminated, dtype=get_dtype())
        base_trunc = jnp.asarray(state.truncated, dtype=get_dtype())
        zero = jnp.zeros_like(base_term)
        episode_length = jnp.array(self.episode_length, dtype=jnp.int32)
        time_trunc = jnp.where(steps >= episode_length, 1.0 - jnp.maximum(base_term, base_trunc), zero)
        terminated = base_term
        truncated = jnp.maximum(base_trunc, time_trunc)
        done = jnp.maximum(terminated, truncated)
        state.info['truncation'] = jnp.where(steps >= episode_length, time_trunc, zero)
        state.info['steps'] = steps
        inner_tc = state.info.get('terminal_cause', zero)
        state.info['terminal_cause'] = jnp.where(time_trunc > 0.5, jnp.int8(4), inner_tc.astype(jnp.int8))
        prev_done = state.info['episode_done']
        alive = 1 - prev_done
        gamma_power = jnp.power(jnp.asarray(self.gamma, dtype=get_dtype()), prev_steps)
        step_cost = jnp.asarray(state.cost_vector, dtype=get_dtype())
        state.info['step_cost'] = step_cost
        state.info['discounted_rewards'] = gamma_power * state.reward
        state.info['discounted_costs'] = gamma_power[..., None] * step_cost
        em = dict(state.info['episode_metrics'])
        em['sum_reward'] = (em['sum_reward'] + state.reward) * alive
        em['sum_d_reward'] = (em['sum_d_reward'] + state.info['discounted_rewards']) * alive
        em['length'] = (em['length'] + 1) * alive
        em['sum_r_costs'] = (em['sum_r_costs'] + step_cost) * alive[..., None]
        em['sum_d_costs'] = (em['sum_d_costs'] + state.info['discounted_costs']) * alive[..., None]
        state.info['episode_metrics'] = em
        state.info['episode_done'] = done
        state = getattr(state, 'replace')(done=done, terminated=terminated, truncated=truncated)
        state.info['terminal_obs'] = state.obs
        done_mask = done.astype(bool)

        def _where_terminal(new_val, old_val):
            mask = done_mask.reshape(done_mask.shape + (1,) * (new_val.ndim - done_mask.ndim))
            return jnp.where(mask, new_val, old_val)
        state.info['terminal_episode_metrics'] = jax.tree.map(_where_terminal, state.info['episode_metrics'], state.info['terminal_episode_metrics'])
        if self.variable_episode:
            completed = state.info['completed_episode_sums']
            state.info['completed_episode_sums'] = {
                name: completed[name] + _where_terminal(value, jnp.zeros_like(value))
                for name, value in state.info['episode_metrics'].items()}
            state.info['completed_episode_sums']['count'] = completed['count'] + done.astype(jnp.int32)
        next_state = self._auto_reset_done_envs(state=state, done=done, done_mask=done_mask, terminated=terminated, truncated=truncated, rng_reset=rng_reset, rng_next=rng_next)
        return (next_state, state) if return_transition else next_state


def build_env(*, task_env_config: TaskEnvConfig, episode_length: int, batch_size: int, gamma: float) -> Environment:
    from massive_saferl.utils.task import build_task
    task = build_task(dict(task_env_config.task_config))
    return Environment(task, episode_length=episode_length, batch_size=batch_size, gamma=gamma)
