from __future__ import annotations
import math
import os
from typing import Any
import jax
import jax.numpy as jnp
import optax
from archieve.algos import OnPolicyBase, ValueCritic, gaussian_kl
from archieve.utils import build_optimizer
from archieve import get_dtype, EnvState, OnBatchState, PIDRWeights, Metrics, RuntimeState
from archieve.envs import Wrapper
from archieve.train import RolloutBuffer, empirical_bernstein_radius, compute_pidr_weighted_upper, compute_raw_phase_mixture_tightening, normalize_pidr_weights

_POLICY_FREEZE_AFTER = int(os.environ.get('POLICY_FREEZE_AFTER', '-1'))

class Pipeline:

    def __init__(self, *, policy: OnPolicyBase, env: Wrapper) -> None:
        self.policy = policy
        self.env = env
        self.rollout_buf = RolloutBuffer(num_envs=policy.cfg.num_envs, sample_length=policy.cfg.rollout_horizon, zero_batch_state=OnBatchState.build_zero(policy.cfg.num_envs, policy.cfg.obs_size, policy.cfg.action_size, policy.cfg.num_constraints))
        self._cost_surrogate = bool(getattr(policy.cfg, 'cost_surrogate', False))
        cfg = policy.cfg
        self._tightening_mode = str(getattr(cfg, 'tightening_mode', 'off'))
        if self._tightening_mode == 'on':
            if int(cfg.num_constraints) < 1:
                raise ValueError("tightening_mode='on' requires at least one constraint")

            self._tightening_bc = float(cfg.tightening_bc)
            self._tightening_delta_conf = float(getattr(cfg, 'tightening_delta_conf', 0.05))
            self._tightening_cmax = jnp.asarray(getattr(cfg, 'tightening_cmax', 1.0), dtype=get_dtype())
            self._tightening_max = jnp.asarray(cfg.tightening_max, dtype=get_dtype())
            self._tightening_gamma = float(cfg.gamma) if bool(getattr(cfg, 'discounted', False)) else 1.0
            self._tighten_terms = str(getattr(cfg, 'tighten_terms', 'both'))
            self._tightening_delta_event = self._tightening_delta_conf / float(2 * int(cfg.num_constraints))

            self._tightening_ratio_clip = 3.0
            self._tightening_pidr_lr = 1e-3

            self._tightening_pidr_steps = 40
            self._tightening_pidr_r2_regularizer = 0.5

            self._tightening_pidr_anchor = float(os.environ.get("PIDR_ANCHOR", "0.0"))

            self._pidr_holdout = os.environ.get("PIDR_HOLDOUT", "0") not in ("", "0", "false")

            self._pidr_force_w1 = os.environ.get("PIDR_FORCE_W1", "0") not in ("", "0", "false")
            self._tightening_pidr_eps = 1e-6

            self._pidr_shrink = False

            hs = tuple(cfg.hidden_sizes)
            dtype = get_dtype()

            self._pidr_phase_fourier = max(0, int(os.environ.get("PIDR_PHASE_FOURIER", "0")))
            self._pidr_input_size = int(cfg.obs_size) + 1 + 2 * self._pidr_phase_fourier
            self.pidr_f = ValueCritic(hidden_sizes=hs, activation=cfg.activation, weight_init=cfg.weight_init, layer_norm=cfg.layer_norm, compute_dtype=dtype)
            self.pidr_ratio = ValueCritic(hidden_sizes=hs, activation=cfg.activation, weight_init=cfg.weight_init, layer_norm=cfg.layer_norm, compute_dtype=dtype)
            self.pidr_f_opt = build_optimizer(lr=self._tightening_pidr_lr, max_grad_norm=cfg.max_grad_norm)
            self.pidr_ratio_opt = build_optimizer(lr=self._tightening_pidr_lr, max_grad_norm=cfg.max_grad_norm)

            self._flow_probe_dim = 8
            _probe_key = jax.random.PRNGKey(12345)
            self._flow_probe_W = jax.random.normal(
                _probe_key, (self._pidr_input_size, self._flow_probe_dim), dtype=dtype
            ) / jnp.sqrt(jnp.asarray(float(self._pidr_input_size), dtype=dtype))
        else:
            self._tightening_bc = None
            self._tightening_delta_conf = None
            self._tightening_cmax = None
            self._tightening_max = None
            self._tightening_gamma = None
            self._tighten_terms = 'both'
            self._tightening_delta_event = None
            self._tightening_ratio_clip = 3.0

    def actor_obs(self, policy_state: Any, obs: jax.Array) -> jax.Array:
        return policy_state.obs_rms.normalize(obs) if self.policy.cfg.obs_normalize else obs

    def init_runtime_state(self, rng: jax.Array) -> RuntimeState:
        key_init, key_env, key_next = jax.random.split(rng, 3)
        policy_state = self.policy.init_state(key_init).replace(rng=key_next)
        return RuntimeState(policy_state=policy_state, env_state=self.env.reset(key_env), rollout_buf_state=self.rollout_buf.init())

    def _staggered_init(self, runtime_state: RuntimeState) -> RuntimeState:
        cfg = self.policy.cfg
        num_envs = int(runtime_state.env_state.obs.shape[0])
        num_groups = min(cfg.episode_length // cfg.rollout_horizon, num_envs)
        envs_per_group = num_envs // num_groups
        group_idx = jnp.minimum(jnp.arange(num_envs) // envs_per_group, num_groups - 1)
        offsets = group_idx * cfg.rollout_horizon
        rollout_steps = (num_groups - 1) * cfg.rollout_horizon
        policy_state = runtime_state.policy_state

        def _step(carry, step_idx):
            env_st, rng, buf_st = carry
            rng, rng_act = jax.random.split(rng)
            obs = env_st.obs
            steps = env_st.info['steps']
            actor_obs = self.actor_obs(policy_state, obs)
            action, sample_stats = self.policy.actor.sample_action(policy_state.actor_params, actor_obs, rng_act, policy_state.obs_rms)
            next_env = self.env.step(env_st, action)
            active = step_idx < offsets
            batch_state = OnBatchState.from_step(obs, action, next_env, steps, sample_stats)
            next_env = jax.tree.map(lambda new, old: jnp.where(active.reshape(active.shape + (1,) * (new.ndim - active.ndim)), new, old), next_env, env_st)
            buf_st = self.rollout_buf.add_masked(buf_st, batch_state, active)
            return (next_env, rng, buf_st), None

        (final_env, final_rng, final_buf), _ = jax.lax.scan(_step, (runtime_state.env_state, policy_state.rng, runtime_state.rollout_buf_state), jnp.arange(rollout_steps))
        return runtime_state.replace(policy_state=policy_state.replace(rng=final_rng), env_state=final_env, rollout_buf_state=final_buf)

    def _rollout_env(self, runtime_state: RuntimeState) -> RuntimeState:
        policy_state = runtime_state.policy_state

        def _step(carry, _):
            env_st, rng, buf_st = carry
            rng, rng_act = jax.random.split(rng)
            obs = env_st.obs
            steps = env_st.info['steps']
            actor_obs = self.actor_obs(policy_state, obs)
            action, sample_stats = self.policy.actor.sample_action(policy_state.actor_params, actor_obs, rng_act, policy_state.obs_rms)
            next_env = self.env.step(env_st, action)
            batch_state = OnBatchState.from_step(obs, action, next_env, steps, sample_stats)
            return (next_env, rng, self.rollout_buf.add(buf_st, batch_state)), None

        (final_env, final_rng, final_buf), _ = jax.lax.scan(_step, (runtime_state.env_state, policy_state.rng, runtime_state.rollout_buf_state), None, length=self.policy.cfg.rollout_horizon)
        return runtime_state.replace(policy_state=policy_state.replace(rng=final_rng), env_state=final_env, rollout_buf_state=final_buf)

    def _compute_episode_metrics(self, *, batch: OnBatchState, env_state: EnvState) -> dict[str, jax.Array]:
        cfg = self.policy.cfg
        num_groups = cfg.rollout_steps_per_update / cfg.episode_length
        costs = batch.discounted_costs if cfg.discounted else batch.costs
        cost_signal = jnp.sum(costs, axis=(0, 1)) / num_groups
        done_f = jnp.any(batch.dones > 0, axis=0).astype(get_dtype())
        safe_count = jnp.maximum(jnp.sum(done_f), 1.0)
        terminal = env_state.info['terminal_episode_metrics']
        cast = lambda x: jnp.asarray(x, dtype=get_dtype())
        if getattr(cfg, 'cost_signal_source', 'rollout') == 'terminal_recent_raw':
            cost_signal = jnp.mean(cast(terminal['sum_r_costs']), axis=0)
        return {'Ep/reward': jnp.sum(batch.rewards) / num_groups, 'Ep/d_reward': jnp.sum(batch.discounted_rewards) / num_groups, 'Ep/r_cost': jnp.sum(batch.costs, axis=(0, 1)) / num_groups, 'Ep/d_cost': jnp.sum(batch.discounted_costs, axis=(0, 1)) / num_groups, 'cost_signal': cost_signal, 'cost_limit': cfg.cost_limits, 'Epd/reward': jnp.sum(cast(terminal['sum_reward']) * done_f) / safe_count, 'Epd/d_reward': jnp.sum(cast(terminal['sum_d_reward']) * done_f) / safe_count, 'Epd/r_cost': jnp.sum(cast(terminal['sum_r_costs']) * done_f[..., None], axis=0) / safe_count, 'Epd/d_cost': jnp.sum(cast(terminal['sum_d_costs']) * done_f[..., None], axis=0) / safe_count}

    def prepare_runtime_state(self, runtime_state: RuntimeState) -> RuntimeState:
        if self._tightening_mode == 'on':

            ps = runtime_state.policy_state
            (ps_rng, f_rng, r_rng) = jax.random.split(ps.rng, 3)
            pidr_obs = jnp.zeros((self._pidr_input_size,), dtype=get_dtype())
            f_params = self.pidr_f.init(f_rng, pidr_obs)
            ratio_params = self.pidr_ratio.init(r_rng, pidr_obs)
            baseline = jnp.zeros((int(self.policy.cfg.episode_length), int(self.policy.cfg.num_constraints)), dtype=get_dtype())
            ps = ps.replace(rng=ps_rng, pidr_f_params=f_params, pidr_ratio_params=ratio_params, pidr_f_opt=self.pidr_f_opt.init(f_params), pidr_ratio_opt=self.pidr_ratio_opt.init(ratio_params), pidr_cost_baseline=baseline)
            runtime_state = runtime_state.replace(policy_state=ps)
        if str(getattr(self.policy.cfg, 'rollout_mode', 'standard')) == 'staggered':
            return self._staggered_init(runtime_state)
        return runtime_state

    def compute_advantages(self, batch):
        cfg = self.policy.cfg
        num_constraints = cfg.num_constraints
        adv_standardize = getattr(cfg, 'adv_standardize', True)
        step_weight = getattr(cfg, 'step_weight', False)
        (_, advantages) = self._compute_targets(batch)
        if adv_standardize:
            reward_slot = advantages[..., 0]
            reward_slot = (reward_slot - jnp.mean(reward_slot)) / (jnp.std(reward_slot) + 1e-08)
            advantages = advantages.at[..., 0].set(reward_slot)
            if num_constraints > 0:
                cost_mean = jnp.mean(advantages[..., 1:], axis=(0, 1), keepdims=True)
                advantages = advantages.at[..., 1:].add(-cost_mean)
        if step_weight:
            T = advantages.shape[0]
            steps = jnp.arange(T, dtype=advantages.dtype)[:, None]
            gamma_pow = jnp.power(jnp.asarray(cfg.gamma, dtype=advantages.dtype), steps)
            advantages = advantages * gamma_pow[..., None]
        return batch.replace(advantages=advantages)

    def _batch_critic_values(self, obs: jnp.ndarray, next_obs: jnp.ndarray, critic_params: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
        critic_apply = self.policy.critic.apply
        num_constraints = self.policy.cfg.num_constraints
        (T, B) = obs.shape[:2]
        stacked_obs = jnp.concatenate([obs, next_obs], axis=0)
        flat = stacked_obs.reshape(-1, stacked_obs.shape[-1])
        all_values = jax.vmap(lambda p: critic_apply(p, flat))(critic_params)
        all_2d = jnp.moveaxis(all_values, 0, -1).reshape(2 * T, B, 1 + num_constraints)
        return (all_2d[:T], all_2d[T:])

    def _compute_targets(self, batch):
        cfg = self.policy.cfg
        num_constraints = cfg.num_constraints
        reward_gamma, reward_lambda, cost_lambda = cfg.gamma, cfg.gae_lambda, cfg.cost_gae_lambda
        finite = getattr(cfg, 'horizon', 'infinite') == 'finite'
        if finite:
            bootstrap_mask = 1.0 - jnp.asarray(batch.dones)
        else:
            bootstrap_mask = 1.0 - batch.terminated

        continue_mask = 1.0 - jnp.asarray(batch.dones)
        signals = jnp.concatenate([batch.rewards[..., None], batch.costs], axis=-1)
        signals_t = jnp.moveaxis(signals, -1, 0)
        values_t = jnp.moveaxis(batch.values, -1, 0)
        next_values_t = jnp.moveaxis(batch.next_values, -1, 0)
        bootstrap_t = jnp.moveaxis(batch.bootstrap, -1, 0)
        lambdas = jnp.concatenate([jnp.asarray([reward_lambda], dtype=signals.dtype), jnp.full((num_constraints,), cost_lambda, dtype=signals.dtype)])

        def _gae_slot(r, v, nv, b, lam):
            deltas = r + reward_gamma * bootstrap_mask * nv - v
            discounts = reward_gamma * lam * continue_mask

            def scan_fn(gae, step):
                (delta, discount) = step
                gae = delta + discount * gae
                return (gae, gae)
            (_, advantages) = jax.lax.scan(scan_fn, jnp.zeros_like(b), (deltas, discounts), reverse=True, unroll=16)
            return (advantages + v, advantages)
        (returns_t, advs_t) = jax.vmap(_gae_slot)(signals_t, values_t, next_values_t, bootstrap_t, lambdas)
        returns = jnp.moveaxis(returns_t, 0, -1)
        advs = jnp.moveaxis(advs_t, 0, -1)
        return (returns, advs)

    def _critic_outputs(self, policy_state: Any, obs: jax.Array, next_obs: jax.Array) -> dict[str, Any]:
        (values, next_values) = self._batch_critic_values(obs, next_obs, policy_state.critic_params)
        return {'values': values, 'next_values': next_values, 'bootstrap': next_values[-1]}

    def prepare_batch_for_critic(self, runtime_state: RuntimeState, *, collected_rollout: OnBatchState, cost_signal: jax.Array, cost_limit: jax.Array) -> OnBatchState:
        policy_state = runtime_state.policy_state
        normalized_obs = self.actor_obs(policy_state, collected_rollout.obs)
        next_obs = self.actor_obs(policy_state, collected_rollout.next_obs)
        critic_out = self._critic_outputs(policy_state, normalized_obs, next_obs)
        rollout = collected_rollout.replace(obs=normalized_obs, next_obs=next_obs, cost_signal=cost_signal, cost_limit=cost_limit, **critic_out)
        returns, _ = self._compute_targets(rollout)
        return rollout.replace(returns=returns)

    def prepare_actor_advantages(self, runtime_state: RuntimeState, rollout: OnBatchState) -> OnBatchState:
        policy_state = runtime_state.policy_state
        rollout = rollout.replace(**self._critic_outputs(policy_state, rollout.obs, rollout.next_obs))
        return self.compute_advantages(rollout)

    def _pidr_features(self, obs_ND: jnp.ndarray, phase_N: jnp.ndarray) -> jnp.ndarray:

        obs = jnp.asarray(obs_ND, dtype=get_dtype())
        dtype = get_dtype()
        H = jnp.asarray(max(1, int(self.policy.cfg.episode_length) - 1), dtype=dtype)
        unit = jnp.asarray(phase_N, dtype=dtype).reshape((-1, 1)) / H
        phase = (unit - jnp.asarray(0.5, dtype=dtype)) * jnp.asarray(3.4641016151377544, dtype=dtype)
        K = int(getattr(self, "_pidr_phase_fourier", 0))
        if K <= 0:
            return jnp.concatenate([obs, phase], axis=-1)
        k = jnp.arange(1, K + 1, dtype=dtype).reshape((1, K))
        ang = jnp.asarray(2.0 * math.pi, dtype=dtype) * unit * k
        root2 = jnp.asarray(1.4142135623730951, dtype=dtype)
        return jnp.concatenate(
            [obs, phase, root2 * jnp.sin(ang), root2 * jnp.cos(ang)], axis=-1,
        )

    @staticmethod
    def _phase_means(samples_NC: jnp.ndarray, phase_N: jnp.ndarray, episode_length: int) -> jnp.ndarray:

        dtype = get_dtype()
        samples = jnp.asarray(samples_NC, dtype=dtype)
        phases = jnp.asarray(phase_N, dtype=jnp.int32).reshape(-1)
        counts = jnp.bincount(phases, length=int(episode_length)).astype(dtype)
        sums = jax.vmap(lambda col: jnp.bincount(phases, weights=col, length=int(episode_length)).astype(dtype),
                        in_axes=1, out_axes=1)(samples)
        return sums / jnp.maximum(counts, jnp.asarray(1.0, dtype=dtype))[:, None]

    def _train_pidr_weights(self, policy_state: Any, rollout: OnBatchState) -> tuple[Any, PIDRWeights, dict[str, jnp.ndarray]]:

        policy = self.policy
        dtype = get_dtype()
        start_phase_N = jnp.clip(rollout.steps[0].astype(jnp.int32), 0, int(policy.cfg.episode_length) - 1)
        next_phase_N = jnp.mod(start_phase_N + int(rollout.obs.shape[0]), int(policy.cfg.episode_length))
        valid_N = (next_phase_N > start_phase_N).astype(dtype)

        env_idx_N = jnp.arange(start_phase_N.shape[0], dtype=jnp.int32)
        if self._pidr_holdout:
            train_sel_N = (jnp.mod(env_idx_N, 2) == 0).astype(dtype)
        else:
            train_sel_N = jnp.ones_like(start_phase_N, dtype=dtype)

        (vals_TNc, next_vals_TNc) = self._batch_critic_values(
            rollout.obs, rollout.next_obs, policy_state.critic_params)
        cost_value_N = jnp.mean(vals_TNc[0][..., 1:], axis=-1)
        cost_value_next_N = jnp.mean(next_vals_TNc[-1][..., 1:], axis=-1)
        x_ND = self._pidr_features(rollout.obs[0], start_phase_N)
        x_next_ND = self._pidr_features(rollout.next_obs[-1], next_phase_N)
        r2_regularizer = jnp.asarray(self._tightening_pidr_r2_regularizer, dtype=dtype)
        anchor_coef = jnp.asarray(self._tightening_pidr_anchor, dtype=dtype)
        eps = jnp.asarray(self._tightening_pidr_eps, dtype=dtype)
        phase_count = int(policy.cfg.episode_length)

        def positive_ratio(params, features, phase_B):

            logits = self.pidr_ratio.apply(params, features)
            counts = jnp.bincount(phase_B, length=phase_count).astype(dtype)

            gmax = jax.ops.segment_max(logits, phase_B, num_segments=phase_count)
            z = jnp.exp(logits - jax.lax.stop_gradient(gmax)[phase_B])
            gsum = jnp.bincount(phase_B, weights=z, length=phase_count).astype(dtype)
            return z * counts[phase_B] / jnp.maximum(gsum[phase_B], eps)

        holdout = self._pidr_holdout

        def loss_parts(f_params, ratio_params, minibatch):
            (x_BD, x_next_BD, phase_B, valid_B, sel_B) = minibatch

            sel_sum = jnp.maximum(jnp.sum(sel_B), jnp.asarray(1.0, dtype=dtype))
            if holdout:
                valid_B = valid_B * sel_B

            raw_f_x = self.pidr_f.apply(f_params, x_BD)
            raw_f_next = self.pidr_f.apply(f_params, x_next_BD)
            f_scale = jnp.sqrt(0.5 * (jnp.mean(jnp.square(raw_f_x))
                                      + jnp.mean(jnp.square(raw_f_next)))
                               + jnp.asarray(1e-8, dtype=dtype))
            f_x = raw_f_x / f_scale
            f_next = raw_f_next / f_scale
            ratio = positive_ratio(ratio_params, x_BD, phase_B)
            delta = f_x - f_next

            mw = valid_B.reshape((-1,) + (1,) * (delta.ndim - 1))
            denom = jnp.maximum(jnp.sum(mw * jnp.ones_like(delta)), 1.0)
            flow_f = -jnp.sum(mw * jax.lax.stop_gradient(ratio) * delta) / denom
            flow_r = jnp.sum(mw * ratio * jax.lax.stop_gradient(delta)) / denom
            f_loss = flow_f

            is_start = (phase_B == 0).astype(dtype)
            if holdout:
                is_start = is_start * sel_B
            anchor_loss = jnp.sum(is_start * jnp.square(ratio - 1.0)) / jnp.maximum(jnp.sum(is_start), jnp.asarray(1.0, dtype=dtype))

            r_loss = (flow_r
                      + r2_regularizer * (jnp.sum(sel_B * jnp.square(ratio - 1.0)) / sel_sum
                                          if holdout else jnp.mean(jnp.square(ratio - 1.0)))
                      + anchor_coef * anchor_loss)
            return (f_loss, r_loss)

        full_batch = (x_ND, x_next_ND, start_phase_N, valid_N, train_sel_N)
        num_steps = int(self._tightening_pidr_steps)

        def train_step(carry, minibatch):
            (f_params, ratio_params, f_opt, ratio_opt) = carry
            f_loss_fn = lambda p: loss_parts(p, ratio_params, minibatch)[0]
            r_loss_fn = lambda p: loss_parts(f_params, p, minibatch)[1]
            (f_loss, f_grads) = jax.value_and_grad(f_loss_fn)(f_params)
            (r_loss, r_grads) = jax.value_and_grad(r_loss_fn)(ratio_params)
            (f_updates, f_opt) = self.pidr_f_opt.update(f_grads, f_opt, f_params)
            (r_updates, ratio_opt) = self.pidr_ratio_opt.update(r_grads, ratio_opt, ratio_params)
            f_params = optax.apply_updates(f_params, f_updates)
            ratio_params = optax.apply_updates(ratio_params, r_updates)
            return ((f_params, ratio_params, f_opt, ratio_opt), (f_loss, r_loss))

        def one_step(carry, _):

            return train_step(carry, full_batch)

        (rng, _pidr_rng) = jax.random.split(policy_state.rng)

        init_carry = (
            policy_state.pidr_f_params,
            policy_state.pidr_ratio_params,
            policy_state.pidr_f_opt,
            policy_state.pidr_ratio_opt,
        )
        ((f_params, ratio_params, f_opt, ratio_opt), losses) = jax.lax.scan(
            one_step, init_carry, jnp.arange(num_steps, dtype=jnp.int32)
        )
        raw_weights = positive_ratio(ratio_params, x_ND, start_phase_N)

        is_start0 = (start_phase_N == 0).astype(dtype)
        phase0_ratio_sq = jnp.sum(is_start0 * jnp.square(raw_weights - 1.0)) / jnp.maximum(jnp.sum(is_start0), jnp.asarray(1.0, dtype=dtype))

        def phase0_var_norm_fn(wn, sel=None):
            m = is_start0 if sel is None else is_start0 * sel
            return jnp.sum(m * jnp.square(wn - 1.0)) / jnp.maximum(jnp.sum(m), jnp.asarray(1.0, dtype=dtype))
        normalized = normalize_pidr_weights(raw_weights, ratio_clip=self._tightening_ratio_clip, phase_N=start_phase_N, phase_count=phase_count, f_loss=losses[0][-1], r_loss=losses[1][-1])
        if self._pidr_force_w1:

            ones = jnp.ones_like(normalized.weights)
            zero = jnp.zeros_like(normalized.tv)
            normalized = normalized._replace(
                weights=ones, ratio_mean=jnp.ones_like(normalized.ratio_mean),
                ratio_std=jnp.zeros_like(normalized.ratio_std),
                ratio_min=jnp.ones_like(normalized.ratio_min),
                ratio_max=jnp.ones_like(normalized.ratio_max),
                ratio_ess=jnp.ones_like(normalized.ratio_ess),
                tv=zero, chi2=jnp.zeros_like(normalized.chi2),
                tv_by_phase=jnp.zeros_like(normalized.tv_by_phase))
        policy_state = policy_state.replace(rng=rng, pidr_f_params=f_params, pidr_ratio_params=ratio_params, pidr_f_opt=f_opt, pidr_ratio_opt=ratio_opt)

        g_x = jnp.tanh(x_ND @ self._flow_probe_W)
        g_next = jnp.tanh(x_next_ND @ self._flow_probe_W)
        dg = g_x - g_next
        w_col = normalized.weights.reshape((-1, 1))
        mv = valid_N.reshape((-1, 1))
        denom = jnp.maximum(jnp.sum(valid_N), jnp.asarray(1.0, dtype=dtype))
        resid_one = jnp.abs(jnp.sum(mv * dg, axis=0) / denom)
        resid_w = jnp.abs(jnp.sum(mv * w_col * dg, axis=0) / denom)
        r1 = jnp.mean(resid_one)
        rw = jnp.mean(resid_w)

        idx = jnp.arange(phase_count, dtype=jnp.int32)
        K_shift = int(rollout.obs.shape[0])
        cnt_all = jnp.bincount(start_phase_N, length=phase_count).astype(dtype)
        denom_ph = jnp.maximum(cnt_all, jnp.asarray(1.0, dtype=dtype))

        def _per_phase(vals_N):
            return jnp.bincount(start_phase_N, weights=vals_N, length=phase_count).astype(dtype) / denom_ph

        has_src = cnt_all > 0
        has_tgt = jnp.roll(cnt_all, -K_shift) > 0
        in_range = idx < (phase_count - K_shift)
        ok_ph = has_src & has_tgt & in_range
        n_ok = jnp.maximum(jnp.sum(ok_ph).astype(dtype), jnp.asarray(1.0, dtype=dtype))

        def _matched(wcol):
            acc1 = jnp.zeros((), dtype=dtype)
            for k in range(self._flow_probe_dim):
                fwd = _per_phase(wcol[:, 0] * g_next[:, k])
                here = _per_phase(wcol[:, 0] * g_x[:, k])
                acc1 = acc1 + jnp.sum(jnp.where(ok_ph, jnp.abs(fwd - jnp.roll(here, -K_shift)), 0.0)) / n_ok
            return acc1 / float(self._flow_probe_dim)

        ones_col = jnp.ones_like(w_col)
        mp1 = _matched(ones_col)

        vc_x = jax.lax.stop_gradient(cost_value_N).reshape((-1, 1))
        vc_next = jax.lax.stop_gradient(cost_value_next_N).reshape((-1, 1))

        def _matched_h(wcol, h_here, h_fwd):
            fwd = _per_phase(wcol[:, 0] * h_fwd[:, 0])
            here = _per_phase(wcol[:, 0] * h_here[:, 0])
            return jnp.sum(jnp.where(ok_ph, jnp.abs(fwd - jnp.roll(here, -K_shift)), 0.0)) / n_ok

        mp_cost_w1 = _matched_h(ones_col, vc_x, vc_next)

        coin = jax.random.bernoulli(_pidr_rng, 0.5, (x_ND.shape[0],)).astype(dtype)
        cnt_a = jnp.bincount(start_phase_N, weights=coin, length=phase_count).astype(dtype)
        cnt_b = jnp.bincount(start_phase_N, weights=1.0 - coin, length=phase_count).astype(dtype)
        ok_ab = (cnt_a > 0) & (cnt_b > 0)
        n_ab = jnp.maximum(jnp.sum(ok_ab).astype(dtype), jnp.asarray(1.0, dtype=dtype))
        acc_null = jnp.zeros((), dtype=dtype)
        for k in range(self._flow_probe_dim):
            m_a = jnp.bincount(start_phase_N, weights=coin * g_x[:, k], length=phase_count).astype(dtype) / jnp.maximum(cnt_a, 1.0)
            m_b = jnp.bincount(start_phase_N, weights=(1.0 - coin) * g_x[:, k], length=phase_count).astype(dtype) / jnp.maximum(cnt_b, 1.0)
            acc_null = acc_null + jnp.sum(jnp.where(ok_ab, jnp.abs(m_a - m_b), 0.0)) / n_ab

        mp_null = acc_null / float(self._flow_probe_dim) / jnp.asarray(1.4142135623730951, dtype=dtype)

        if self._pidr_shrink:
            lam = jnp.clip(1.0 - jnp.square(mp_null) / jnp.maximum(jnp.square(mp1), jnp.asarray(1e-12, dtype=dtype)), 0.0, 1.0)
        else:
            lam = jnp.asarray(1.0, dtype=dtype)
        applied_w = 1.0 + lam * (normalized.weights - 1.0)
        w_col = applied_w.reshape((-1, 1))
        mpw = _matched(w_col)

        phf_w = start_phase_N.astype(dtype) * 2.0
        shuf_idx = jnp.argsort(phf_w + jax.random.uniform(
            jax.random.fold_in(_pidr_rng, 11), (applied_w.shape[0],), dtype=dtype))
        order = jnp.argsort(phf_w + jnp.arange(applied_w.shape[0], dtype=dtype)
                            / float(applied_w.shape[0]))
        w_shuf = jnp.zeros_like(applied_w).at[order].set(applied_w[shuf_idx])
        mp_wshuf = _matched(w_shuf.reshape((-1, 1)))
        mp_cost_w = _matched_h(w_col, vc_x, vc_next)
        mp_cost_wshuf = _matched_h(w_shuf.reshape((-1, 1)), vc_x, vc_next)

        acc_cost_null = jnp.zeros((), dtype=dtype)
        m_a_c = jnp.bincount(start_phase_N, weights=coin * vc_x[:, 0], length=phase_count).astype(dtype) / jnp.maximum(cnt_a, 1.0)
        m_b_c = jnp.bincount(start_phase_N, weights=(1.0 - coin) * vc_x[:, 0], length=phase_count).astype(dtype) / jnp.maximum(cnt_b, 1.0)
        acc_cost_null = jnp.sum(jnp.where(ok_ab, jnp.abs(m_a_c - m_b_c), 0.0)) / n_ab
        mp_cost_null = acc_cost_null / jnp.asarray(1.4142135623730951, dtype=dtype)
        tv_applied = 0.5 * jnp.mean(jnp.abs(applied_w - 1.0))
        normalized = normalized._replace(weights=applied_w)

        cnt_ph = jnp.bincount(start_phase_N, length=phase_count).astype(dtype)
        sq_ph = jnp.bincount(start_phase_N, weights=jnp.square(normalized.weights - 1.0), length=phase_count).astype(dtype)
        var_ph = sq_ph / jnp.maximum(cnt_ph, jnp.asarray(1.0, dtype=dtype))
        seen_ph = cnt_ph > 0
        edges = [0, phase_count // 4, phase_count // 2, (3 * phase_count) // 4, phase_count]

        def _quartile_spread(lo, hi):
            sel = seen_ph & (idx >= lo) & (idx < hi)
            nsel = jnp.maximum(jnp.sum(sel).astype(dtype), jnp.asarray(1.0, dtype=dtype))
            return jnp.sqrt(jnp.sum(jnp.where(sel, var_ph, 0.0)) / nsel)

        (q1, q2, q3, q4) = [_quartile_spread(edges[i], edges[i + 1]) for i in range(4)]
        probe = {
            'flow_resid_w1': r1,
            'flow_resid_w': rw,
            'flow_resid_ratio': rw / jnp.maximum(r1, jnp.asarray(1e-8, dtype=dtype)),

            'flow_mp_w1': mp1,
            'flow_mp_w': mpw,
            'flow_mp_ratio': mpw / jnp.maximum(mp1, jnp.asarray(1e-8, dtype=dtype)),

            'pidr_shrink_lambda': lam,
            'pidr_w_tv_applied': tv_applied,
            'phase0_ratio_sq': phase0_ratio_sq,
            'phase0_var_norm': phase0_var_norm_fn(normalized.weights),

            'phase0_var_norm_holdout': phase0_var_norm_fn(
                normalized.weights, 1.0 - train_sel_N),
            'phase0_count': jnp.sum(is_start0),

            'phase_occ_min': jnp.min(jnp.where(cnt_all > 0, cnt_all, jnp.max(cnt_all))),
            'phase_occ_mean': jnp.sum(cnt_all) / jnp.maximum(jnp.sum(cnt_all > 0).astype(dtype), jnp.asarray(1.0, dtype=dtype)),
            'phase_occ_std': jnp.std(jnp.where(cnt_all > 0, cnt_all, jnp.mean(cnt_all))),
            'phases_occupied': jnp.sum(cnt_all > 0),
            'pidr_train_sel': train_sel_N,
            'flow_mp_null': mp_null,
            'flow_mp_wshuf': mp_wshuf,
            'flow_mp_cost_w1': mp_cost_w1,
            'flow_mp_cost_w': mp_cost_w,
            'flow_mp_cost_wshuf': mp_cost_wshuf,
            'flow_mp_cost_null': mp_cost_null,
            'flow_mp_cost_w1_over_null': mp_cost_w1 / jnp.maximum(mp_cost_null, jnp.asarray(1e-8, dtype=dtype)),
            'flow_mp_cost_w_over_shuf': mp_cost_w / jnp.maximum(mp_cost_wshuf, jnp.asarray(1e-8, dtype=dtype)),
            'flow_mp_w_over_shuf': mpw / jnp.maximum(mp_wshuf, jnp.asarray(1e-8, dtype=dtype)),
            'flow_mp_w1_over_null': mp1 / jnp.maximum(mp_null, jnp.asarray(1e-8, dtype=dtype)),

            'pidr_w_spread_q1': q1,
            'pidr_w_spread_q2': q2,
            'pidr_w_spread_q3': q3,
            'pidr_w_spread_q4': q4,
            'pidr_w_spread_late_over_early': q4 / jnp.maximum(q1, jnp.asarray(1e-8, dtype=dtype)),
        }
        return (policy_state, normalized, probe)

    def _flatten_for_update(self, rollout: OnBatchState) -> OnBatchState:
        cfg = self.policy.cfg
        nc = cfg.num_constraints
        kwargs = dict(obs=rollout.obs.reshape((-1, cfg.obs_size)), actions=rollout.actions.reshape((-1, cfg.action_size)), pre_tanh=rollout.pre_tanh.reshape((-1, cfg.action_size)), logp=rollout.logp.reshape((-1,)), policy_mean=rollout.policy_mean.reshape((-1, cfg.action_size)), policy_log_std=rollout.policy_log_std.reshape((-1, cfg.action_size)))
        if rollout.returns is not None:
            kwargs['returns'] = rollout.returns.reshape((-1, 1 + nc))
        if rollout.advantages is not None:
            kwargs['advantages'] = rollout.advantages.reshape((-1, 1 + nc))
        return rollout.replace(**kwargs)

    def _apply_terms(self, pidr_upper: Any) -> jnp.ndarray:

        if self._tighten_terms == 'beta':
            return pidr_upper.surrogate + pidr_upper.base_beta
        if self._tighten_terms == 'epsilon':
            return pidr_upper.surrogate + pidr_upper.epsilon
        return pidr_upper.upper

    def _split_half_eps(self, weights_N, samples_NC, start_phase_N, train_sel_N, phase_count):

        dtype = get_dtype()
        Y_N = jnp.sum(samples_NC, axis=-1) if samples_NC.ndim > 1 else samples_NC

        def side(sel_N):
            cnt = jnp.bincount(start_phase_N, weights=sel_N, length=phase_count).astype(dtype)
            wsum = jnp.bincount(start_phase_N, weights=weights_N * sel_N, length=phase_count).astype(dtype)
            phase_mean = wsum / jnp.maximum(cnt, jnp.asarray(1.0, dtype=dtype))
            w_ren = weights_N / jnp.maximum(phase_mean[start_phase_N], jnp.asarray(1e-8, dtype=dtype))
            z_N = (w_ren - 1.0) * Y_N
            n = jnp.maximum(jnp.sum(sel_N), jnp.asarray(1.0, dtype=dtype))
            mean = jnp.sum(sel_N * z_N) / n
            var = jnp.sum(sel_N * jnp.square(z_N - mean)) / jnp.maximum(n - 1.0, jnp.asarray(1.0, dtype=dtype))
            radius = jnp.asarray(1.959963984540054, dtype=dtype) * jnp.sqrt(jnp.maximum(var, 0.0) / n)
            return mean, radius, jnp.maximum(mean + radius, 0.0)

        m_fit, r_fit, e_fit = side(train_sel_N)
        m_out, r_out, e_out = side(1.0 - train_sel_N)

        eval_sel_N = 1.0 - train_sel_N
        cnt_t = jnp.bincount(start_phase_N, weights=train_sel_N, length=phase_count).astype(dtype)
        wsum_t = jnp.bincount(start_phase_N, weights=weights_N * train_sel_N, length=phase_count).astype(dtype)
        pm_t = wsum_t / jnp.maximum(cnt_t, jnp.asarray(1.0, dtype=dtype))
        w_tn = weights_N / jnp.maximum(pm_t[start_phase_N], jnp.asarray(1e-8, dtype=dtype))
        z_tn = (w_tn - 1.0) * Y_N
        n_e = jnp.maximum(jnp.sum(eval_sel_N), jnp.asarray(1.0, dtype=dtype))
        m_tn = jnp.sum(eval_sel_N * z_tn) / n_e

        return {'eps_mean_fit': m_fit, 'eps_radius_fit': r_fit, 'eps_fit': e_fit,
                'eps_mean_holdout': m_out, 'eps_radius_holdout': r_out, 'eps_holdout': e_out,
                'eps_mean_holdout_trainnorm': m_tn}

    def _eps_ceiling(self, weights_N, samples_NC, start_phase_N, phase_count, shuffle_rng=None):

        dtype = get_dtype()
        y = jnp.asarray(samples_NC, dtype=dtype)[:, 0]
        w = jnp.asarray(weights_N, dtype=dtype).reshape(-1)
        ph = jnp.clip(jnp.asarray(start_phase_N, dtype=jnp.int32).reshape(-1), 0, phase_count - 1)
        cnt = jnp.bincount(ph, length=phase_count).astype(dtype)
        den = jnp.maximum(cnt, jnp.asarray(1.0, dtype=dtype))

        def _std(v):
            m = jnp.bincount(ph, weights=v, length=phase_count).astype(dtype) / den
            m2 = jnp.bincount(ph, weights=jnp.square(v), length=phase_count).astype(dtype) / den
            return jnp.sqrt(jnp.maximum(m2 - jnp.square(m), 0.0)), m

        sw, _ = _std(w)
        sy, my = _std(y)
        cov = jnp.bincount(ph, weights=w * y, length=phase_count).astype(dtype) / den - my
        seen = cnt > 0
        q = cnt / jnp.maximum(jnp.sum(cnt), jnp.asarray(1.0, dtype=dtype))
        ceiling = jnp.sum(jnp.where(seen, q * sw * sy, 0.0))
        achieved = jnp.sum(jnp.where(seen, q * cov, 0.0))

        n = w.shape[0]
        if shuffle_rng is None:
            fill_null = jnp.zeros((), dtype=dtype)
        else:
            phf = ph.astype(dtype) * 2.0
            det = jnp.argsort(phf + jnp.arange(n, dtype=dtype) / float(n))
            rnd = jnp.argsort(phf + jax.random.uniform(shuffle_rng, (n,), dtype=dtype))
            w_shuf = jnp.zeros_like(w).at[det].set(w[rnd])
            cov_s = jnp.bincount(ph, weights=w_shuf * y, length=phase_count).astype(dtype) / den - my
            m_s = jnp.bincount(ph, weights=w_shuf, length=phase_count).astype(dtype) / den
            m2_s = jnp.bincount(ph, weights=jnp.square(w_shuf), length=phase_count).astype(dtype) / den
            sw_s = jnp.sqrt(jnp.maximum(m2_s - jnp.square(m_s), 0.0))
            ceil_s = jnp.sum(jnp.where(seen, q * sw_s * sy, 0.0))
            ach_s = jnp.sum(jnp.where(seen, q * cov_s, 0.0))
            fill_null = jnp.abs(ach_s) / jnp.maximum(ceil_s, jnp.asarray(1e-12, dtype=dtype))
        nseen = jnp.maximum(jnp.sum(seen).astype(dtype), jnp.asarray(1.0, dtype=dtype))
        return {
            'eps_ceiling': ceiling,
            'eps_achieved': achieved,
            'eps_fill': jnp.abs(achieved) / jnp.maximum(ceiling, jnp.asarray(1e-12, dtype=dtype)),
            'eps_fill_null': fill_null,
            'y_std_within': jnp.sum(jnp.where(seen, q * sy, 0.0)),
            'w_std_within': jnp.sum(jnp.where(seen, q * sw, 0.0)),
            'y_mean_within': jnp.sum(jnp.where(seen, q * my, 0.0)),
            'phases_seen': nseen,
        }

    def one_update(self, runtime_state: RuntimeState, update_index: jax.Array, *, logging_hook: Any | None=None) -> tuple[RuntimeState, Metrics]:
        policy = self.policy
        runtime_state = self._rollout_env(runtime_state)
        collected_rollout = self.rollout_buf.sample(runtime_state.rollout_buf_state)
        episode_metrics = self._compute_episode_metrics(batch=collected_rollout, env_state=runtime_state.env_state)
        rollout_TN = self.prepare_batch_for_critic(runtime_state, collected_rollout=collected_rollout, cost_signal=episode_metrics['cost_signal'], cost_limit=episode_metrics['cost_limit'])
        batch_for_critic = self._flatten_for_update(rollout_TN)
        (policy_state, critic_metrics) = policy.update_critics(runtime_state.policy_state, batch_for_critic)
        runtime_state = runtime_state.replace(policy_state=policy_state)
        rollout_TN = self.prepare_actor_advantages(runtime_state, rollout_TN)
        if self._cost_surrogate:
            cost_values_TNc = rollout_TN.values[..., 1:]
            rollout_TN = rollout_TN.replace(cost_signal=jnp.mean(cost_values_TNc, axis=(0, 1)))
        tighten_metrics = {}
        if self._tightening_mode == 'on':
            (policy_state, pidr_weights, pidr_probe) = self._train_pidr_weights(policy_state, rollout_TN)
            runtime_state = runtime_state.replace(policy_state=policy_state)
            EL = int(policy.cfg.episode_length)
            delta = self._tightening_delta_event
            start_phase_N = jnp.clip(rollout_TN.steps[0].astype(jnp.int32), 0, EL - 1)
            baseline_NC = jnp.asarray(policy_state.pidr_cost_baseline, dtype=get_dtype())[start_phase_N]
            if self._cost_surrogate:
                per_env_cost_return = jnp.mean(cost_values_TNc, axis=0)
                cv_base_beta = empirical_bernstein_radius(per_env_cost_return, bound=self._tightening_bc, delta=delta)
                pidr_upper = compute_pidr_weighted_upper(per_env_cost_return, base_surrogate_C=rollout_TN.cost_signal, base_beta_C=cv_base_beta, weights_N=pidr_weights.weights, delta=delta, baseline_NC=baseline_NC)
                policy_state = policy_state.replace(pidr_cost_baseline=self._phase_means(per_env_cost_return, start_phase_N, EL))
                runtime_state = runtime_state.replace(policy_state=policy_state)
                tightened = jnp.minimum(self._apply_terms(pidr_upper), self._tightening_max)
                rollout_TN = rollout_TN.replace(cost_signal=tightened)
                eps_diag = self._eps_ceiling(pidr_weights.weights, per_env_cost_return,
                                             jnp.clip(rollout_TN.steps[0].astype(jnp.int32), 0, EL - 1), EL,
                                             shuffle_rng=jax.random.fold_in(policy_state.rng, 7))
                eps_diag = {**eps_diag, **self._split_half_eps(
                    pidr_weights.weights, per_env_cost_return,
                    jnp.clip(rollout_TN.steps[0].astype(jnp.int32), 0, EL - 1),
                    pidr_probe['pidr_train_sel'], EL)}
            else:
                gamma = self._tightening_gamma
                base_result = compute_raw_phase_mixture_tightening(rollout_TN.costs, rollout_TN.steps, gamma=gamma, episode_length=EL, delta_beta=delta, c_max=self._tightening_cmax)
                raw_steps = jnp.clip(rollout_TN.steps.astype(jnp.int32), 0, EL - 1)
                flat_steps = raw_steps.reshape(-1)
                counts = jnp.bincount(flat_steps, length=EL).astype(get_dtype())
                sample_counts = counts[flat_steps].reshape(raw_steps.shape)
                gamma_arr = jnp.asarray(gamma, dtype=get_dtype())
                coverage_weights = jnp.asarray(policy.cfg.num_envs, dtype=get_dtype()) * jnp.power(gamma_arr, raw_steps.astype(get_dtype())) / jnp.maximum(sample_counts, jnp.asarray(1.0, dtype=get_dtype()))
                per_env_raw = jnp.sum(coverage_weights[..., None] * rollout_TN.costs, axis=0)
                pidr_upper = compute_pidr_weighted_upper(per_env_raw, base_surrogate_C=base_result.surrogate, base_beta_C=base_result.beta, weights_N=pidr_weights.weights, delta=delta, baseline_NC=baseline_NC)
                policy_state = policy_state.replace(pidr_cost_baseline=self._phase_means(per_env_raw, start_phase_N, EL))
                runtime_state = runtime_state.replace(policy_state=policy_state)
                tightened_with_epsilon = jnp.where(base_result.coverage_complete, self._apply_terms(pidr_upper), base_result.tightened)
                tightened = jnp.minimum(tightened_with_epsilon, self._tightening_max)
                rollout_TN = rollout_TN.replace(cost_signal=tightened)
                eps_diag = self._eps_ceiling(pidr_weights.weights, per_env_raw,
                                             jnp.clip(rollout_TN.steps[0].astype(jnp.int32), 0, EL - 1), EL,
                                             shuffle_rng=jax.random.fold_in(policy_state.rng, 7))
                eps_diag = {**eps_diag, **self._split_half_eps(
                    pidr_weights.weights, per_env_raw,
                    jnp.clip(rollout_TN.steps[0].astype(jnp.int32), 0, EL - 1),
                    pidr_probe['pidr_train_sel'], EL)}
            tighten_metrics = {

                'tightening_beta': pidr_upper.base_beta,
                'tightening_epsilon': pidr_upper.epsilon,

                'tightening_eps_mean': pidr_upper.correction_mean,
                'tightening_eps_radius': pidr_upper.correction_uncertainty,

                'pidr_w_mean': pidr_weights.ratio_mean,
                'pidr_w_std': pidr_weights.ratio_std,
                'pidr_w_min': pidr_weights.ratio_min,
                'pidr_w_max': pidr_weights.ratio_max,
                'pidr_w_tv': pidr_weights.tv,
                'pidr_w_chi2': pidr_weights.chi2,
                'pidr_w_ess': pidr_weights.ratio_ess,
                'pidr_norm_error': pidr_weights.norm_error,
                'pidr_f_loss': pidr_weights.f_loss,
                'pidr_r_loss': pidr_weights.r_loss,

                **{k: v for k, v in pidr_probe.items() if k != 'pidr_train_sel'},
                **eps_diag,
            }
        batch_for_actor = self._flatten_for_update(rollout_TN)
        actor_before = (policy_state.actor_params, runtime_state.policy_state.obs_rms)
        (policy_state, actor_metrics) = policy.update_actor_only(policy_state, batch_for_actor)
        if _POLICY_FREEZE_AFTER >= 0:

            frozen = update_index >= jnp.asarray(_POLICY_FREEZE_AFTER, dtype=update_index.dtype)

            _flat = lambda tree: jnp.concatenate(
                [jnp.ravel(x) for x in jax.tree.leaves(tree)])
            keep = lambda new, old: jax.tree.map(
                lambda a, b: jnp.where(frozen, b, a), new, old)

            policy_state = policy_state.replace(
                actor_params=keep(policy_state.actor_params, actor_before[0]),
                obs_rms=keep(policy_state.obs_rms, actor_before[1]))
            freeze_diag = {
                'actor_param_delta': jnp.linalg.norm(
                    _flat(policy_state.actor_params) - _flat(actor_before[0])),
                'obs_rms_count': jnp.asarray(policy_state.obs_rms.count, dtype=get_dtype()),
                'actor_frozen': frozen.astype(get_dtype()),
            }
        else:
            freeze_diag = {}
        (policy_state, lag_metrics) = policy.update_lagrange(policy_state, batch_for_actor)
        if self._tightening_mode == 'on':

            (kl_mean_new, kl_log_std_new) = policy.actor.apply(policy_state.actor_params, batch_for_actor.obs)
            tighten_metrics['policy_kl_realized'] = jnp.mean(
                gaussian_kl(batch_for_actor.policy_mean, batch_for_actor.policy_log_std, kl_mean_new, kl_log_std_new))

        if _POLICY_FREEZE_AFTER >= 0:
            _rms_before = policy_state.obs_rms
            policy_state = policy._update_obs_stats(collected_rollout.obs, policy_state)
            _frozen = update_index >= jnp.asarray(_POLICY_FREEZE_AFTER, dtype=update_index.dtype)
            policy_state = policy_state.replace(obs_rms=jax.tree.map(
                lambda new, old: jnp.where(_frozen, old, new), policy_state.obs_rms, _rms_before))
        else:
            policy_state = policy._update_obs_stats(collected_rollout.obs, policy_state)
        update_metrics = {**critic_metrics, **actor_metrics, **lag_metrics, **tighten_metrics, **freeze_diag}
        merged = {**episode_metrics, **{key: value for key, value in update_metrics.items() if key != 'state'}}
        if logging_hook is not None:
            logging_hook.emit(merged, debug=update_index, seed_idx=runtime_state.seed_idx)
        return (runtime_state.replace(policy_state=policy_state), merged)
