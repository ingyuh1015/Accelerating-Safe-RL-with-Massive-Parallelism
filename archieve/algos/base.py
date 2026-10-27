from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, ClassVar
import jax
import optax
import jax.numpy as jnp
from archieve import get_dtype, OnPolicyState, RunningMeanStd
from archieve.algos import GaussianLearningActor, ValueCritic
from archieve.utils import build_optimizer
import os
from functools import partial

_POLICY_FREEZE = os.environ.get('POLICY_FREEZE', '0') not in ('', '0', 'false')

class OnPolicyBase(ABC):
    policy_type: ClassVar[str] = 'on_policy'

    def __init__(self, *, config: Any, compute_dtype: Any | None=None) -> None:
        self.cfg = config
        hs = tuple(self.cfg.hidden_sizes)
        dtype = compute_dtype or get_dtype()
        self.actor = GaussianLearningActor(action_dim=self.cfg.action_size, hidden_sizes=hs, activation=self.cfg.activation, weight_init=self.cfg.weight_init, layer_norm=self.cfg.layer_norm, compute_dtype=dtype)
        _critic_hs = getattr(self.cfg, 'critic_hidden_sizes', None)
        critic_hs = tuple(_critic_hs) if _critic_hs else hs
        _critic_act = getattr(self.cfg, 'critic_activation', None) or self.cfg.activation
        _critic_wi = getattr(self.cfg, 'critic_weight_init', None) or self.cfg.weight_init
        _critic_ln = getattr(self.cfg, 'critic_layer_norm', None)
        _critic_ln = self.cfg.layer_norm if _critic_ln is None else bool(_critic_ln)
        self.critic = ValueCritic(hidden_sizes=critic_hs, activation=_critic_act, weight_init=_critic_wi, layer_norm=_critic_ln, compute_dtype=dtype)
        self.critic_opt = build_optimizer(lr=self.cfg.critic_lr, max_grad_norm=self.cfg.max_grad_norm)
        self.actor_loss = partial(self.actor.loss, entropy_coef=self.cfg.entropy_coef)
        self.reward_loss_fn = lambda p, b: self.actor_loss(p, b, adv=b.advantages[..., 0])
        if self.cfg.obs_normalize and not _POLICY_FREEZE:
            obs_size = self.cfg.obs_size
            self._update_obs_stats = lambda obs, state: state.replace(obs_rms=state.obs_rms.update(obs.reshape((-1, obs_size))))
        elif self.cfg.obs_normalize:

            self._update_obs_stats = lambda obs, state: state
        else:
            self._update_obs_stats = lambda obs, state: state

    @abstractmethod
    def update_actor(self, state: Any, batch: Any) -> tuple[Any, dict[str, Any]]:
        ...

    def update_lagrange(self, state: Any, batch: Any) -> tuple[Any, dict[str, Any]]:
        del batch
        return (state, {})

    def init_state(self, rng: jax.Array) -> OnPolicyState:
        (k_a, k_c) = jax.random.split(rng, 2)
        obs = jnp.zeros((self.cfg.obs_size,))
        rms = RunningMeanStd.create(self.cfg.obs_size)
        a_p = self.actor.init(k_a, obs, rms)
        critic_keys = jax.random.split(k_c, 1 + self.cfg.num_constraints)
        critic_params = jax.vmap(lambda k: self.critic.init(k, obs, rms))(critic_keys)
        critic_opt = jax.vmap(self.critic_opt.init)(critic_params)
        return OnPolicyState(actor_params=a_p, critic_params=critic_params, critic_opt=critic_opt, rng=rng, obs_rms=rms)

    def update_actor_only(self, state, batch) -> tuple[Any, dict[str, Any]]:

        (new_state, metrics) = self.update_actor(state, batch)
        if _POLICY_FREEZE:
            new_state = new_state.replace(actor_params=state.actor_params)
        return (new_state, metrics)

    def critic_step(self, state, batch):
        targets = jnp.moveaxis(batch.returns, -1, 0)
        critic_loss = self.critic.loss
        opt_update = self.critic_opt.update
        use_norm = self.cfg.use_critic_norm
        norm_coef = self.cfg.critic_norm_coef

        def per_slot_update(params, opt, target):
            loss_fn = lambda p: critic_loss(p, batch, target=target, use_critic_norm=use_norm, critic_norm_coef=norm_coef)
            (loss, grads) = jax.value_and_grad(loss_fn)(params)
            (updates, new_opt) = opt_update(grads, opt, params)
            return (optax.apply_updates(params, updates), new_opt, loss)
        (new_p, new_o, losses) = jax.vmap(per_slot_update)(state.critic_params, state.critic_opt, targets)
        value_means = jax.vmap(lambda p: jnp.mean(self.critic.apply(p, batch.obs)))(new_p)
        return (state.replace(critic_params=new_p, critic_opt=new_o), {'reward_critic_loss': losses[0], 'cost_critic_loss': losses[1:], 'reward_value_mean': value_means[0], 'cost_value_mean': value_means[1:]})

    def update_critics(self, state, batch):
        minibatch_size = int(self.cfg.update_batch_size)
        batch_size = batch.obs.shape[0]
        if batch_size == 0 or minibatch_size <= 0 or self.cfg.update_epochs <= 0:
            raise ValueError('Critic updates require a nonempty batch, positive minibatch size and positive epochs.')
        minibatch_size = min(minibatch_size, batch_size)
        num_minibatches = batch_size // minibatch_size
        tail_size = batch_size % minibatch_size

        def _one_epoch(carry, epoch_rng):
            st = carry
            perm = jax.random.permutation(epoch_rng, batch_size)
            shuffled = jax.tree_util.tree_map(lambda x: x[perm] if x.ndim > 0 and x.shape[0] == batch_size else x, batch)

            def minibatch(start, size):
                return jax.tree_util.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, start, size) if x.ndim > 0 and x.shape[0] == batch_size else x, shuffled)

            def _one_minibatch(carry, mb_idx):
                st = carry
                mb = minibatch(mb_idx * minibatch_size, minibatch_size)
                (st, critic_metrics) = self.critic_step(st, mb)
                return (st, critic_metrics)
            (st, metrics) = jax.lax.scan(_one_minibatch, st, jnp.arange(num_minibatches))
            last_metrics = jax.tree_util.tree_map(lambda x: x[-1], metrics)
            if tail_size:
                tail = minibatch(num_minibatches * minibatch_size, tail_size)
                st, last_metrics = self.critic_step(st, tail)
            return (st, last_metrics)
        epoch_rngs = jax.random.split(state.rng, self.cfg.update_epochs)
        (state, all_metrics) = jax.lax.scan(_one_epoch, state, epoch_rngs)
        last_metrics = jax.tree_util.tree_map(lambda x: x[-1], all_metrics)
        return (state, last_metrics)
