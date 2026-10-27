from __future__ import annotations
import jax.numpy as jnp
from massive_saferl import EPS, get_dtype, LagrangeOnPolicyState
import optax
from massive_saferl.algos import TRPO

class TRPOLagrange(TRPO):

    def __init__(self, *, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.lagrange_opt = optax.adam(self.cfg.lagrange_lr)

    def init_state(self, rng):
        state = super().init_state(rng)
        lagrange = jnp.zeros((self.cfg.num_constraints,), dtype=get_dtype()) + self.cfg.lagrange_multiplier_init
        state_dict = {k: v for (k, v) in state.__dict__.items()}
        state_dict.update(lagrange=lagrange, lagrange_opt=self.lagrange_opt.init(lagrange))
        return LagrangeOnPolicyState(**state_dict)

    def update_actor_only(self, state, batch):
        lagrange = jnp.maximum(state.lagrange, 0.0)
        reward_adv = batch.advantages[..., 0]
        cost_advs = batch.advantages[..., 1:]
        combined = (reward_adv - jnp.sum(lagrange * cost_advs, axis=-1)) / (1.0 + jnp.sum(lagrange))
        combined = (combined - jnp.mean(combined)) / (jnp.std(combined) + EPS)
        batch = batch.replace(advantages=batch.advantages.at[..., 0].set(combined))
        return super().update_actor_only(state, batch)

    def update_lagrange(self, state, batch):
        delta = batch.cost_signal - batch.cost_limit
        grad = -delta
        (updates, new_opt) = self.lagrange_opt.update(grad, state.lagrange_opt)
        new_lagrange = jnp.maximum(optax.apply_updates(state.lagrange, updates), 0.0)
        return (state.replace(lagrange=new_lagrange, lagrange_opt=new_opt), {'lagrange': new_lagrange, 'lagrange_delta': delta})
