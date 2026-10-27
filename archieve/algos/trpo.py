from __future__ import annotations
from typing import Any, Callable
import jax
import jax.numpy as jnp
from archieve.algos import OnPolicyBase, gaussian_kl
from archieve import EPS, get_dtype
from jax.flatten_util import ravel_pytree

class TRPO(OnPolicyBase):

    def update_actor(self, state, batch):
        (new_params, metrics) = self.ng_linesearch_update(actor_loss_fn=self.reward_loss_fn, actor_params=state.actor_params, batch=batch)
        return (state.replace(actor_params=new_params), metrics)

    def ng_linesearch_update(self, *, actor_loss_fn: Callable, actor_params: Any, batch: Any) -> tuple[Any, dict[str, Any]]:
        cfg = self.cfg
        (kl_fn, fvp_fn, flat_params, unravel_fn) = self.prepare_trust_region(actor_params, batch)

        def scalar_loss(params):
            return actor_loss_fn(params, batch)[0]
        tkl = jnp.asarray(cfg.target_kl, dtype=get_dtype())
        (old_loss, grad) = jax.value_and_grad(scalar_loss)(actor_params)
        (flat_grad, _) = ravel_pytree(grad)
        x = self._safe_nan_zero(jax.scipy.sparse.linalg.cg(fvp_fn, -flat_grad, maxiter=max(1, cfg.cg_max_iters))[0])
        xHx = jnp.dot(x, fvp_fn(x))
        alpha = jnp.sqrt(2.0 * tkl / (jnp.maximum(xHx, 0.0) + EPS))
        full_step = alpha * x
        (new_flat, acceptance_step) = self._line_search(direction=full_step, objective_fn=scalar_loss, kl_fn=kl_fn, old_loss=old_loss, old_flat=flat_params, unravel_fn=unravel_fn)
        new_params = unravel_fn(new_flat)
        return (new_params, {'actor_loss': scalar_loss(new_params), 'alpha': alpha, 'acceptance_step': acceptance_step})

    def prepare_trust_region(self, actor_params, batch):
        (flat_params, unravel_fn) = ravel_pytree(actor_params)
        (old_mean, old_log_std) = self.actor.apply(actor_params, batch.obs)
        old_mean = jax.lax.stop_gradient(old_mean)
        old_log_std = jax.lax.stop_gradient(old_log_std)

        def kl_fn(flat_actor_params):
            params = unravel_fn(flat_actor_params)
            (mean_new, log_std_new) = self.actor.apply(params, batch.obs)
            return self._safe_nan_zero(jnp.mean(gaussian_kl(old_mean, old_log_std, mean_new, log_std_new)))

        def fvp_fn(v):
            v = self._safe_nan_zero(v)
            h = self._safe_nan_zero(jax.jvp(jax.grad(kl_fn), (flat_params,), (v,))[1])
            return h + self.cfg.cg_damping * v
        return (kl_fn, fvp_fn, flat_params, unravel_fn)

    def _line_search(self, *, direction: jax.Array, objective_fn: Callable[[object], jax.Array], kl_fn: Callable[[jax.Array], jax.Array], old_loss: jax.Array, old_flat: jax.Array, unravel_fn: Callable[[jax.Array], object]) -> tuple[jax.Array, jax.Array]:
        max_iters = self.cfg.line_search_max_iter
        coeff = self.cfg.line_search_coeff
        target_kl = jnp.asarray(self.cfg.target_kl, dtype=get_dtype())

        def evaluate_fn(new_flat_params):
            new_loss = objective_fn(unravel_fn(new_flat_params))
            kl = kl_fn(new_flat_params)
            return (new_loss, kl)

        def acceptance_fn(eval_state):
            (new_loss, kl) = eval_state
            return (new_loss <= old_loss) & (kl <= target_kl)
        initial_eval = evaluate_fn(old_flat)

        def cond_fn(val):
            (iter_, _params, _eval, accepted) = val
            return ~accepted & (iter_ < max_iters)

        def body_fn(val):
            (iter_, _params, _eval, _accepted) = val
            step_size = coeff ** iter_
            new_params = old_flat + step_size * direction
            eval_state = evaluate_fn(new_params)
            accepted = acceptance_fn(eval_state)
            return (iter_ + 1, new_params, eval_state, accepted)
        init = (jnp.int8(0), old_flat, initial_eval, jnp.bool_(False))
        (iters, new_params, _eval, accepted) = jax.lax.while_loop(cond_fn, body_fn, init)
        new_params = jax.lax.cond(accepted, lambda : new_params, lambda : old_flat)
        acceptance_step = jnp.where(accepted, iters, jnp.int8(0))
        return (new_params, acceptance_step)

    @staticmethod
    def _safe_nan_zero(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
