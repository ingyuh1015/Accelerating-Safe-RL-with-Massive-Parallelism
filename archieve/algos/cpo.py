from __future__ import annotations
import jax
import jax.numpy as jnp
from jax.scipy import sparse
from archieve.algos import TRPO
from jax.flatten_util import ravel_pytree

class CPO(TRPO):

    def update_actor(self, state, batch):
        if self.cfg.num_constraints == 0:
            return super().update_actor(state, batch)
        actor_params = state.actor_params
        actor_loss_fn = self.actor_loss
        cfg = self.cfg
        nc = int(cfg.num_constraints)
        (kl_fn, fvp_fn, flat_params, unravel_fn) = self.prepare_trust_region(actor_params, batch)

        def reward_scalar_fn(params):
            return actor_loss_fn(params, batch, adv=batch.advantages[..., 0])[0]

        def cost_scalars_fn(params):
            return jax.vmap(lambda i: -self.actor.loss(params, batch, adv=batch.advantages[..., 1 + i], entropy_coef=0.0)[0])(jnp.arange(nc))
        (old_reward, reward_grad_tree) = jax.value_and_grad(reward_scalar_fn)(actor_params)
        reward_grad = -ravel_pytree(reward_grad_tree)[0]
        cost_grad_cols = jax.vmap(lambda i: ravel_pytree(jax.grad(lambda p: cost_scalars_fn(p)[i])(actor_params))[0])(jnp.arange(nc))
        cost_grads = cost_grad_cols.T
        violations = batch.cost_signal[:nc] - batch.cost_limit[:nc]
        if nc == 1:
            direction, optim_case = self._single_constraint_direction(reward_grad, cost_grads[:, 0], violations[0], fvp_fn)
        else:
            direction, optim_case = self._multiple_constraint_direction(reward_grad, cost_grads, violations, fvp_fn)
        old_costs = cost_scalars_fn(actor_params)

        def evaluate_policy(new_flat_params):
            new_params = unravel_fn(new_flat_params)
            return (reward_scalar_fn(new_params), cost_scalars_fn(new_params), kl_fn(new_flat_params))
        (new_flat, acceptance_step) = self._constrained_line_search(direction=direction, evaluate_fn=evaluate_policy, old_loss=old_reward, old_surr_cost=old_costs, optim_case=optim_case, c=violations, old_flat_params=flat_params, cost_tolerance=16.0 * jnp.finfo(direction.dtype).eps * (1.0 + jnp.linalg.norm(cost_grads, axis=0) * jnp.linalg.norm(direction)))
        new_params = unravel_fn(new_flat)
        return (state.replace(actor_params=new_params), {'Misc/AcceptanceStep': acceptance_step, 'Misc/OptimCase': optim_case.astype(jnp.float32), 'Misc/Violation': jnp.max(violations).astype(jnp.float32), 'reward_grad_norm': jnp.linalg.norm(reward_grad).astype(jnp.float32), 'cost_grad_norm': jnp.linalg.norm(cost_grads, axis=0).astype(jnp.float32)})

    def _single_constraint_direction(self, g_flat, b_flat, c, damped_fvp_fn):
        target_kl = self.cfg.target_kl
        cg_iters = self.cfg.cg_max_iters
        v = sparse.linalg.cg(damped_fvp_fn, g_flat, maxiter=cg_iters)[0]
        w = sparse.linalg.cg(damped_fvp_fn, b_flat, maxiter=cg_iters)[0]
        tiny = jnp.finfo(g_flat.dtype).tiny
        radius_sq = 2.0 * target_kl
        q = jnp.maximum(jnp.dot(v, damped_fvp_fn(v)), 0.0)
        s = jnp.maximum(jnp.dot(b_flat, w), 0.0)
        unconstrained = v * jnp.sqrt(radius_sq / jnp.maximum(q, tiny))
        feasible = jnp.dot(b_flat, unconstrained) + c <= 0.0

        boundary = -c * w / jnp.maximum(s, tiny)
        tangent = v - w * (jnp.dot(b_flat, v) / jnp.maximum(s, tiny))
        tangent_norm_sq = jnp.maximum(jnp.dot(tangent, damped_fvp_fn(tangent)), 0.0)
        remaining = jnp.maximum(radius_sq - c * c / jnp.maximum(s, tiny), 0.0)
        tangent = jnp.where(tangent_norm_sq > q * (32.0 * jnp.finfo(g_flat.dtype).eps) ** 2, tangent, 0.0)
        constrained = boundary + tangent * jnp.sqrt(remaining / jnp.maximum(tangent_norm_sq, tiny))
        recovery = -w * jnp.sqrt(radius_sq / jnp.maximum(s, tiny))
        infeasible = (c > 0.0) & (c * c > radius_sq * s)
        direction = jnp.where(feasible, unconstrained, jnp.where(infeasible, recovery, constrained))
        optim_case = jnp.where(feasible, jnp.where(s > 0.0, 3, 4), jnp.where(infeasible, 0, jnp.where(c >= 0.0, 1, 2)))
        return direction, optim_case.astype(jnp.int32)

    def _multiple_constraint_direction(self, g, cost_grads, violations, fvp):
        target_kl = self.cfg.target_kl
        cg_iters = self.cfg.cg_max_iters
        gradients = jnp.column_stack((g, cost_grads))
        gradients = gradients / jnp.maximum(jnp.linalg.norm(gradients, axis=0), 1e-20)
        basis = jax.vmap(lambda b: sparse.linalg.cg(fvp, b, maxiter=cg_iters)[0], in_axes=1, out_axes=1)(gradients)
        def orthogonalize(i, orthogonal):
            direction = basis[:, i]
            initial_norm = jnp.sqrt(jnp.maximum(direction @ fvp(direction), 1e-30))
            direction = direction / initial_norm
            direction = direction - orthogonal @ (orthogonal.T @ fvp(direction))
            direction = direction - orthogonal @ (orthogonal.T @ fvp(direction))
            norm_sq = jnp.maximum(direction @ fvp(direction), 0.0)
            direction = jnp.where(norm_sq > (32.0 * jnp.finfo(g.dtype).eps) ** 2, direction / jnp.sqrt(jnp.maximum(norm_sq, 1e-30)), 0.0)
            return orthogonal.at[:, i].set(direction)

        basis = jax.lax.fori_loop(0, gradients.shape[1], orthogonalize, jnp.zeros_like(basis))
        reward = basis.T @ g
        reward = reward / jnp.maximum(jnp.linalg.norm(reward), 1e-20)
        costs = basis.T @ cost_grads
        scale = jnp.linalg.norm(costs, axis=0)
        scale = jnp.where(scale > 1e-12, scale, 1.0)
        costs = costs / scale
        violations = violations / scale
        radius_sq = 2.0 * target_kl
        dimension = reward.size

        def minimize_barrier(point, linear, normals, bounds, ball_mask):
            def outer(stage, point):
                mu = jnp.asarray(0.1, g.dtype) * jnp.power(jnp.asarray(0.1, g.dtype), stage)

                def objective(x):
                    slack = normals.T @ x - bounds
                    ball = radius_sq - jnp.sum(ball_mask * x * x)
                    value = linear @ x - mu * (jnp.sum(jnp.log(jnp.maximum(slack, 1e-30))) + jnp.log(jnp.maximum(ball, 1e-30)))
                    return value, jnp.all(slack > 0.0) & (ball > 0.0)

                def newton(_, point):
                    slack = normals.T @ point - bounds
                    ball = radius_sq - jnp.sum(ball_mask * point * point)
                    masked = ball_mask * point
                    gradient = linear - mu * (normals @ (1.0 / slack)) + 2.0 * mu * masked / ball
                    scaled = normals / slack
                    hessian = mu * (scaled @ scaled.T) + 2.0 * mu * jnp.diag(ball_mask) / ball + 4.0 * mu * jnp.outer(masked, masked) / (ball * ball)
                    direction = -jnp.linalg.solve(hessian, gradient)
                    direction = jnp.where(jnp.all(jnp.isfinite(direction)), direction, 0.0)
                    value, _ = objective(point)
                    slope = gradient @ direction

                    def backtrack(carry):
                        iteration, step, _ = carry
                        step = step * 0.5
                        trial_value, interior = objective(point + step * direction)
                        accepted = interior & (trial_value <= value + 1e-4 * step * slope + 1e-7)
                        return iteration + 1, step, accepted

                    trial_value, interior = objective(point + direction)
                    initial = (jnp.int32(0), jnp.asarray(1.0, g.dtype), interior & (trial_value <= value + 1e-4 * slope + 1e-7))
                    _, step, accepted = jax.lax.while_loop(lambda carry: (carry[0] < 30) & ~carry[2], backtrack, initial)
                    return jnp.where(accepted, point + step * direction, point)

                return jax.lax.fori_loop(0, 30, newton, point)

            return jax.lax.fori_loop(0, 7, outer, point)

        phase_normals = jnp.vstack((-costs, jnp.ones_like(violations)))
        phase_start = jnp.concatenate((jnp.zeros_like(reward), jnp.maximum(jnp.max(violations), 0.0)[None] + 1.0))
        phase_linear = jnp.concatenate((jnp.zeros_like(reward), jnp.ones((1,), g.dtype)))
        phase_mask = jnp.concatenate((jnp.ones_like(reward), jnp.zeros((1,), g.dtype)))
        phase_point = minimize_barrier(phase_start, phase_linear, phase_normals, violations, phase_mask)[:-1]
        residual = costs.T @ phase_point + violations
        feasible = jnp.max(residual) <= 2e-5

        def optimize():
            relaxation = jnp.where(jnp.max(residual) < 0.0, jnp.zeros_like(residual), jnp.maximum(residual, 0.0) + 1e-7)
            point = minimize_barrier(phase_point, -reward, -costs, violations - relaxation, jnp.ones_like(reward))

            active = costs.T @ point + violations > -1e-4
            u, singular, vh = jnp.linalg.svd(costs * active, full_matrices=False)
            cutoff = jnp.maximum(jnp.max(singular), 1e-20) * 1e-7
            retained = singular > cutoff
            inverse = jnp.where(retained, 1.0 / jnp.maximum(singular, cutoff), 0.0)
            boundary = u @ ((vh @ (-violations * active)) * inverse)
            tangent = reward - u @ (retained * (u.T @ reward))
            tangent_sq = tangent @ tangent
            tangent = jnp.where(tangent_sq > (reward @ reward) * (32.0 * jnp.finfo(g.dtype).eps) ** 2, tangent, 0.0)
            candidate = boundary + tangent * jnp.sqrt(jnp.maximum(radius_sq - boundary @ boundary, 0.0) / jnp.maximum(tangent_sq, 1e-30))
            admissible = jnp.all(costs.T @ candidate + violations <= 2e-6) & (candidate @ candidate <= radius_sq + 2e-6)
            point = jnp.where(admissible & (reward @ candidate >= reward @ point), candidate, point)
            return point, jnp.where(jnp.any(violations >= 0.0), 1, 2).astype(jnp.int32)

        point, optim_case = jax.lax.cond(feasible, optimize, lambda: (phase_point, jnp.int32(0)))
        return basis @ point, optim_case

    def _constrained_line_search(self, direction, evaluate_fn, old_loss, old_surr_cost, optim_case, c, old_flat_params, *, cost_tolerance=0.0):
        max_iters = self.cfg.line_search_max_iter
        coeff = self.cfg.line_search_coeff
        target_kl = self.cfg.target_kl
        violations = jnp.atleast_1d(c)
        old_loss = jnp.squeeze(old_loss)
        old_surr_cost = jnp.atleast_1d(old_surr_cost)

        def body_fn(val):
            iteration, _, _ = val
            params = old_flat_params + coeff ** iteration * direction
            loss, costs, kl = evaluate_fn(params)
            improves = (optim_case <= 1) | (loss <= old_loss)
            tolerance = 1e-5 * jnp.maximum(1.0, jnp.maximum(jnp.abs(old_surr_cost), jnp.abs(jnp.atleast_1d(costs))))
            tolerance = jnp.maximum(tolerance, cost_tolerance)
            safe = jnp.all(jnp.atleast_1d(costs) - old_surr_cost <= jnp.maximum(-violations, 0.0) + tolerance)
            finite = jnp.isfinite(loss) & jnp.all(jnp.isfinite(costs)) & jnp.isfinite(kl)
            accepted = improves & safe & (kl <= target_kl + 1e-6) & finite
            return iteration + 1, params, accepted

        initial = (jnp.int32(0), old_flat_params, jnp.bool_(False))
        iterations, params, accepted = jax.lax.while_loop(lambda val: ~val[2] & (val[0] < max_iters), body_fn, initial)
        return jnp.where(accepted, params, old_flat_params), jnp.where(accepted, iterations, 0)
