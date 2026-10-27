from __future__ import annotations
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp


def jit_batch_step(step_fn: Callable, model) -> Callable:

    @partial(jax.jit, static_argnums=())
    def batched(data, action):
        return jax.vmap(step_fn, in_axes=(None, 0, 0))(model, data, action)
    return batched


def frame_skip_step(step_fn: Callable, model, data, action, n_substeps: int):

    def body(_, d):
        return step_fn(model, d, action)
    return jax.lax.fori_loop(0, n_substeps, body, data)


def tree_concat(trees, axis: int = 0):

    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=axis), *trees)


def tree_stack(trees, axis: int = 0):

    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=axis), *trees)


def tile_for_batch(tree, batch_size: int):

    return jax.tree.map(lambda x: jnp.broadcast_to(x[None], (batch_size,) + x.shape), tree)


def has_nan(tree) -> bool:

    leaves = jax.tree.leaves(tree)
    return any(bool(jnp.any(jnp.isnan(l))) for l in leaves)
