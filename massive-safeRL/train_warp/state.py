import jax
import jax.numpy as jnp
from mujoco.mjx.warp.types import DATA_NON_VMAP, DataWarp


def conditional_worlds(predicate, true_fun, false_fun, operand):


    @jax.custom_batching.custom_vmap
    def apply(pred, args):
        return jax.lax.cond(jnp.any(pred), true_fun, false_fun, args)

    @apply.def_vmap
    def apply_vmap(axis_size, in_batched, pred, args):
        def axis_for(flag):
            if isinstance(flag, DataWarp):
                return 0 if any(jax.tree.leaves(flag)) else None
            return 0 if flag else None

        axes = jax.tree.map(axis_for, in_batched[1],
                            is_leaf=lambda value: isinstance(value, DataWarp))
        mapped_true = jax.vmap(true_fun, in_axes=(axes,), axis_size=axis_size)
        mapped_false = jax.vmap(false_fun, in_axes=(axes,), axis_size=axis_size)
        result = conditional_worlds(pred, mapped_true, mapped_false, args)

        def output_batches(value):
            if isinstance(value, DataWarp):
                return jax.tree.map_with_path(
                    lambda path, _: path[-1].name not in DATA_NON_VMAP, value)
            return True

        batched = jax.tree.map(output_batches, result,
                               is_leaf=lambda value: isinstance(value, DataWarp))
        return result, batched

    return apply(predicate, operand)


def select_state(mask, new, old):
    def select_array(a, b):
        shape = mask.shape + (1,) * (a.ndim - mask.ndim)
        return jnp.where(mask.reshape(shape), a, b)

    def select_leaf(a, b):
        if isinstance(a, DataWarp):


            return b.replace(**{
                field.name: select_array(getattr(a, field.name), getattr(b, field.name))
                for field in DataWarp.fields() if field.name not in DATA_NON_VMAP
            })
        return select_array(a, b)

    return jax.tree.map(select_leaf, new, old, is_leaf=lambda x: isinstance(x, DataWarp))
