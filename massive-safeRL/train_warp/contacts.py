from __future__ import annotations

from functools import lru_cache
from math import prod

import jax
import jax.numpy as jnp


def _reduce_contact_flags(
    geoms,
    distances,
    margins,
    world_ids,
    count,
    target_ids,
    partner_ids,
    *,
    num_worlds: int,
    use_margin: bool,
    restrict_partner: bool,
):


    count = jnp.asarray(count).reshape(())
    valid = (jnp.arange(distances.shape[0]) < count)
    valid &= (world_ids >= 0) & (world_ids < num_worlds)
    if use_margin:
        valid &= (distances < margins) & (geoms[:, 0] != geoms[:, 1])
    else:
        valid &= distances <= 0.0

    left = geoms[:, 0, None] == target_ids[None, :]
    right = geoms[:, 1, None] == target_ids[None, :]
    if restrict_partner:
        right_partner = jnp.any(geoms[:, 1, None] == partner_ids[None, :], axis=-1)
        left_partner = jnp.any(geoms[:, 0, None] == partner_ids[None, :], axis=-1)
        left &= right_partner[:, None]
        right &= left_partner[:, None]
    matches = ((left | right) & valid[:, None]).astype(jnp.int32)


    destinations = jnp.where(valid, world_ids, num_worlds)
    flags = jnp.zeros((num_worlds, target_ids.shape[0]), dtype=jnp.int32)
    flags = flags.at[destinations].max(matches, mode="drop")
    return flags.astype(jnp.bool_)


@lru_cache(maxsize=4)
def _query_function(use_margin: bool, restrict_partner: bool):
    @jax.custom_batching.custom_vmap
    def query(qpos, geoms, distances, margins, world_ids, count, targets, partners):
        world_shape = qpos.shape[:-1]
        flags = _reduce_contact_flags(
            geoms, distances, margins, world_ids, count, targets, partners,
            num_worlds=prod(world_shape),
            use_margin=use_margin, restrict_partner=restrict_partner,
        )
        return flags.reshape(world_shape + (targets.shape[0],))

    @query.def_vmap
    def query_vmap(axis_size, in_batched, *args):
        del axis_size
        if any(in_batched[1:]):
            raise ValueError(
                "Warp contact arrays and queried geom IDs must be shared across "
                "worlds; only the qpos world dimensions may be vmapped."
            )


        return query(*args), in_batched[0]

    return query


def contact_flags(data, geom_ids, other_geom_ids=None, *, use_margin=False):


    if not isinstance(use_margin, bool):
        raise TypeError("use_margin must be a static Python bool")
    impl = data._impl
    if not hasattr(impl, "contact__worldid"):
        raise TypeError("contact_flags requires MJX-Warp Data")
    targets = jnp.asarray(geom_ids, dtype=jnp.int32)
    partners = jnp.asarray(() if other_geom_ids is None else other_geom_ids,
                           dtype=jnp.int32)
    if targets.ndim != 1 or partners.ndim != 1:
        raise ValueError("geom_ids and other_geom_ids must be one-dimensional")
    query = _query_function(use_margin, other_geom_ids is not None)
    return query(
        data.qpos,
        impl.contact__geom,
        impl.contact__dist,
        impl.contact__includemargin,
        impl.contact__worldid,
        impl.nacon,
        targets,
        partners,
    )
