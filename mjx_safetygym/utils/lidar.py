from __future__ import annotations

import jax.numpy as jnp


def pseudo_lidar(
    agent_xy: jnp.ndarray,
    agent_heading: jnp.ndarray,
    obj_xys: jnp.ndarray,
    n_bins: int = 16,
    max_dist: float = 3.0,
    obj_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:


    cos_h = jnp.cos(-agent_heading)
    sin_h = jnp.sin(-agent_heading)
    rel = obj_xys - agent_xy
    local_x = rel[:, 0] * cos_h - rel[:, 1] * sin_h
    local_y = rel[:, 0] * sin_h + rel[:, 1] * cos_h
    dists = jnp.sqrt(local_x * local_x + local_y * local_y)


    sensor = jnp.maximum(0.0, max_dist - dists) / max_dist
    if obj_mask is not None:
        sensor = sensor * obj_mask.astype(sensor.dtype)


    bin_size = (2.0 * jnp.pi) / n_bins
    angles = jnp.mod(jnp.arctan2(local_y, local_x), 2.0 * jnp.pi)
    bin_pos = angles / bin_size
    bin_idx = jnp.floor(bin_pos).astype(jnp.int32) % n_bins
    alias = bin_pos - bin_idx


    plus_idx  = (bin_idx + 1) % n_bins
    minus_idx = (bin_idx - 1) % n_bins
    lidar = jnp.zeros((n_bins,), dtype=sensor.dtype)
    lidar = lidar.at[bin_idx].max(sensor)
    lidar = lidar.at[plus_idx].max(alias * sensor)
    lidar = lidar.at[minus_idx].max((1.0 - alias) * sensor)
    return lidar


def compass(
    agent_xy: jnp.ndarray,
    agent_heading: jnp.ndarray,
    target_xy: jnp.ndarray,
) -> jnp.ndarray:

    cos_h = jnp.cos(-agent_heading)
    sin_h = jnp.sin(-agent_heading)
    rel = target_xy - agent_xy
    local_x = rel[0] * cos_h - rel[1] * sin_h
    local_y = rel[0] * sin_h + rel[1] * cos_h
    norm = jnp.sqrt(local_x * local_x + local_y * local_y) + 1e-8
    return jnp.stack([local_x, local_y]) / norm
