from __future__ import annotations
from typing import Any
import jax.numpy as jnp
from archieve import get_dtype

def compute_hazards(data: Any, info: dict[str, Any], *, radius: float) -> None:
    del data
    agent_pos = info['agent_pos']
    objects = info['hazards_positions']
    dist_sq = jnp.sum((objects - agent_pos[None, :]) ** 2, axis=-1)
    info['cost_hazards'] = jnp.any(dist_sq < radius ** 2).astype(get_dtype())

def compute_vases(data: Any, info: dict[str, Any], *, agent_geom_ids: jnp.ndarray, obj_geom_ids: jnp.ndarray) -> None:
    geoms = data.contact.geom
    dist = data.contact.dist
    g1 = geoms[:, 0]
    g2 = geoms[:, 1]
    g1_is_agent = jnp.any(g1[:, None] == agent_geom_ids[None, :], axis=-1)
    g2_is_agent = jnp.any(g2[:, None] == agent_geom_ids[None, :], axis=-1)
    g1_is_obj = jnp.any(g1[:, None] == obj_geom_ids[None, :], axis=-1)
    g2_is_obj = jnp.any(g2[:, None] == obj_geom_ids[None, :], axis=-1)
    pair_hit = g1_is_agent & g2_is_obj | g1_is_obj & g2_is_agent
    info['cost_vases'] = jnp.any(pair_hit & (dist <= 0.0)).astype(get_dtype())

def compute_out_of_boundary(data: Any, info: dict[str, Any], *, boundaries: tuple) -> None:
    del data
    agent_pos = info['agent_pos']
    result = jnp.asarray(False)
    for (axis_idx, coord) in boundaries:
        pos_on_axis = agent_pos[axis_idx]
        result = result | (jnp.sign(coord) * pos_on_axis > abs(coord))
    info['cost_out_of_boundary'] = result.astype(get_dtype())

def compute_velocity_limit_high(data: Any, info: dict[str, Any], *, threshold: float) -> None:
    del data
    info['cost_velocity_limit_high'] = (info['agent_speed'] > threshold).astype(get_dtype())

def compute_velocity_limit_low(data: Any, info: dict[str, Any], *, threshold: float) -> None:
    del data
    info['cost_velocity_limit_low'] = (info['agent_speed'] < threshold).astype(get_dtype())
