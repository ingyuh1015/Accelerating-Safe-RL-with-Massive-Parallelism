from __future__ import annotations
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from mujoco import mjx


class State(NamedTuple):


    data: Any
    obs: jax.Array
    reward: jax.Array
    cost: jax.Array
    done: jax.Array
    key: jax.Array
    info: dict


def physics_step(mjx_model, data, ctrl, n_substeps: int):


    data = data.replace(ctrl=ctrl)

    def body(_, d):
        return mjx.step(mjx_model, d)

    return jax.lax.fori_loop(0, n_substeps, body, data)


def agent_xy(data, xy_adr=(0, 1)) -> jax.Array:

    return jnp.stack([data.qpos[xy_adr[0]], data.qpos[xy_adr[1]]])


def agent_heading(data, heading_adr: int = 2, mode: str = 'hinge') -> jax.Array:


    if mode == 'free':
        w = data.qpos[heading_adr + 0]
        x = data.qpos[heading_adr + 1]
        y = data.qpos[heading_adr + 2]
        z = data.qpos[heading_adr + 3]
        return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return data.qpos[heading_adr]


def quat_to_mat(q: jax.Array) -> jax.Array:

    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ])


def agent_vel_xy(data, xy_adr=(0, 1)) -> jax.Array:

    return jnp.stack([data.qvel[xy_adr[0]], data.qvel[xy_adr[1]]])


def make_init_data(mjx_model):

    return mjx.make_data(mjx_model, device=jax.local_devices()[0])
