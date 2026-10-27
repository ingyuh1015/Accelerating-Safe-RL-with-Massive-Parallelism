from __future__ import annotations
from typing import Optional

import numpy as np
import jax.numpy as jnp


def body_pos(mjx_data, body_id: int) -> jnp.ndarray:

    return mjx_data.xpos[body_id]


def body_mat(mjx_data, body_id: int) -> jnp.ndarray:

    return mjx_data.xmat[body_id].reshape(3, 3)


def body_xy(mjx_data, body_id: int) -> jnp.ndarray:

    return mjx_data.xpos[body_id, :2]


def set_qpos_slice(mjx_data, idx: slice, value):

    new_qpos = mjx_data.qpos.at[idx].set(value)
    return mjx_data.replace(qpos=new_qpos)


def set_body_pos(mjx_data, body_addr: int, xy: np.ndarray):

    new_qpos = mjx_data.qpos.at[body_addr:body_addr + 2].set(jnp.asarray(xy))
    return mjx_data.replace(qpos=new_qpos)
