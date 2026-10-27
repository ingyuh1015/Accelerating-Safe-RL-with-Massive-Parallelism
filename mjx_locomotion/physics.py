from __future__ import annotations

import jax
import jax.numpy as jnp
from mujoco import mjx


def make_init_data(mjx_model):

    return mjx.make_data(mjx_model)


def mjx_forward(mjx_model, data):

    return mjx.forward(mjx_model, data)


def physics_step(mjx_model, data, ctrl, n_substeps: int):


    data = data.replace(ctrl=ctrl)

    def body(_, d):
        return mjx.step(mjx_model, d)

    return jax.lax.fori_loop(0, n_substeps, body, data)


def site_world_velocity(data, site_id: int, site_bodyid: int, body_rootid: int):


    cvel = data.cvel[site_bodyid]
    ang = cvel[:3]
    lin_com = cvel[3:]
    com = data.subtree_com[body_rootid]
    site_xpos = data.site_xpos[site_id]
    lin = lin_com + jnp.cross(ang, site_xpos - com)
    return lin, ang
