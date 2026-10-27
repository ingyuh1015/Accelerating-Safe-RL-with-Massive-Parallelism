from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import mujoco
from mujoco import mjx


@dataclass
class World:


    robot_xml_path: str

    timestep: float = 0.0
    n_substeps: int = 10
    mj_model: Optional[object] = None
    mjx_model: Optional[object] = None
    body_ids: dict = field(default_factory=dict)
    site_ids: dict = field(default_factory=dict)
    geom_ids: dict = field(default_factory=dict)
    sensor_adr: dict = field(default_factory=dict)

    def build(self) -> "World":

        mj_model = mujoco.MjModel.from_xml_path(self.robot_xml_path)
        self.timestep = float(mj_model.opt.timestep)

        self.mj_model = mj_model
        self.mjx_model = mjx.put_model(mj_model)

        for i in range(mj_model.nbody):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                self.body_ids[name] = i
        for i in range(mj_model.nsite):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, i)
            if name:
                self.site_ids[name] = i
        for i in range(mj_model.ngeom):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name:
                self.geom_ids[name] = i
        for i in range(mj_model.nsensor):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name:
                self.sensor_adr[name] = (int(mj_model.sensor_adr[i]),
                                         int(mj_model.sensor_dim[i]))
        return self

    @property
    def action_dim(self) -> int:
        return int(self.mj_model.nu)

    @property
    def nq(self) -> int:
        return int(self.mj_model.nq)

    @property
    def nv(self) -> int:
        return int(self.mj_model.nv)


def build_world(robot_xml_path: str, *, n_substeps: int = 10) -> World:
    return World(robot_xml_path=robot_xml_path, n_substeps=n_substeps).build()
