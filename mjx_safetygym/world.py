from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import jax
import mujoco
from mujoco import mjx


@dataclass
class AgentLayout:

    xy_qpos_adr: tuple = (0, 1)
    heading_qpos_adr: int = 2
    heading_mode: str = 'hinge'
    n_qpos: int = 0
    n_qvel: int = 0
    n_act: int = 0


_OBS_SENSOR_CONF = ('accelerometer', 'velocimeter', 'gyro', 'magnetometer')


def _resolve_obs_sensors(mj_model) -> tuple:

    by_name = {}
    hinge_vel, ballangvel, hinge_pos, ballquat = [], [], [], []
    for i in range(mj_model.nsensor):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        adr, dim = int(mj_model.sensor_adr[i]), int(mj_model.sensor_dim[i])
        st = mj_model.sensor_type[i]
        by_name[name] = (adr, dim)
        if st == mujoco.mjtSensor.mjSENS_JOINTVEL:
            hinge_vel.append((name, adr, dim))
        elif st == mujoco.mjtSensor.mjSENS_BALLANGVEL:
            ballangvel.append((name, adr, dim))
        elif st == mujoco.mjtSensor.mjSENS_JOINTPOS:
            hinge_pos.append((name, adr, dim))
        elif st == mujoco.mjtSensor.mjSENS_BALLQUAT:
            ballquat.append((name, adr, dim))
    layout = []
    for n in _OBS_SENSOR_CONF:
        if n in by_name:
            adr, dim = by_name[n]
            layout.append((n, adr, dim, 'plain', dim))
    for n, adr, dim in hinge_vel:
        layout.append((n, adr, dim, 'plain', dim))
    for n, adr, dim in ballangvel:
        layout.append((n, adr, dim, 'plain', dim))
    for n, adr, dim in hinge_pos:
        layout.append((n, adr, dim, 'hinge_pos', 2))
    for n, adr, dim in ballquat:
        layout.append((n, adr, dim, 'ballquat', 9))
    return tuple(layout)


_VASE_SIZE = 0.1
_VASE_SINK = 4e-5
_VASE_DENSITY = 0.001
_VASE_RGBA = '0 1 1 1'
_BUTTON_SIZE = 0.1
_PILLAR_R = 0.2
_PILLAR_H = 0.5
_GREMLIN_SIZE = 0.1
_GREMLIN_DENSITY = 0.001


def _box_or_sphere(name: str, half: float, density: float, rgba: str, box: bool) -> str:
    if box:
        return (f'<geom name="{name}" type="box" size="{half} {half} {half}" '
                f'density="{density}" rgba="{rgba}"/>')
    return (f'<geom name="{name}" type="sphere" size="{half}" '
            f'density="{density}" rgba="{rgba}"/>')


def _vase_body_xml(i: int, box: bool = True) -> str:
    z = _VASE_SIZE - _VASE_SINK
    geom = _box_or_sphere(f'vase{i}', _VASE_SIZE, _VASE_DENSITY, _VASE_RGBA, box)
    return f'<body name="vase{i}" pos="0 0 {z}"><freejoint name="vase{i}"/>{geom}</body>'


def _push_box_body_xml(box: bool, size: float, density: float) -> str:
    geom = _box_or_sphere('push_box', size, density, '1 1 0 1', box)
    return f'<body name="push_box" pos="0 0 {size}"><freejoint name="push_box"/>{geom}</body>'


def _button_body_xml(i: int) -> str:


    return (f'<body name="button{i}" mocap="true" pos="{100.0 + i} 100 {_BUTTON_SIZE}">'
            f'<geom name="button{i}" type="sphere" size="{_BUTTON_SIZE}" rgba="1 .5 0 1"/>'
            f'</body>')


def _pillar_body_xml(i: int) -> str:
    half_cyl = _PILLAR_H / 2.0 - _PILLAR_R
    return (f'<body name="pillar{i}" mocap="true" pos="{100.0 + i} 200 {_PILLAR_H / 2.0}">'
            f'<geom name="pillar{i}" type="capsule" size="{_PILLAR_R} {half_cyl}" '
            f'rgba=".5 .5 1 1"/>'
            f'</body>')


def _gremlin_body_xml(i: int, box: bool) -> str:
    geom = _box_or_sphere(f'gremlin{i}', _GREMLIN_SIZE, _GREMLIN_DENSITY,
                          '.5 0 1 1', box)
    return f'<body name="gremlin{i}" mocap="true" pos="{100.0 + i} 300 {_GREMLIN_SIZE}">{geom}</body>'


@dataclass
class World:

    agent_xml_path: str
    frame_skip: int = 2
    n_vases: int = 0
    n_buttons: int = 0
    n_gremlins: int = 0
    n_pillars: int = 0
    push_box: bool = False
    push_box_size: float = 0.2
    push_box_density: float = 0.001

    timestep: float = field(init=False, default=0.0)
    mj_model: Optional[object] = None
    mjx_model: Optional[object] = None
    agent_layout: AgentLayout = field(default_factory=AgentLayout)
    body_ids: dict = field(default_factory=dict)
    site_ids: dict = field(default_factory=dict)
    geom_ids: dict = field(default_factory=dict)
    vase_body_ids: tuple = ()
    vase_qpos_adr: tuple = ()
    push_box_body_id: int = -1
    push_box_qpos_adr: int = -1
    obs_sensors: tuple = ()
    button_mocap_ids: tuple = ()
    pillar_mocap_ids: tuple = ()
    gremlin_mocap_ids: tuple = ()

    def build(self) -> "World":

        with open(self.agent_xml_path, 'r') as f:
            xml = f.read()


        box = 'cylinder' not in xml
        inject = ''
        inject += ''.join(_vase_body_xml(i, box) for i in range(self.n_vases))
        if self.push_box:
            inject += _push_box_body_xml(box, self.push_box_size, self.push_box_density)
        inject += ''.join(_button_body_xml(i) for i in range(self.n_buttons))
        inject += ''.join(_pillar_body_xml(i) for i in range(self.n_pillars))
        inject += ''.join(_gremlin_body_xml(i, box) for i in range(self.n_gremlins))
        if inject:
            assert '</worldbody>' in xml, 'agent XML has no <worldbody>'
            xml = xml.replace('</worldbody>', inject + '</worldbody>')

        mj_model = mujoco.MjModel.from_xml_string(xml)


        self.timestep = float(mj_model.opt.timestep)

        self.mj_model = mj_model
        self.mjx_model = mjx.put_model(mj_model, device=jax.local_devices()[0])

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


        agent_jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, 'agent')
        if agent_jid >= 0 and mj_model.jnt_type[agent_jid] == mujoco.mjtJoint.mjJNT_FREE:
            heading_mode = 'free'
            heading_adr = int(mj_model.jnt_qposadr[agent_jid]) + 3
        else:
            heading_mode = 'hinge'
            heading_adr = 2 if mj_model.nq >= 3 else -1

        self.agent_layout = AgentLayout(
            xy_qpos_adr=(0, 1),
            heading_qpos_adr=heading_adr,
            heading_mode=heading_mode,
            n_qpos=int(mj_model.nq),
            n_qvel=int(mj_model.nv),
            n_act=int(mj_model.nu),
        )
        self.obs_sensors = _resolve_obs_sensors(mj_model)


        if self.n_vases > 0:
            bids, qadr = [], []
            for i in range(self.n_vases):
                bids.append(self.body_ids[f'vase{i}'])
                jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, f'vase{i}')
                qadr.append(int(mj_model.jnt_qposadr[jid]))
            self.vase_body_ids = tuple(bids)
            self.vase_qpos_adr = tuple(qadr)
        if self.push_box:
            self.push_box_body_id = self.body_ids['push_box']
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, 'push_box')
            self.push_box_qpos_adr = int(mj_model.jnt_qposadr[jid])

        def _mocap_ids(prefix, n):
            return tuple(int(mj_model.body_mocapid[self.body_ids[f'{prefix}{i}']])
                         for i in range(n))
        self.button_mocap_ids = _mocap_ids('button', self.n_buttons)
        self.pillar_mocap_ids = _mocap_ids('pillar', self.n_pillars)
        self.gremlin_mocap_ids = _mocap_ids('gremlin', self.n_gremlins)
        return self

    @property
    def action_dim(self) -> int:
        return self.agent_layout.n_act

    @property
    def nq(self) -> int:
        return self.agent_layout.n_qpos

    @property
    def nv(self) -> int:
        return self.agent_layout.n_qvel


def build_world(agent_xml_path: str, frame_skip: int = 2, n_vases: int = 0,
                n_buttons: int = 0, n_gremlins: int = 0, n_pillars: int = 0,
                push_box: bool = False, push_box_size: float = 0.2,
                push_box_density: float = 0.001) -> World:
    return World(agent_xml_path=agent_xml_path, frame_skip=frame_skip,
                 n_vases=n_vases, n_buttons=n_buttons, n_gremlins=n_gremlins,
                 n_pillars=n_pillars, push_box=push_box,
                 push_box_size=push_box_size, push_box_density=push_box_density).build()
