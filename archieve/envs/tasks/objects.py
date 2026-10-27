from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Mapping
import jax.numpy as jnp
import mujoco
from archieve import get_dtype

@dataclass(frozen=True)
class BaseObjConfig:
    num: int
    size: float
    contact: bool
    keepout: float
    lidar_observable: bool
    obj_type: str
    model_info: Any = None
    collision_geom_count: int = 1

    def add(self, world: mujoco.MjsBody, mjcf: mujoco.MjSpec) -> None:
        raise NotImplementedError

    def resolve(self, mj_model: Any) -> 'BaseObjConfig':
        return dataclasses.replace(self, model_info=SimpleNamespace())

    def get_positions(self, data: Any, info: dict[str, Any]) -> jnp.ndarray:
        raise NotImplementedError

@dataclass(frozen=True)
class HazardConfig(BaseObjConfig):
    obj_type: str = 'geom'
    contact: bool = False
    size: float = 0.2
    keepout: float = 0.18
    lidar_observable: bool = True
    geom_id: Callable[[int], str] = lambda i: f'hazard_{i}'

    def add(self, world: mujoco.MjsBody, mjcf: mujoco.MjSpec) -> None:
        for i in range(self.num):
            world.add_geom(name=f'hazard_{i}', type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[self.size, 0, 0], pos=[0, 0, 0.1], rgba=[0, 0, 1, 0.25], contype=0, conaffinity=0)

    def get_positions(self, data: Any, info: dict[str, Any]) -> jnp.ndarray:
        return info['hazards_positions']

@dataclass(frozen=True)
class SigwallConfig(BaseObjConfig):
    obj_type: str = 'geom'
    contact: bool = False
    size: float = 0.1
    keepout: float = 0.0
    lidar_observable: bool = False
    walls: tuple = field(default_factory=tuple)

    def add(self, world: mujoco.MjsBody, mjcf: mujoco.MjSpec) -> None:
        half_thickness = 0.05
        half_perp = 3.5
        half_height = 0.3
        center_z = 0.25
        for (i, (axis, coord)) in enumerate(self.walls):
            if axis == 'x':
                world.add_geom(name=f'sigwall_{i}', type=mujoco.mjtGeom.mjGEOM_BOX, size=[half_thickness, half_perp, half_height], pos=[float(coord), 0.0, center_z], rgba=[1.0, 0.5, 0.0, 0.1], contype=0, conaffinity=0)
            else:
                world.add_geom(name=f'sigwall_{i}', type=mujoco.mjtGeom.mjGEOM_BOX, size=[half_perp, half_thickness, half_height], pos=[0.0, float(coord), center_z], rgba=[1.0, 0.5, 0.0, 0.1], contype=0, conaffinity=0)

@dataclass(frozen=True)
class VaseConfig(BaseObjConfig):
    obj_type: str = 'freejoint'
    contact: bool = True
    size: float = 0.1
    keepout: float = 0.15
    lidar_observable: bool = True
    body_id: Callable[[int], str] = lambda i: f'vase_{i}'
    joint_id: Callable[[int], str] = lambda i: f'vase_{i}_joint'
    geom_id: Callable[[int], str] = lambda i: f'vase_{i}'

    def add(self, world: mujoco.MjsBody, mjcf: mujoco.MjSpec) -> None:
        for i in range(self.num):
            body = world.add_body(name=f'vase_{i}', pos=[0, 0, -10.0])
            body.add_freejoint(name=f'vase_{i}_joint')
            body.add_geom(name=f'vase_{i}', type=mujoco.mjtGeom.mjGEOM_BOX, size=[self.size, self.size, self.size], density=0.001, rgba=[0, 1, 1, 1], contype=2, conaffinity=3)

    def resolve(self, mj_model: Any) -> 'VaseConfig':
        adrs = [int(mj_model.jnt_qposadr[int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, self.joint_id(i)))]) for i in range(self.num)]
        return dataclasses.replace(self, model_info=SimpleNamespace(jnt_qposadr=jnp.array(adrs, dtype=jnp.int32)))

    def get_positions(self, data: Any, info: dict[str, Any]) -> jnp.ndarray:
        adrs = self.model_info.jnt_qposadr
        indices = adrs[:, None] + jnp.array([0, 1], dtype=jnp.int32)[None, :]
        return data.qpos[indices].astype(get_dtype())

@dataclass(frozen=True)
class GoalConfig(BaseObjConfig):
    obj_type: str = 'geom'
    contact: bool = False
    size: float = 0.3
    keepout: float = 0.305
    lidar_observable: bool = True
    spawn_radius: float = 0.0
    geom_id: Callable[[int], str] = lambda i: 'goal'

    def add(self, world: mujoco.MjsBody, mjcf: mujoco.MjSpec) -> None:
        world.add_geom(name='goal', type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[self.size, 0, 0], pos=[1, 1, 0.1], rgba=[0, 1, 0, 0.25], contype=0, conaffinity=0)

    def resolve(self, mj_model: Any) -> 'GoalConfig':
        gid = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, self.geom_id(0)))
        return dataclasses.replace(self, model_info=SimpleNamespace(geom_id=gid))

    def get_positions(self, data: Any, info: dict[str, Any]) -> jnp.ndarray:
        return info['target_pos'][None, :]

@dataclass(frozen=True)
class CircleConfig(BaseObjConfig):
    obj_type: str = 'geom'
    contact: bool = False
    size: float = 0.0
    keepout: float = 0.0
    lidar_observable: bool = True
    radius: float = 1.5

    def add(self, world: mujoco.MjsBody, mjcf: mujoco.MjSpec) -> None:
        world.add_geom(name='circle', type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[self.radius, 0.01, 0], pos=[0, 0, 0.01], rgba=[0, 1, 0, 0.1], contype=0, conaffinity=0)

    def get_positions(self, data: Any, info: dict[str, Any]) -> jnp.ndarray:
        return jnp.zeros((1, 2), dtype=get_dtype())

def build_object_config(name: str, object_config: Mapping[str, Any], num: int) -> BaseObjConfig:
    object_types = {'hazards': HazardConfig, 'sigwalls': SigwallConfig, 'vases': VaseConfig, 'goal': GoalConfig, 'circle': CircleConfig}
    key = str(name)
    if key not in object_types:
        raise ValueError(f'Unknown object kind: {name!r}. Available: {sorted(object_types)}')
    cls = object_types[key]
    field_names = {f.name for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {'num': num}
    for (key, value) in object_config.items():
        if key in field_names and key != 'num':
            if key == 'walls' and isinstance(value, list):
                kwargs[key] = tuple(((str(w['axis']), float(w['coord'])) for w in value))
            else:
                kwargs[key] = value
    return cls(**kwargs)
