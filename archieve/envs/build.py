from __future__ import annotations
import dataclasses
import os
from typing import Any, Mapping
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from archieve.utils import get_task_env_class
from archieve import ConstraintConfig, TaskEnvConfig
from archieve.envs.agents import BaseAgentConfig, PointAgent, CarAgent
from archieve.envs.tasks import BaseObjConfig, build_object_config, cost
from archieve.envs import Wrapper, EpisodeWrapper, VmapWrapper

def _build_mj_model(mjcf: mujoco.MjSpec, objects: dict[str, BaseObjConfig], agent: BaseAgentConfig) -> tuple[mujoco.MjModel, mjx.Model]:
    n_g = int(agent.collision_geom_count)
    n_o = sum((cfg.num * int(cfg.collision_geom_count) for cfg in objects.values() if cfg.contact))
    budget = n_g * (n_o + 1) + n_o
    mjcf.nconmax = budget
    mjcf.njmax = budget
    mj_model = mjcf.compile()
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    iters = int(os.environ.get('MJX_SOLVER_ITER', '1'))
    ls_iters = int(os.environ.get('MJX_SOLVER_LS', '4'))
    if os.environ.get('MJX_EULERDAMP', 'off').lower() not in ('on', '1', 'true'):
        mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    mj_model.opt.iterations = iters
    mj_model.opt.ls_iterations = ls_iters
    return (mj_model, mjx.put_model(mj_model, device=jax.local_devices()[0]))

def _build_constraint(constraint_config: Mapping[str, Any], objects: dict[str, BaseObjConfig], mj_model: mujoco.MjModel, agent_gids: jnp.ndarray) -> ConstraintConfig:
    name = str(constraint_config['name'])
    cost_limit = float(constraint_config['cost_limit'])
    cost_type = str(constraint_config.get('type', 'binary'))
    if name == 'hazards':
        return ConstraintConfig(name=name, cost_limit=cost_limit, compute_fn=cost.compute_hazards, params={'radius': float(objects['hazards'].size)}, cost_type=cost_type)
    if name == 'out_of_boundary':
        boundaries = tuple(((0 if w[0] == 'x' else 1, w[1]) for w in objects['sigwalls'].walls))
        return ConstraintConfig(name=name, cost_limit=cost_limit, compute_fn=cost.compute_out_of_boundary, params={'boundaries': boundaries}, cost_type=cost_type)
    if name == 'vases':
        obj = objects[name]
        obj_gids = jnp.asarray([int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, obj.geom_id(i))) for i in range(obj.num)], dtype=jnp.int32)
        return ConstraintConfig(name=name, cost_limit=cost_limit, compute_fn=getattr(cost, f'compute_{name}'), params={'agent_geom_ids': agent_gids, 'obj_geom_ids': obj_gids}, cost_type=cost_type)
    if name in ('velocity_limit_high', 'velocity_limit_low'):
        return ConstraintConfig(name=name, cost_limit=cost_limit, compute_fn=getattr(cost, f'compute_{name}'), params={'threshold': float(constraint_config['threshold'])}, cost_type=cost_type)
    raise ValueError(f'Unsupported constraint name: {name}')

def _agent_geom_ids(mj_model: mujoco.MjModel) -> jnp.ndarray:
    root_bid = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'agent'))
    body_ids: list[int] = []
    for bid in range(int(mj_model.nbody)):
        current = bid
        while current > 0 and current != root_bid:
            current = int(mj_model.body_parentid[current])
        if current == root_bid:
            body_ids.append(bid)
    return jnp.asarray([gid for gid in range(int(mj_model.ngeom)) if int(mj_model.geom_bodyid[gid]) in body_ids], dtype=jnp.int32)

def build_env(*, task_env_config: TaskEnvConfig, episode_length: int, batch_size: int, gamma: float) -> Wrapper:
    raw = dict(task_env_config.task_config)
    objects: dict[str, BaseObjConfig] = {}
    for object_config in raw['objects']:
        name = str(object_config['name'])
        num = int(object_config['num'])
        objects[name] = build_object_config(name, object_config, num)
    task_type = str(raw['task_type'])
    task_specific = dict(raw.get('task_specific', {}))
    placements_extents = tuple(raw['placements_extents'])
    agent_types = {'point': PointAgent, 'car': CarAgent}
    agent_name = str(raw['agent_type'])
    if agent_name not in agent_types:
        raise ValueError(f'Unknown agent: {agent_name}. Available: {list(agent_types)}')
    agent = agent_types[agent_name]()
    agent_overrides = dict(raw.get('agent', {}))
    if agent_overrides:
        known = {f.name for f in dataclasses.fields(agent)}
        unknown = set(agent_overrides) - known
        if unknown:
            raise ValueError(f'Unknown agent YAML keys: {sorted(unknown)}')
        agent = dataclasses.replace(agent, **agent_overrides)
    mjcf = mujoco.MjSpec.from_file(agent.xml_path)
    for obj_cfg in objects.values():
        obj_cfg.add(mjcf.worldbody, mjcf)
    (mj_model, mjx_model) = _build_mj_model(mjcf, objects, agent)
    for (name, cfg) in list(objects.items()):
        objects[name] = cfg.resolve(mj_model)
    agent_gids = _agent_geom_ids(mj_model)
    constraints = tuple(sorted((_build_constraint(item, objects, mj_model, agent_gids) for item in raw.get('constraints', [])), key=lambda item: item.name))
    sim_dt = float(mj_model.opt.timestep)
    n_substeps = int(agent.n_substeps)
    env_cls = get_task_env_class(task_type)
    raw_env = env_cls(mj_model=mj_model, mjx_model=mjx_model, agent=agent, objects=objects, placements_extents=placements_extents, constraints=constraints, action_size=int(agent.action_size), ctrl_dt=sim_dt * n_substeps, n_substeps=n_substeps, agent_geom_ids=agent_gids, **task_specific)
    env = VmapWrapper(raw_env, batch_size=batch_size)
    env = EpisodeWrapper(env, episode_length, gamma=gamma)
    return env
