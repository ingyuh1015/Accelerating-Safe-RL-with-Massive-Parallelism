from __future__ import annotations

import math
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import mjx_safetygym
from mjx_safetygym.tasks.safe_navigation.goal import GoalLevel0
from mjx_safetygym.tasks.safe_navigation.circle import CircleLevel0
from mjx_safetygym import build_world

from massive_saferl import get_dtype
from massive_saferl.utils import backend


def _check_keys(values, allowed, context):
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"Unsupported {context} keys: {sorted(unknown)}")


def _candidate(key, radius, fixed, separations):
    candidates = jax.random.uniform(key, (21, 2), minval=-radius, maxval=radius)
    distance_sq = jnp.sum((candidates[1:, None] - fixed[None]) ** 2, axis=-1)
    valid = jnp.all(distance_sq > jnp.asarray(separations)[None] ** 2, axis=1)
    index = jnp.max(jnp.where(valid, jnp.arange(1, 21), 0))
    return candidates[index]


class _Task(GoalLevel0):
    def __init__(self, world, raw, objects, constraints):
        self.family = raw['task_type']
        self.objects = objects
        self.constraints = constraints
        self.constraint_names = tuple(c['name'] for c in constraints)
        self.cost_limits = tuple(float(c['cost_limit']) for c in constraints)
        self.n_costs = len(constraints)
        self.n_hazards = objects.get('hazards', {}).get('num', 0)
        self.n_vases = objects.get('vases', {}).get('num', 0)
        self.num_steps = int(raw['episode_length'])
        settings = raw['task_specific']
        self.n_bins = int(settings['lidar_num_bins'])
        self.lidar_max_dist = float(settings['lidar_max_dist'])
        self.circle_radius = objects.get('circle', {}).get('radius', 1.5)
        self.reward_factor = float(settings.get('reward_factor', 0.1))
        self._radius = (raw['placements_extents'][2] - raw['placements_extents'][0]) / 2
        self._goal_radius = objects.get('goal', {}).get('spawn_radius', 0.0) or self._radius
        self._randomize_heading = raw.get('agent', {}).get('randomize_heading', True)
        super().__init__(world, extents=tuple(raw['placements_extents']),
                         goal_size=objects.get('goal', {}).get('size', 0.3),
                         hazard_size=objects.get('hazards', {}).get('size', 0.2))
        self.agent_keepout = float(raw.get('agent', {}).get('keepout', 0.4))
        self.goal_keepout = float(objects.get('goal', {}).get('keepout', 0.0))
        self._vase_indices = jnp.asarray(world.vase_qpos_adr, dtype=jnp.int32)[:, None] + jnp.arange(2)
        self._vase_geoms = jnp.asarray([world.geom_ids[f'vase{i}'] for i in range(self.n_vases)])
        root = world.body_ids['agent']
        bodies = []
        for bid in range(world.mj_model.nbody):
            parent = bid
            while parent > 0 and parent != root:
                parent = int(world.mj_model.body_parentid[parent])
            if parent == root:
                bodies.append(bid)
        self._agent_geoms = jnp.asarray([i for i, body in enumerate(world.mj_model.geom_bodyid) if body in bodies])

    def sample_layout(self, key):
        agent_key, goal_key, object_key = jax.random.split(key, 3)
        agent = self._sample_xy(agent_key)
        goal = (_candidate(goal_key, self._goal_radius, agent[None],
                           jnp.array([self.agent_keepout + self.goal_keepout]))
                if self.family == 'goal' else jnp.zeros(2))
        result = {'agent_xy': agent, 'goal_xy': goal}
        for name in ('hazards', 'vases'):
            cfg = self.objects.get(name, {'num': 0, 'keepout': 0.0})
            count, keepout = cfg['num'], cfg['keepout']
            if not count:
                result[f'{name}_xy'] = jnp.zeros((0, 2))
                continue
            rng = jax.random.fold_in(object_key, int.from_bytes(name.encode(), 'little') % (2 ** 31))
            fixed = jnp.stack([agent, goal])
            separations = jnp.array([keepout + self.agent_keepout, keepout + self.goal_keepout,
                                    *([2 * keepout] * count)])

            def place(placed, item):
                index, subkey = item
                position = _candidate(subkey, self._radius, jnp.concatenate([fixed, placed]), separations)
                return placed.at[index].set(position), position

            _, positions = jax.lax.scan(place, jnp.full((count, 2), 100.0),
                                       (jnp.arange(count), jax.random.split(rng, count)))
            result[f'{name}_xy'] = positions
        return result

    def _resample_goal_keepout(self, key, data, info):
        fixed = jnp.concatenate([data.qpos[:2][None], info['hazards_xy'], data.qpos[self._vase_indices]])
        separations = jnp.array([self.agent_keepout + self.goal_keepout,
                                *([self.goal_keepout] * (self.n_hazards + self.n_vases))])
        return _candidate(key, self._goal_radius, fixed, separations)

    def _apply_layout_extras(self, data, layout):
        if self.n_vases:
            positions = jnp.concatenate([layout['vases_xy'], jnp.full((self.n_vases, 1), 0.1),
                                         jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]), (self.n_vases, 1))], axis=1)
            indices = jnp.asarray(self.world.vase_qpos_adr)[:, None] + jnp.arange(7)
            data = data.replace(qpos=data.qpos.at[indices].set(positions))
        return data

    def _write_heading(self, qpos, head):
        return super()._write_heading(qpos, head) if self._randomize_heading else qpos

    def _post_physics_info(self, data, info):
        info['vases_xy'] = data.qpos[self._vase_indices]
        info['move_dist'] = jnp.linalg.norm(data.qpos[:2] - info['previous_agent_xy'])
        info['agent_speed'] = info['move_dist'] / (self.world.timestep * self.frame_skip)
        info['goal_reached'] = ((self.dist_goal(data, info) < self.goal_size).astype(get_dtype())
                                if self.family == 'goal' else jnp.zeros((), dtype=get_dtype()))
        return info

    def is_goal_reached(self, data, info):
        return info['goal_reached']

    def _on_goal_reached(self, key, data, info, reached):
        return super()._on_goal_reached(key, data, info, reached) if self.family == 'goal' else info

    def _update_trackers(self, data, info):
        if self.family == 'circle':
            xy = data.qpos[:2]
            info['goal_xy'] = xy * self.circle_radius / jnp.maximum(jnp.linalg.norm(xy), 1e-6)
        info = super()._update_trackers(data, info)
        info['previous_agent_xy'] = data.qpos[:2]
        return info

    def reset(self, key):
        state = super().reset(key)
        info = dict(state.info, previous_agent_xy=state.data.qpos[:2])
        info = self._post_physics_info(state.data, info)
        reward = self.reward_vector(info['last_dist_goal'], state.data, jnp.zeros(self.world.action_dim), info)
        cost = self.cost_vector(state.data, info)
        key, goal_key = jax.random.split(state.key)
        info = self._on_goal_reached(goal_key, state.data, info, info['goal_reached'])
        info = self._update_trackers(state.data, info)
        return state._replace(obs=self.observation(state.data, info), reward=reward, cost=cost, key=key, info=info)

    def step(self, state, action):
        return super().step(state, jnp.asarray(action, dtype=state.data.ctrl.dtype))

    def reward_vector(self, prev_dist, data, action, info):
        if self.family == 'goal':
            return super().reward_vector(prev_dist, data, action, info).astype(get_dtype())
        x, y = data.qpos[:2]
        vx, vy = data.qvel[:2]
        reward = (-vx * y + vy * x) / (1 + jnp.abs(jnp.linalg.norm(data.qpos[:2]) - self.circle_radius))
        return jnp.asarray([reward * self.reward_factor], dtype=get_dtype())

    def cost_vector(self, data, info):
        costs = []
        for constraint in self.constraints:
            name = constraint['name']
            if name == 'hazards':
                value = jnp.any(jnp.sum((info['hazards_xy'] - data.qpos[:2]) ** 2, axis=-1) < self.hazard_size ** 2)
            elif name == 'vases':
                contact_query = getattr(self.world, 'contact_query', None)
                if contact_query is not None:
                    value = jnp.any(contact_query(data, self._vase_geoms, self._agent_geoms))
                else:
                    geoms = data.contact.geom
                    agent1 = jnp.any(geoms[:, 0, None] == self._agent_geoms, axis=-1)
                    agent2 = jnp.any(geoms[:, 1, None] == self._agent_geoms, axis=-1)
                    vase1 = jnp.any(geoms[:, 0, None] == self._vase_geoms, axis=-1)
                    vase2 = jnp.any(geoms[:, 1, None] == self._vase_geoms, axis=-1)
                    value = jnp.any(((agent1 & vase2) | (agent2 & vase1)) & (data.contact.dist <= 0))
            elif name == 'out_of_boundary':
                value = jnp.array(False)
                for wall in self.objects['sigwalls']['walls']:
                    axis, coordinate = (0 if wall['axis'] == 'x' else 1), wall['coord']
                    value = value | (jnp.sign(coordinate) * data.qpos[axis] > abs(coordinate))
            elif name == 'velocity_limit_high':
                value = info['agent_speed'] > constraint['threshold']
            else:
                value = info['agent_speed'] < constraint['threshold']
            costs.append(value)
        return jnp.asarray(costs, dtype=get_dtype())

    def observation(self, data, info):
        observation = (super().observation(data, info) if self.family == 'goal'
                       else CircleLevel0.observation(self, data, info))
        return observation.astype(get_dtype())

    @property
    def obs_dim(self):
        return super().obs_dim if self.family == 'goal' else self._sensor_obs_dim + self.n_bins


class _NativeTask:


    def __init__(self, raw, *, world_transform=None, apply_solver_overrides=True):
        _check_keys(raw, ('source', 'agent_type', 'task_type', 'level', 'episode_length',
                          'constraints'), 'native task')
        self.family = raw['task_type']
        self.agent_type = raw['agent_type']
        self._world_transform = world_transform
        self._apply_solver_overrides = apply_solver_overrides
        self.num_steps = int(raw['episode_length'])
        if self.num_steps < 1:
            raise ValueError('episode_length must be positive')
        level = raw['level']
        if self.family == 'velocity':
            if level not in (0, 1):
                raise ValueError('Velocity version must be 0 or 1')
            native = mjx_safetygym.make_velocity(self.agent_type, version=level,
                                                world_transform=self._configure_world).task
        else:
            if self.family not in ('goal', 'circle', 'button', 'push'):
                raise ValueError(f'Unsupported native task family: {self.family}')
            _, native = mjx_safetygym.make_task(self.agent_type, self.family, level,
                                               world_transform=self._configure_world)
        self.native_task = native
        self.world = native.world
        bounds = jnp.asarray(self.world.mj_model.actuator_ctrlrange)
        self._action_scale = (bounds[:, 1] - bounds[:, 0]) / 2
        self._action_offset = (bounds[:, 1] + bounds[:, 0]) / 2
        self.mjx_model = native.mjx_model
        self.frame_skip = native.frame_skip
        self.n_rewards = native.n_rewards
        self.n_hazards = int(getattr(native, 'n_hazards', 0))
        self.n_vases = int(getattr(native, 'n_vases', 0))
        self.goal_size = float(getattr(native, 'goal_size', 0.3))
        self.hazard_size = float(getattr(native, 'hazard_size', 0.2))
        self.circle_radius = float(getattr(native, 'circle_radius', 1.5))
        self.is_variable_episode = self.family == 'velocity' and native.terminate_when_unhealthy
        self.objects = {}
        if self.family == 'circle':
            axes = ('x', 'y') if native.n_sigwalls == 4 else ('x',) if native.n_sigwalls else ()
            self.objects['sigwalls'] = {'walls': [
                {'axis': axis, 'coord': sign * native.sigwall_locate}
                for axis in axes for sign in (-1, 1)]}
        constraints = tuple(sorted(raw.get('constraints', ()), key=lambda item: item['name']))
        for item in constraints:
            _check_keys(item, ('name', 'cost_limit', 'source_index'), 'native constraint')
            index = item['source_index']
            if int(index) != index or not 0 <= index < native.n_costs:
                raise ValueError(f'Invalid native cost index: {index}')
            if not math.isfinite(float(item['cost_limit'])):
                raise ValueError('Constraint limits must be finite')
        self.constraint_names = tuple(item['name'] for item in constraints)
        self.cost_limits = tuple(float(item['cost_limit']) for item in constraints)
        self.n_costs = len(constraints)
        self._cost_indices = jnp.asarray([item['source_index'] for item in constraints], dtype=jnp.int32)

    def _configure_world(self, world):
        model = world.mj_model
        if self._apply_solver_overrides:
            model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
            model.opt.iterations = int(os.environ.get('MJX_SOLVER_ITER', '1'))
            model.opt.ls_iterations = int(os.environ.get('MJX_SOLVER_LS', '4'))
            if self.agent_type in ('car', 'doggo', 'half_cheetah'):
                model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        world.mjx_model = mjx.put_model(model, device=jax.local_devices()[0])
        return self._world_transform(world) if self._world_transform is not None else world

    @property
    def obs_dim(self):
        return self.native_task.obs_dim

    def reset(self, key):
        state = self.native_task.reset(key)
        return state._replace(cost=state.cost[self._cost_indices])

    def step(self, state, action):
        action = jnp.asarray(action, dtype=state.data.ctrl.dtype)
        action = self._action_offset + self._action_scale * action
        state = self.native_task.step(state, action)
        return state._replace(cost=state.cost[self._cost_indices])


class _LocomotionTask:


    def __init__(self, raw, *, world_transform=None):
        _check_keys(raw, ('source', 'agent_type', 'task_type', 'episode_length',
                          'frame_skip', 'is_earlystop', 'lin_vel_cmd_range',
                          'ang_vel_cmd_range', 'reward_weights', 'constraints'),
                    'locomotion task')
        self.family = raw['task_type']
        self.agent_type = raw['agent_type']
        if self.family != 'locomotion' or self.agent_type != 'laikago':
            raise ValueError('Locomotion tasks support laikago')
        length = raw['episode_length']
        if isinstance(length, bool) or int(length) != length or length < 1:
            raise ValueError('episode_length must be a positive integer')
        self.num_steps = int(length)
        earlystop = raw.get('is_earlystop', False)
        if not isinstance(earlystop, bool):
            raise ValueError('is_earlystop must be a boolean')
        commands = {}
        for name in ('lin_vel_cmd_range', 'ang_vel_cmd_range'):
            if name in raw:
                bounds = tuple(float(value) for value in raw[name])
                if (len(bounds) != 2 or not all(math.isfinite(value) for value in bounds)
                        or bounds[0] > bounds[1]):
                    raise ValueError(f'{name} must contain two finite ordered bounds')
                commands[name] = bounds
        import mjx_locomotion
        _, native = mjx_locomotion.make_task(
            self.agent_type, frame_skip=raw.get('frame_skip', 10),
            max_episode_steps=self.num_steps, is_earlystop=earlystop,
            world_transform=world_transform, **commands)
        self.native_task = native
        self._state_type = mjx_locomotion.State
        self.world = native.world
        self.mjx_model = native.mjx_model
        self.frame_skip = native.frame_skip
        self.is_variable_episode = native.is_variable_episode
        self.n_rewards = 1
        self._action_scale = jnp.asarray(native.action_bounds)
        weights = tuple(float(value) for value in raw['reward_weights'])
        if len(weights) != native.n_rewards or not all(math.isfinite(value) for value in weights):
            raise ValueError(f'reward_weights must contain {native.n_rewards} finite values')
        self._reward_weights = jnp.asarray(weights)
        constraints = tuple(sorted(raw.get('constraints', ()), key=lambda item: item['name']))
        for item in constraints:
            _check_keys(item, ('name', 'cost_limit', 'source_index'), 'locomotion constraint')
            index = item['source_index']
            if isinstance(index, bool) or int(index) != index or not 0 <= index < native.n_costs:
                raise ValueError(f'Invalid locomotion cost index: {index}')
            if item['name'] != native.cost_names[int(index)]:
                raise ValueError(f'Cost index {index} is named {native.cost_names[int(index)]!r}')
            if not math.isfinite(float(item['cost_limit'])):
                raise ValueError('Constraint limits must be finite')
        self.constraint_names = tuple(item['name'] for item in constraints)
        self.cost_limits = tuple(float(item['cost_limit']) for item in constraints)
        self.n_costs = len(constraints)
        self._cost_indices = jnp.asarray([item['source_index'] for item in constraints], dtype=jnp.int32)

    @property
    def obs_dim(self):
        return self.native_task.obs_dim

    def _adapt_state(self, state):

        info = dict(state.info, _raw_reward=state.reward, _raw_cost=state.cost)
        reward = jnp.sum(state.reward * self._reward_weights, keepdims=True)
        return state._replace(reward=reward.astype(get_dtype()),
                              cost=state.cost[self._cost_indices].astype(get_dtype()), info=info)

    def reset(self, key):
        return self._adapt_state(self.native_task.reset(key))

    def step(self, state, action):
        info = dict(state.info)
        reward, cost = info.pop('_raw_reward'), info.pop('_raw_cost')
        done = info['is_terminated'] if self.is_variable_episode else jnp.zeros_like(info['is_terminated'])
        native_state = self._state_type(
            data=state.data, obs=state.obs.astype(info['term_obs'].dtype),
            reward=reward, cost=cost, done=done, key=state.key, info=info)
        action = jnp.asarray(action, dtype=state.data.ctrl.dtype) * self._action_scale
        return self._adapt_state(self.native_task.step(native_state, action))


def build_task(raw, *, world_transform=None, apply_solver_overrides=True):
    backend.configure_mjx()
    if raw.get('source') == 'mjx_locomotion':
        return _LocomotionTask(raw, world_transform=world_transform)
    if raw.get('source') == 'mjx_safetygym':
        return _NativeTask(raw, world_transform=world_transform,
                           apply_solver_overrides=apply_solver_overrides)
    _check_keys(raw, ('agent_type', 'task_type', 'episode_length', 'placements_extents', 'objects',
                      'constraints', 'task_specific', 'agent'), 'task')
    agent, family = raw['agent_type'], raw['task_type']
    if agent not in ('point', 'car') or family not in ('goal', 'circle'):
        raise ValueError('massive-safeRL supports Point/Car Goal and Circle tasks')
    overrides = raw.get('agent', {})
    _check_keys(overrides, ('n_substeps', 'keepout', 'randomize_heading'), 'agent')
    settings = raw['task_specific']
    _check_keys(settings, ('lidar_num_bins', 'lidar_max_dist', 'reward_factor') if family == 'circle'
                else ('lidar_num_bins', 'lidar_max_dist'), 'task_specific')
    extents = tuple(float(x) for x in raw['placements_extents'])
    if len(extents) != 4 or not all(math.isfinite(x) for x in extents) or extents[0] >= extents[2] or extents[1] >= extents[3]:
        raise ValueError('placements_extents must be a finite nonempty rectangle')
    if int(raw['episode_length']) < 1 or int(settings['lidar_num_bins']) < 1 or float(settings['lidar_max_dist']) <= 0:
        raise ValueError('Episode length and lidar dimensions must be positive')
    defaults = {'goal': {'size': .3, 'keepout': .305, 'spawn_radius': 0.0},
                'hazards': {'size': .2, 'keepout': .18}, 'vases': {'size': .1, 'keepout': .15},
                'circle': {'radius': 1.5}, 'sigwalls': {'walls': []}}
    allowed = ('goal', 'hazards', 'vases') if family == 'goal' else ('circle', 'sigwalls')
    objects = {}
    for item in raw['objects']:
        name = item['name']
        if name not in allowed or name in objects:
            raise ValueError(f'Unsupported or duplicate object: {name}')
        _check_keys(item, ('name', 'num', *defaults[name]), f'{name} object')
        cfg = {**defaults[name], **item}
        if int(cfg['num']) != cfg['num'] or cfg['num'] < 0:
            raise ValueError('Object counts must be nonnegative integers')
        if name in ('goal', 'circle') and cfg['num'] != 1:
            raise ValueError(f'{name} requires exactly one object')
        if name == 'vases' and cfg['size'] != .1:
            raise ValueError('The external vase asset uses size 0.1')
        if name == 'sigwalls':
            if len(cfg['walls']) != cfg['num']:
                raise ValueError('sigwalls.num must match the wall list')
            for wall in cfg['walls']:
                _check_keys(wall, ('axis', 'coord'), 'wall')
                if wall['axis'] not in ('x', 'y') or not math.isfinite(float(wall['coord'])):
                    raise ValueError('Each wall needs an x/y axis and finite coordinate')
        objects[name] = cfg
    if ('goal' if family == 'goal' else 'circle') not in objects:
        raise ValueError(f'{family} object is required')
    constraints = tuple(sorted(raw.get('constraints', ()), key=lambda item: item['name']))
    for item in constraints:
        _check_keys(item, ('name', 'cost_limit', 'type', 'threshold'), 'constraint')
        name = item['name']
        if name not in ('hazards', 'vases', 'out_of_boundary', 'velocity_limit_high', 'velocity_limit_low'):
            raise ValueError(f'Unsupported constraint: {name}')
        if item.get('type', 'binary') != 'binary' or not math.isfinite(float(item['cost_limit'])):
            raise ValueError('Constraints must use binary costs and finite limits')
        required = 'sigwalls' if name == 'out_of_boundary' else name
        if name.startswith('velocity_limit_'):
            if not math.isfinite(float(item['threshold'])):
                raise ValueError('Velocity thresholds must be finite')
        elif required not in objects:
            raise ValueError(f'Constraint {name} requires {required}')
        elif 'threshold' in item:
            raise ValueError(f'Constraint {name} does not use a threshold')
    frame_skip = int(overrides.get('n_substeps', 10))
    if frame_skip < 1:
        raise ValueError('n_substeps must be positive')
    xml = Path(mjx_safetygym.__file__).parent / 'agents' / 'xmls' / f'{agent}.xml'
    world = build_world(str(xml), frame_skip=frame_skip, n_vases=objects.get('vases', {}).get('num', 0))
    model = world.mj_model
    if apply_solver_overrides:
        model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        model.opt.iterations = int(os.environ.get('MJX_SOLVER_ITER', '1'))
        model.opt.ls_iterations = int(os.environ.get('MJX_SOLVER_LS', '4'))
        if os.environ.get('MJX_EULERDAMP', 'off').lower() not in ('on', '1', 'true'):
            model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    decorative = ('pointarrow',) if agent == 'point' else ('back_bumper', 'back_connector', 'front_bumper', 'front_connector')
    for name in decorative:
        gid = world.geom_ids[name]
        model.geom_contype[gid], model.geom_conaffinity[gid] = 0, 0
    for name in ('floor', 'agent', *(('left', 'right', 'rear') if agent == 'car' else ())):
        model.geom_conaffinity[world.geom_ids[name]] = 3
    for index in range(world.n_vases):
        gid = world.geom_ids[f'vase{index}']
        model.geom_contype[gid], model.geom_conaffinity[gid] = 2, 3
    if agent == 'car':
        order = ('ballquat_rear', 'ballangvel_rear', 'accelerometer', 'velocimeter', 'gyro', 'magnetometer')
        sensors = {item[0]: item for item in world.obs_sensors}
        world.obs_sensors = tuple(sensors[name] for name in order)
    world.mjx_model = mjx.put_model(model, device=jax.local_devices()[0])
    if world_transform is not None:
        world = world_transform(world)
    return _Task(world, raw, objects, constraints)
