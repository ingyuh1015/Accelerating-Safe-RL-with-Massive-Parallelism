from __future__ import annotations
from pathlib import Path

from ..world import build_world

_XML_DIR = Path(__file__).parent.parent / 'agents' / 'xmls'


_TASK_SPECS = {
    ('goal', 0):   'tasks.safe_navigation.goal:GoalLevel0',
    ('goal', 1):   'tasks.safe_navigation.goal:GoalLevel1',
    ('goal', 2):   'tasks.safe_navigation.goal:GoalLevel2',
    ('button', 0): 'tasks.safe_navigation.button:ButtonLevel0',
    ('button', 1): 'tasks.safe_navigation.button:ButtonLevel1',
    ('button', 2): 'tasks.safe_navigation.button:ButtonLevel2',
    ('push', 0):   'tasks.safe_navigation.push:PushLevel0',
    ('push', 1):   'tasks.safe_navigation.push:PushLevel1',
    ('push', 2):   'tasks.safe_navigation.push:PushLevel2',
    ('circle', 0): 'tasks.safe_navigation.circle:CircleLevel0',
    ('circle', 1): 'tasks.safe_navigation.circle:CircleLevel1',
    ('circle', 2): 'tasks.safe_navigation.circle:CircleLevel2',
}

_AGENTS = ('point', 'car', 'doggo')


def _resolve_task_class(task: str, level: int):
    import importlib
    spec = _TASK_SPECS[(task, level)]
    mod_path, cls_name = spec.split(':')
    mod = importlib.import_module(f'mjx_safetygym.{mod_path}')
    return getattr(mod, cls_name)


def make_task(agent: str, task: str, level: int, *, frame_skip: int = 10,
              world_transform=None, **task_kwargs):

    agent = agent.lower()
    task = task.lower()
    assert agent in _AGENTS, f'unknown agent {agent}'
    assert (task, level) in _TASK_SPECS, f'unknown task/level {task}{level}'

    xml = str(_XML_DIR / f'{agent}.xml')
    TaskCls = _resolve_task_class(task, level)


    has_box = bool(getattr(TaskCls, 'has_push_box', False))

    box_size, box_density = (0.125, 0.0005) if (has_box and agent == 'car') else (0.2, 0.001)
    world = build_world(xml, frame_skip=frame_skip,
                        n_vases=int(getattr(TaskCls, 'n_vases', 0)),
                        n_buttons=int(getattr(TaskCls, 'n_buttons', 0)),
                        n_gremlins=int(getattr(TaskCls, 'n_gremlins', 0)),
                        n_pillars=int(getattr(TaskCls, 'n_pillars', 0)),
                        push_box=has_box, push_box_size=box_size,
                        push_box_density=box_density)
    if world_transform is not None:
        world = world_transform(world)
    if has_box:
        task_kwargs.setdefault('push_box_size', box_size)
    task_obj = TaskCls(world, **task_kwargs)
    return world, task_obj


def make(agent: str, task: str, level: int, *, num_envs: int | None = None,
         max_episode_steps: int | None = None, seed: int = 0, frame_skip: int = 10,
         wrappers=(), **task_kwargs):

    _, task_obj = make_task(agent, task, level, frame_skip=frame_skip, **task_kwargs)
    for w in wrappers:
        task_obj = w(task_obj)
    if max_episode_steps is None:
        max_episode_steps = int(getattr(task_obj, 'num_steps', 1000))
    if num_envs is None:
        from ..builder import Builder
        return Builder(task_obj, max_episode_steps=max_episode_steps, seed=seed)
    from ..vector import MJXVectorEnv
    return MJXVectorEnv(task_obj, num_envs=num_envs,
                        max_episode_steps=max_episode_steps, seed=seed)


def parse_id(env_id: str):

    s = env_id
    if s.startswith('Safety'):
        s = s[len('Safety'):]
    level = int(s[-1])
    s = s[:-1]
    for ag in _AGENTS:
        cap = ag.capitalize()
        if s.lower().startswith(ag):
            return ag, s[len(cap):].lower(), level
    raise ValueError(f'cannot parse env id {env_id}')


def make_id(env_id: str, **kwargs):

    agent, task, level = parse_id(env_id)
    return make(agent, task, level, **kwargs)


def list_envs():

    return [(ag, t, lvl) for ag in _AGENTS for (t, lvl) in sorted(_TASK_SPECS.keys())]


_VELOCITY_XML_DIR = Path(__file__).parent.parent / 'tasks' / 'safe_velocity' / 'xmls'


_VELOCITY_SPECS = {
    'hopper':       ('hopper',       'tasks.safe_velocity.hopper_velocity:HopperVelocity'),
    'walker2d':     ('walker2d',     'tasks.safe_velocity.walker2d_velocity:Walker2dVelocity'),
    'half_cheetah': ('half_cheetah', 'tasks.safe_velocity.half_cheetah_velocity:HalfCheetahVelocity'),
    'ant':          ('ant',          'tasks.safe_velocity.ant_velocity:AntVelocity'),
    'humanoid':     ('humanoid',     'tasks.safe_velocity.humanoid_velocity:HumanoidVelocity'),
    'swimmer':      ('swimmer',      'tasks.safe_velocity.swimmer_velocity:SwimmerVelocity'),
}


def _resolve(spec: str):
    import importlib
    mod_path, cls = spec.split(':')
    return getattr(importlib.import_module(f'mjx_safetygym.{mod_path}'), cls)


def make_velocity(robot: str, *, version: int = 0, num_envs: int | None = None,
                  max_episode_steps: int = 1000, seed: int = 0, wrappers=(),
                  world_transform=None):


    robot = robot.lower()
    assert robot in _VELOCITY_SPECS, f'unknown velocity robot {robot}'
    xml_stem, spec = _VELOCITY_SPECS[robot]
    Cls = _resolve(spec)
    world = build_world(str(_VELOCITY_XML_DIR / f'{xml_stem}.xml'), frame_skip=Cls.frame_skip)
    if world_transform is not None:
        world = world_transform(world)
    thr = Cls.__dict__.get('THRESHOLDS', None)
    if thr is None:
        import importlib
        mod = importlib.import_module(Cls.__module__)
        thr = getattr(mod, 'THRESHOLDS', {0: Cls.velocity_threshold})
    task = Cls(world, velocity_threshold=thr.get(version, Cls.velocity_threshold))
    if robot == 'swimmer':
        task.use_planar_speed = version == 0
    for w in wrappers:
        task = w(task)
    if num_envs is None:
        from ..builder import Builder
        return Builder(task, max_episode_steps=max_episode_steps, seed=seed)
    from ..vector import MJXVectorEnv
    return MJXVectorEnv(task, num_envs=num_envs, max_episode_steps=max_episode_steps, seed=seed)


_MULTI_XML_DIR = Path(__file__).parent.parent / 'tasks' / 'safe_multi_agent' / 'xmls'

_MULTI_SPECS = {
    ('multi_goal', 0): ('multi_point', 'tasks.safe_multi_agent.multi_goal:MultiGoalLevel0'),
    ('multi_goal', 1): ('multi_point', 'tasks.safe_multi_agent.multi_goal:MultiGoalLevel1'),
    ('multi_goal', 2): ('multi_point', 'tasks.safe_multi_agent.multi_goal:MultiGoalLevel2'),
    ('velocity', 0):   ('multi_ant',   'tasks.safe_multi_agent.velocity:MultiAntVelocity'),
}


def make_multi(task: str, *, level: int = 0, num_envs: int | None = None,
               max_episode_steps: int = 1000, seed: int = 0, frame_skip: int = 2):

    key = (task.lower(), level)
    assert key in _MULTI_SPECS, f'unknown multi task {task}{level}'
    xml_stem, spec = _MULTI_SPECS[key]
    Cls = _resolve(spec)
    fs = getattr(Cls, 'frame_skip', frame_skip)
    world = build_world(str(_MULTI_XML_DIR / f'{xml_stem}.xml'), frame_skip=fs)
    task_obj = Cls(world)
    if num_envs is None:
        from ..builder import Builder
        return Builder(task_obj, max_episode_steps=max_episode_steps, seed=seed)
    from ..vector import MJXVectorEnv
    return MJXVectorEnv(task_obj, num_envs=num_envs, max_episode_steps=max_episode_steps, seed=seed)


def list_velocity():
    return list(_VELOCITY_SPECS.keys())


def list_multi():
    return list(_MULTI_SPECS.keys())
