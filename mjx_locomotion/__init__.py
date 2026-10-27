from pathlib import Path

from .base_locomotion_task import BaseLocomotionTask
from .laikago import LaikagoTask
from .state import State
from .world import World, build_world

_XML_DIR = Path(__file__).resolve().parent / 'xmls'


def make_task(robot='laikago', *, frame_skip=10, max_episode_steps=1000,
              is_earlystop=False, lin_vel_cmd_range=None, ang_vel_cmd_range=None,
              world_transform=None):


    robot = robot.lower()
    if robot.endswith('-v0'):
        robot = robot[:-3]
    if robot != 'laikago':
        raise ValueError(f'Unknown robot {robot!r}; choose laikago')
    if isinstance(frame_skip, bool) or int(frame_skip) != frame_skip or frame_skip < 1:
        raise ValueError('frame_skip must be a positive integer')
    if (isinstance(max_episode_steps, bool) or int(max_episode_steps) != max_episode_steps
            or max_episode_steps < 1):
        raise ValueError('max_episode_steps must be a positive integer')
    world = build_world(str(_XML_DIR / 'laikago.xml'), n_substeps=int(frame_skip))
    if world_transform is not None:
        world = world_transform(world)
        world.timestep = float(world.mj_model.opt.timestep)
    task = LaikagoTask(world, max_episode_steps=int(max_episode_steps),
                         is_earlystop=is_earlystop,
                         lin_vel_cmd_range=lin_vel_cmd_range,
                         ang_vel_cmd_range=ang_vel_cmd_range)
    return world, task


def build_task(robot='laikago', **kwargs):

    return make_task(robot, **kwargs)[1]


def make(robot='laikago', **kwargs):

    return build_task(robot, **kwargs)


def make_locomotion(robot='laikago', **kwargs):

    return make(robot, **kwargs)


def list_envs():

    return ('Laikago-v0',)


__all__ = [
    'make', 'make_task', 'build_task', 'make_locomotion', 'list_envs',
    'BaseLocomotionTask', 'LaikagoTask', 'State',
    'World', 'build_world',
]
