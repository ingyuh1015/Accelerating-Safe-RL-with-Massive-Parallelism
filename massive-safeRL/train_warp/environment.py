from functools import partial

import jax
from mujoco import mjx
from mujoco.mjx.warp.types import GraphMode

from massive_saferl.utils.environment import Environment
from massive_saferl.utils.task import build_task
from massive_saferl.train_warp.contacts import contact_flags
from massive_saferl.train_warp.state import conditional_worlds, select_state


class WarpEnvironment(Environment):
    select_state = staticmethod(select_state)
    reset_conditional = staticmethod(conditional_worlds)


def _configure_world(world, *, worlds, graph_mode, naconmax, njmax):


    reference = mjx.make_data(world.mj_model, impl='jax')
    required_contacts = 2 * int(reference._impl.ncon) * worlds
    required_constraints = 2 * int(reference._impl.nefc)
    if naconmax is not None and naconmax < required_contacts:
        raise ValueError(
            f'warp naconmax={naconmax} is below this model\'s safe contact capacity '
            f'{required_contacts} across {worlds} worlds.')
    if njmax is not None and njmax < required_constraints:
        raise ValueError(
            f'warp njmax={njmax} is below this model\'s safe constraint capacity '
            f'{required_constraints} per world.')
    contacts = max(64 * worlds, required_contacts) if naconmax is None else naconmax
    constraints = max(128, required_constraints) if njmax is None else njmax
    mode = GraphMode.WARP
    world.mjx_model = mjx.put_model(world.mj_model, impl='warp', graph_mode=mode)
    world.init_data = mjx.make_data(world.mj_model, impl='warp', naconmax=contacts,
                                   naccdmax=contacts, njmax=constraints)
    world.contact_query = contact_flags
    world.select_data = select_state
    world.physics_config = {
        'physics_backend': 'warp', 'warp_graph_mode': graph_mode,
        'warp_naconmax': contacts, 'warp_njmax': constraints,
        'warp_worlds_per_device': worlds,
    }
    return world


def build_env(*, task_env_config, episode_length, batch_size, gamma,
              world_multiplier=1, graph_mode='warp', naconmax=None, njmax=None):
    if graph_mode != 'warp':
        raise ValueError(
            'Use Warp graph mode "warp". MuJoCo MJX 3.6 staged mode does not '
            'return live contact buffers, so it cannot preserve contact-based '
            'observations and costs.')
    from massive_saferl.train_warp.run import _check_runtime
    _check_runtime()
    if jax.default_backend() != 'gpu' or not jax.devices('cuda'):
        raise ValueError('Warp physics requires an NVIDIA CUDA GPU.')
    for name, value in (('world_multiplier', world_multiplier), ('naconmax', naconmax), ('njmax', njmax)):
        if value is not None and (isinstance(value, bool) or int(value) != value or value < 1):
            raise ValueError(f'{name} must be a positive integer.')
    transform = partial(_configure_world, worlds=batch_size * world_multiplier,
                        graph_mode=graph_mode, naconmax=naconmax, njmax=njmax)
    task = build_task(dict(task_env_config.task_config), world_transform=transform,
                      apply_solver_overrides=False)
    env = WarpEnvironment(task, episode_length=episode_length, batch_size=batch_size, gamma=gamma)
    env.physics_config = task.world.physics_config
    return env
