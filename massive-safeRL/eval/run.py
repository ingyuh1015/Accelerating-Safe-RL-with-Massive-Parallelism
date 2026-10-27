from __future__ import annotations
import json
import pickle
import math
import sys
from time import perf_counter
from uuid import uuid4
from contextlib import nullcontext
from datetime import datetime
from functools import partial
from pathlib import Path


def run_evaluation(args, *, env_builder=None) -> dict:
    started = perf_counter()
    compile_seconds = rollout_seconds = render_seconds = 0.0
    from massive_saferl.utils import backend, build_env, load_task_config
    from massive_saferl import get_dtype, TaskEnvConfig
    from massive_saferl.eval import GridVideoRecorder
    from massive_saferl.algos import GaussianLearningActor
    rendering = bool(getattr(args, 'render', False) or getattr(args, 'render_grid', None) or getattr(args, 'render_path', None))
    grid = getattr(args, 'render_grid', None) or (1, 1)
    requested_envs = getattr(args, 'num_envs', None)
    if rendering and requested_envs is not None and requested_envs != grid[0] * grid[1]:
        raise ValueError('--num-envs must match --render-grid; omit --num-envs to use the grid size.')
    num_envs = grid[0] * grid[1] if rendering else (8 if requested_envs is None else requested_envs)
    episode_count = getattr(args, 'episodes', None)
    episode_count = (num_envs if rendering else 10) if episode_count is None else episode_count
    if num_envs < 1 or episode_count < 1:
        raise ValueError('--num-envs and --episodes must be positive')
    if rendering and episode_count % num_envs:
        raise ValueError('--episodes must be a multiple of the render grid size; omit it for one episode per cell.')
    num_envs = min(num_envs, episode_count)
    selected = backend.bootstrap_backend(args.backend, render=rendering)
    backend.configure_host_device_count(selected, 1)
    import jax
    import jax.numpy as jnp
    import numpy as np
    import yaml
    run_dir = Path(args.run_dir).expanduser().resolve()
    config = yaml.safe_load((run_dir / 'config.yaml').read_text())
    task_name = str(config['task_name'])
    task_config = config.get('task_config') or load_task_config(task_name)
    if 'task_config' not in config and config.get('constraint_names'):
        by_name = {c['name']: c for c in task_config['constraints']}
        task_config = {**task_config, 'constraints': [dict(by_name[n]) for n in config['constraint_names']]}
    episode_length = int(config['episode_length'])
    env_config = TaskEnvConfig(task_name=task_name, task_config=task_config, episode_length=episode_length, num_envs=num_envs, gamma=float(config['gamma']))
    env = (build_env if env_builder is None else env_builder)(task_env_config=env_config, episode_length=episode_length, batch_size=num_envs, gamma=env_config.gamma)
    actor = GaussianLearningActor(action_dim=env.action_size, hidden_sizes=tuple(config['hidden_sizes']), activation=config['activation'], weight_init=config['weight_init'], layer_norm=config['layer_norm'], compute_dtype=get_dtype())
    with (run_dir / 'model/treedef.pkl').open('rb') as handle:
        treedef = pickle.load(handle)
    with np.load(run_dir / 'model/params.npz', allow_pickle=False) as archive:
        leaves = [jnp.asarray(archive[f'arr_{i}']) for i in range(len(archive.files))]
    saved = jax.tree_util.tree_unflatten(treedef, leaves)
    if not all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves):
        raise ValueError('The saved model contains nonfinite parameters or statistics.')
    actor_params = saved['actor_params']
    obs_rms = saved.get('obs_rms')
    if config['obs_normalize'] and obs_rms is None:
        raise ValueError('The model requires saved observation normalization statistics.')

    def capture(state):
        info = state.info['task_info']
        metrics = state.info['episode_metrics']
        frame = {
            'qpos': state.data.qpos, 'qvel': state.data.qvel,
            'mocap_pos': state.data.mocap_pos, 'mocap_quat': state.data.mocap_quat,
            'reward': metrics['sum_reward'], 'costs': metrics['sum_r_costs'],
            'step_costs': state.cost_vector, 'step': metrics['length'], 'done': state.done,
        }
        frame.update({name: info[name] for name in ('goal_xy', 'hazards_xy') if name in info})
        return frame

    @jax.jit
    def initialize_batch(key):
        reset_key, action_key = jax.random.split(key)
        state = env.reset(reset_key)
        return (state, action_key, jnp.ones((num_envs,), dtype=jnp.bool_),
                jnp.zeros((num_envs,), dtype=get_dtype()), jnp.zeros_like(state.cost_vector),
                jnp.zeros((num_envs,), dtype=jnp.int32), capture(state) if rendering else None)

    def step(carry, _):
        state, key, active, rewards, costs, lengths, previous_frame = carry
        key, action_key = jax.random.split(key)
        obs = obs_rms.normalize(state.obs) if config['obs_normalize'] else state.obs
        if args.stochastic:
            action, _ = actor.sample_action(actor_params, obs, action_key)
        else:
            mean, _ = actor.apply(actor_params, obs)
            action = jnp.tanh(mean)
        if rendering:
            state, transition = env.step(state, action, return_transition=True)
        else:
            state = env.step(state, action)
        completed = active & state.done.astype(jnp.bool_)
        terminal = state.info['terminal_episode_metrics']
        rewards = jnp.where(completed, terminal['sum_reward'], rewards)
        costs = jnp.where(completed[:, None], terminal['sum_r_costs'], costs)
        lengths = jnp.where(completed, terminal['length'].astype(jnp.int32), lengths)
        frame = None
        if rendering:
            def keep_last(current, previous):
                mask = active.reshape(active.shape + (1,) * (current.ndim - 1))
                return jnp.where(mask, current, previous)
            frame = jax.tree.map(keep_last, capture(transition), previous_frame)
        active = active & ~completed
        return (state, key, active, rewards, costs, lengths, frame), frame

    @partial(jax.jit, static_argnames=('steps',))
    def evaluate_chunk(carry, *, steps):
        return jax.lax.scan(step, carry, None, length=steps)

    physics_config = getattr(env, 'physics_config', {'physics_backend': 'jax'})
    physics_backend = physics_config['physics_backend']

    def progress(message):
        print(f'[eval_{physics_backend}] {message}', file=sys.stderr, flush=True)

    progress(f'Loaded {task_name} checkpoint; {episode_count} episodes, {num_envs} environments.')
    recorder = None
    if rendering:
        render_started = perf_counter()
        dt = float(env.mj_model.opt.timestep * env.task.frame_skip)
        fps = getattr(args, 'render_fps', None) or min(30., 1. / dt)
        path = getattr(args, 'render_path', None) or run_dir / 'eval' / f'render_{datetime.now():%Y%m%d_%H%M%S}.mp4'
        if args.output and Path(args.output).expanduser().resolve() == Path(path).expanduser().resolve():
            raise ValueError('JSON and video outputs must use different paths.')
        recorder = GridVideoRecorder(env, path, grid=grid,
                                     width=getattr(args, 'render_width', None),
                                     height=getattr(args, 'render_height', None), fps=fps)
        frame_steps = np.minimum(np.ceil((np.arange(math.ceil(episode_length * dt * fps - 1e-9)) + 1) / (fps * dt) - 1e-9).astype(int) - 1,
                                 episode_length - 1)
        render_seconds += perf_counter() - render_started
    key = jax.random.PRNGKey(args.seed)
    episodes = []
    chunk_size = min(32, episode_length) if rendering and physics_backend != 'warp' else episode_length
    evaluate = None
    with recorder if rendering else nullcontext():
        progress('Compiling episode reset and policy rollout.')
        compile_started = perf_counter()
        initialize = initialize_batch.lower(key).compile()
        compile_seconds += perf_counter() - compile_started
        while len(episodes) < episode_count:
            key, batch_key = jax.random.split(key)
            rollout_started = perf_counter()
            carry = initialize(batch_key)
            jax.block_until_ready(carry)
            rollout_seconds += perf_counter() - rollout_started
            if evaluate is None:
                compile_started = perf_counter()
                evaluate = evaluate_chunk.lower(carry, steps=chunk_size).compile()
                compile_seconds += perf_counter() - compile_started
                progress(f'Compilation finished in {compile_seconds:.2f}s; evaluating.')
            for start in range(0, episode_length, chunk_size):
                count = min(chunk_size, episode_length - start)
                rollout_started = perf_counter()
                carry, frames = evaluate(carry)
                jax.block_until_ready((carry, frames))
                rollout_seconds += perf_counter() - rollout_started
                if rendering:
                    render_started = perf_counter()
                    finished = np.flatnonzero(np.all(jax.device_get(frames['done'][:count]), axis=1))
                    samples = frame_steps
                    if finished.size:
                        total_steps = start + int(finished[0]) + 1
                        frame_count = math.ceil(total_steps * dt * fps - 1e-9)
                        samples = np.minimum(frame_steps[:frame_count], total_steps - 1)
                    indices = samples[(samples >= start) & (samples < start + count)] - start
                    if indices.size:
                        frames = jax.device_get(jax.tree.map(lambda value: value[indices], frames))
                        for index in range(len(indices)):
                            recorder.write({name: value[index] for name, value in frames.items()})
                    render_seconds += perf_counter() - render_started
                    if finished.size:
                        break
            active, rewards, costs, lengths = jax.device_get(carry[2:6])
            if np.any(active):
                raise RuntimeError('Evaluation reached its horizon with unfinished episodes.')
            take = min(num_envs, episode_count - len(episodes))
            for i in range(take):
                episodes.append({'reward': float(rewards[i]), 'costs': costs[i].tolist(), 'length': int(lengths[i])})
            progress(f'Episodes {len(episodes)}/{episode_count}' +
                     (f'; video frames {recorder.frame_count}' if rendering else ''))
        if rendering:
            render_started = perf_counter()
            recorder.close()
            render_seconds += perf_counter() - render_started
    constraint_names = list(env.constraint_names)
    result = {'task': task_name, 'episodes': episode_count, 'action_mode': 'stochastic' if args.stochastic else 'deterministic', 'constraint_names': constraint_names, 'mean_reward': float(np.mean([e['reward'] for e in episodes])), 'mean_costs': np.mean([e['costs'] for e in episodes], axis=0).tolist(), 'mean_length': float(np.mean([e['length'] for e in episodes])), 'episode_results': episodes}
    result.update({
        'run_dir': str(run_dir), 'seed': int(args.seed),
        'physics_backend': physics_backend, 'physics_config': physics_config,
        'cost_limits': [float(value) for value in env.cost_limits],
        'timing': {'compile_seconds': compile_seconds, 'rollout_seconds': rollout_seconds,
                   'render_seconds': render_seconds, 'total_seconds': perf_counter() - started},
    })
    if rendering:
        result['video'] = {'path': str(recorder.path.resolve()), 'grid': list(grid), 'frames': recorder.frame_count, 'fps': fps, 'num_envs': num_envs, 'renderer': 'mujoco_opengl'}
    serialized = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f'.{output.name}.{uuid4().hex}.tmp')
        try:
            temporary.write_text(serialized + '\n', encoding='utf-8')
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        progress(f'Saved {output.resolve()}')
    print(serialized, flush=True)
    return result
