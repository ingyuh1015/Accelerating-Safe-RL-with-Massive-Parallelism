from __future__ import annotations
import json
import pickle
from pathlib import Path

def run_evaluation(args) -> dict:
    from archieve.utils import backend, load_task_config
    from archieve import get_dtype, TaskEnvConfig
    from archieve.algos import GaussianLearningActor
    selected = backend.bootstrap_backend(args.backend)
    backend.configure_host_device_count(selected, 1)
    from archieve.envs import build_env
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
    num_envs = min(args.num_envs, args.episodes)
    env_config = TaskEnvConfig(task_name=task_name, task_config=task_config, episode_length=episode_length, num_envs=num_envs, gamma=float(config['gamma']))
    env = build_env(task_env_config=env_config, episode_length=episode_length, batch_size=num_envs, gamma=env_config.gamma)
    actor = GaussianLearningActor(action_dim=env.action_size, hidden_sizes=tuple(config['hidden_sizes']), activation=config['activation'], weight_init=config['weight_init'], layer_norm=config['layer_norm'], compute_dtype=get_dtype())
    with (run_dir / 'model/treedef.pkl').open('rb') as handle:
        treedef = pickle.load(handle)
    with np.load(run_dir / 'model/params.npz', allow_pickle=False) as archive:
        leaves = [jnp.asarray(archive[f'arr_{i}']) for i in range(len(archive.files))]
    saved = jax.tree_util.tree_unflatten(treedef, leaves)
    actor_params = saved['actor_params']
    obs_rms = saved.get('obs_rms')
    if config['obs_normalize'] and obs_rms is None:
        raise ValueError('The model requires saved observation normalization statistics.')

    @jax.jit
    def evaluate_batch(key):
        reset_key, action_key = jax.random.split(key)
        state = env.reset(reset_key)
        active = jnp.ones((num_envs,), dtype=jnp.bool_)
        rewards = jnp.zeros((num_envs,), dtype=get_dtype())
        costs = jnp.zeros_like(state.cost_vector)
        lengths = jnp.zeros((num_envs,), dtype=jnp.int32)

        def step(carry, _):
            state, key, active, rewards, costs, lengths = carry
            key, action_key = jax.random.split(key)
            obs = obs_rms.normalize(state.obs) if config['obs_normalize'] else state.obs
            if args.stochastic:
                action, _ = actor.sample_action(actor_params, obs, action_key)
            else:
                mean, _ = actor.apply(actor_params, obs)
                action = jnp.tanh(mean)
            state = env.step(state, action)
            completed = active & state.done.astype(jnp.bool_)
            terminal = state.info['terminal_episode_metrics']
            rewards = jnp.where(completed, terminal['sum_reward'], rewards)
            costs = jnp.where(completed[:, None], terminal['sum_r_costs'], costs)
            lengths = jnp.where(completed, terminal['length'].astype(jnp.int32), lengths)
            active = active & ~completed
            return (state, key, active, rewards, costs, lengths), None

        final, _ = jax.lax.scan(step, (state, action_key, active, rewards, costs, lengths), None, length=episode_length)
        return final[2:]

    key = jax.random.PRNGKey(args.seed)
    episodes = []
    while len(episodes) < args.episodes:
        key, batch_key = jax.random.split(key)
        active, rewards, costs, lengths = jax.device_get(evaluate_batch(batch_key))
        if np.any(active):
            raise RuntimeError('Evaluation reached its horizon with unfinished episodes.')
        take = min(num_envs, args.episodes - len(episodes))
        for i in range(take):
            episodes.append({'reward': float(rewards[i]), 'costs': costs[i].tolist(), 'length': int(lengths[i])})
    constraint_names = [c.name for c in env.unwrapped.constraints]
    result = {'task': task_name, 'episodes': args.episodes, 'action_mode': 'stochastic' if args.stochastic else 'deterministic', 'constraint_names': constraint_names, 'mean_reward': float(np.mean([e['reward'] for e in episodes])), 'mean_costs': np.mean([e['costs'] for e in episodes], axis=0).tolist(), 'mean_length': float(np.mean([e['length'] for e in episodes])), 'episode_results': episodes}
    serialized = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + '\n', encoding='utf-8')
    print(serialized, flush=True)
    return result
