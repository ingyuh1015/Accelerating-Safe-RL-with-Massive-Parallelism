from __future__ import annotations
from datetime import datetime
import logging
import os
import uuid
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
_logger = logging.getLogger(__name__)


def run_training(args, algo_train_config, *, env_builder=None) -> None:
    from massive_saferl.train import mark_start, elapsed, build_run_context, save_run_metadata, spawn_wandb_writers, shutdown_wandb_writers, build_task_env_config, Pipeline, build_compiled_plan, execute_training_run, upload_wandb
    from massive_saferl.utils import backend, build_env, get_algo_class
    from massive_saferl import get_dtype
    import jax

    mark_start()
    selected = backend.bootstrap_backend(args.backend)
    global_seeds = tuple(args.seed_list)
    distributed = bool(getattr(args, 'distributed', False))
    sequential = bool(getattr(args, 'sequential', False))
    if not distributed:
        backend.configure_host_device_count(selected, 1 if sequential else len(global_seeds))
    if distributed and not jax.distributed.is_initialized():
        raise RuntimeError('Initialize distributed execution before building the training configuration.')
    process_index, process_count = jax.process_index(), jax.process_count()
    if process_count > 1 and not distributed:
        raise ValueError('Use --distributed to divide seeds across the initialized JAX processes.')
    if not global_seeds or len(set(global_seeds)) != len(global_seeds):
        raise ValueError('Provide at least one seed, with no duplicates.')
    if len(global_seeds) % process_count:
        raise ValueError('The number of global seeds must be divisible by the JAX process count.')
    local_seed_count = len(global_seeds) // process_count
    seeds = global_seeds[process_index * local_seed_count:(process_index + 1) * local_seed_count]
    if distributed and sequential:
        raise ValueError('--distributed requires parallel seed execution.')
    local_devices = jax.local_devices()
    available_devices = len(local_devices)
    seeds_per_device = getattr(args, 'seeds_per_device', None)
    if not sequential and seeds_per_device is not None:
        if seeds_per_device < 1 or len(seeds) % seeds_per_device:
            raise ValueError('--seeds-per-device must be positive and divide the number of seeds.')
        n_devices = len(seeds) // seeds_per_device
    elif not sequential and available_devices >= len(seeds):
        n_devices = len(seeds)
    elif distributed:
        raise ValueError(f'{len(seeds)} local seeds exceed {available_devices} local devices; set --seeds-per-device explicitly.')
    else:
        n_devices = 1
        sequential = True
    if n_devices > available_devices:
        raise ValueError(f'Requested {n_devices} devices; available: {available_devices}')
    if process_count > 1 and n_devices != available_devices:
        raise ValueError('Distributed execution requires all local devices; provide one or more seeds per device on every host.')

    task_config = build_task_env_config(algo_train_config)
    env_options = dict(task_env_config=task_config, episode_length=task_config.episode_length, batch_size=task_config.num_envs, gamma=task_config.gamma)
    if env_builder is None:
        env = build_env(**env_options)
    else:
        world_multiplier = len(seeds) // n_devices if not sequential else 1
        env = env_builder(**env_options, world_multiplier=world_multiplier)
    config = algo_train_config.with_updates({**getattr(env, 'physics_config', {}), 'obs_size': int(env.observation_size), 'action_size': int(env.action_size), 'backend': backend.get_backend()})
    policy = get_algo_class(str(config.algo_name))(config=config, compute_dtype=get_dtype())
    pipeline = Pipeline(policy=policy, env=env)
    _logger.info('[platform] requested=%s active=%s process=%s/%s local_devices=%s global_devices=%s seeds=%s', args.backend, config.backend, process_index, process_count, available_devices, jax.device_count(), seeds)
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M')}_{uuid.uuid4().hex[:4]}"
    if process_count > 1:
        import numpy as np
        from jax.experimental import multihost_utils
        run_id = multihost_utils.broadcast_one_to_all(np.frombuffer(run_id.encode('ascii'), dtype=np.uint8)).tobytes().decode('ascii')
    preset = getattr(args, 'preset', None)
    deferred_logging = sequential
    parallel_logging = len(seeds) > 1 and not deferred_logging
    execution_metadata = {'distributed': distributed, 'process_index': process_index, 'process_count': process_count, 'local_device_count': available_devices, 'global_device_count': jax.device_count(), 'global_seeds': list(global_seeds), 'local_seeds': list(seeds), 'devices_used_per_process': n_devices, 'seeds_per_device': len(seeds) // n_devices if not sequential else 1, 'sequential': sequential}
    contexts = []
    for i, seed in enumerate(seeds):
        device = local_devices[0 if sequential else i // (len(seeds) // n_devices)]
        seed_metadata = {**execution_metadata, 'local_seed_index': i, 'device_id': int(device.id), 'device_platform': device.platform, 'device_kind': device.device_kind}
        contexts.append(build_run_context(config, seed=seed, run_id=run_id, preset=preset, skip_wandb_init=deferred_logging or parallel_logging, execution_metadata=seed_metadata))
    queues, processes = None, []
    completed = set()
    try:
        if config.use_wandb and parallel_logging:
            queues, processes = spawn_wandb_writers(contexts, seeds, config, run_id, preset=preset)
        plan = build_compiled_plan(pipeline=pipeline, progress_path='', use_logging=config.use_logging and not deferred_logging, log_paths={i: str(ctx.out_dir / 'logs' / 'train.log') for i, ctx in enumerate(contexts)}, wandb_run=contexts[0].wandb_run, wandb_queues=queues)
        results = execute_training_run(plan, seeds=seeds, out_dirs=tuple(ctx.out_dir for ctx in contexts), n_devices=n_devices, sequential=sequential, compile_timing=config.use_logging)
        for i, result in enumerate(results):
            save_run_metadata(contexts[i], result, config)
            completed.add(i)
            if deferred_logging:
                upload_wandb(contexts[i], config, seeds[i])
    finally:
        if queues is not None:
            shutdown_wandb_writers(queues, processes)
        for i, context in enumerate(contexts):
            if i not in completed:
                save_run_metadata(context, None, config)
    if process_count > 1:
        multihost_utils.sync_global_devices('training-complete')
    print(f'[timing] process {process_index}/{process_count} total ({len(seeds)} local seeds, {len(global_seeds)} global seeds): {elapsed():.1f}s', flush=True)
