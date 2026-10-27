from __future__ import annotations
from collections.abc import Iterator
from pathlib import Path
import jax
import jax.numpy as jnp
from massive_saferl import CompiledTrainingPlan, ExecutionResult
from massive_saferl.train import elapsed, save_final_model, save_progress_csv


def execute_training_run(plan: CompiledTrainingPlan, *, seeds: tuple[int, ...], out_dirs: tuple[Path, ...], n_devices: int=1, sequential: bool=False, compile_timing: bool=True) -> Iterator[ExecutionResult]:
    n_seeds = len(seeds)
    if len(set(seeds)) != n_seeds:
        raise ValueError('Training seeds must be distinct.')
    if not n_seeds or len(out_dirs) != n_seeds:
        raise ValueError('Provide one output directory for each training seed.')
    if n_devices < 1 or n_devices > jax.local_device_count():
        raise ValueError(f'Requested {n_devices} devices; available: {jax.local_device_count()}')
    multi_host = jax.process_count() > 1
    if multi_host and (sequential or n_devices != jax.local_device_count()):
        raise ValueError('Multi-host execution requires parallel use of all local devices.')
    parallel = (n_seeds > 1 or multi_host) and not sequential
    if parallel and n_seeds % n_devices:
        raise ValueError(f'{n_seeds} seeds must be divisible by {n_devices} devices.')

    def initialize(key, index):
        state = plan.prepare_runtime_state(plan.init_runtime_state(key))
        return state.replace(seed_idx=index)

    grouped = parallel and n_devices > 1 and n_seeds > n_devices
    if parallel and (n_devices > 1 or multi_host):
        init_fn = jax.pmap(jax.vmap(initialize) if grouped else initialize)
        run_fn = jax.pmap(jax.vmap(plan.run_full) if grouped else plan.run_full)
    elif parallel:
        init_fn = jax.vmap(initialize)
        run_fn = jax.jit(jax.vmap(plan.run_full))
    else:
        init_fn = jax.jit(initialize) if sequential else initialize
        run_fn = jax.jit(plan.run_full)
    groups = [tuple(range(n_seeds))] if parallel else [(i,) for i in range(n_seeds)]
    compiled_fn = None
    for indices in groups:
        started = elapsed()
        keys = jnp.stack([jax.random.PRNGKey(seeds[i]) for i in indices])
        seed_indices = jnp.asarray(indices, dtype=jnp.int32)
        if grouped:
            keys = keys.reshape(n_devices, n_seeds // n_devices, *keys.shape[1:])
            seed_indices = seed_indices.reshape(n_devices, n_seeds // n_devices)
        elif not parallel:
            keys, seed_indices = keys[0], seed_indices[0]
        runtime = init_fn(keys, seed_indices)
        jax.block_until_ready(runtime)
        prepared = elapsed()
        if compiled_fn is None and (compile_timing or sequential):
            compiled_fn = run_fn.lower(runtime).compile()
        compiled = elapsed()
        final, all_metrics = (compiled_fn or run_fn)(runtime)
        jax.block_until_ready((final, all_metrics))
        jax.effects_barrier()
        trained = elapsed()
        policy_state, all_metrics = jax.device_get((final.policy_state, all_metrics))
        if grouped:
            policy_state, all_metrics = jax.tree.map(lambda x: x.reshape(n_seeds, *x.shape[2:]), (policy_state, all_metrics))
        elif not parallel:
            policy_state, all_metrics = jax.tree.map(lambda x: x[None], (policy_state, all_metrics))
        print(f'[timing] prepare: {prepared-started:.1f}s compile: {compiled-prepared:.1f}s train: {trained-compiled:.1f}s', flush=True)
        for offset, index in enumerate(indices):
            state, metrics = jax.tree.map(lambda x: x[offset], (policy_state, all_metrics))
            out_dir = Path(out_dirs[index])
            save_progress_csv(str(out_dir / 'progress.csv'), metrics, train_seconds=trained-compiled)
            save_final_model(out_dir, state)
            yield ExecutionResult(compile_fingerprint=plan.compile_fingerprint, prepare_time_seconds=prepared-started, compile_time_seconds=compiled-prepared, training_time_seconds=trained-compiled, total_time_seconds=elapsed()-started)
