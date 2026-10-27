from __future__ import annotations
import hashlib
import json
from functools import partial
from pathlib import Path
from typing import Any, Mapping
import jax
import jax.numpy as jnp
from massive_saferl import AlgoTrainConfig, CompiledTrainingPlan, Metrics, RuntimeState
from massive_saferl.train import Pipeline, CallbackLoggingHook


def build_compile_fingerprint(cfg: AlgoTrainConfig, *, compile_identity_base: tuple[str, ...]=(), env_signature: Mapping[str, Any] | None=None) -> str:
    compile_fingerprint_keys = ('policy_type', 'discounted', 'cost_signal_source', 'rollout_mode', 'advantage_estimator', 'num_constraints', 'num_envs', 'episode_length', 'rollout_horizon', 'update_epochs', 'horizon', 'step_weight', 'raw_cost_limits', 'target_kl', 'total_updates', 'rollout_steps_per_update', 'tightening_mode', 'critic_hidden_sizes', 'physics_backend', 'warp_graph_mode', 'warp_naconmax', 'warp_njmax', 'warp_worlds_per_device')
    cfg_dict = cfg.to_dict()
    compile_key_data: dict[str, Any] = {key: list(cfg_dict[key]) if isinstance(cfg_dict[key], (list, tuple)) else cfg_dict[key] for key in compile_fingerprint_keys if key in cfg_dict}
    compile_key_data.setdefault('tightening_mode', 'off')
    compile_key_data.setdefault('cost_signal_source', 'rollout')
    compile_key_data['compile_identity_base'] = list(compile_identity_base)
    compile_key_data['env_signature'] = {str(key): env_signature[key] for key in sorted(env_signature)} if env_signature is not None else {}
    return hashlib.sha1(json.dumps(compile_key_data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def build_compiled_plan(*, pipeline: Pipeline, progress_path: str, use_logging: bool=True, log_paths: dict[int, str] | None=None, wandb_run: Any | None=None, wandb_queues: dict[int, Any] | None=None) -> CompiledTrainingPlan:
    policy = pipeline.policy
    logging_hook = None
    if use_logging:
        env = pipeline.env
        constraint_names = env.constraint_names or None
        if log_paths is not None:
            logging_hook = CallbackLoggingHook(log_paths=log_paths, wandb_run=wandb_run, wandb_queues=wandb_queues, constraint_names=constraint_names)
        elif progress_path:
            log_path = Path(progress_path).parent / 'logs' / 'train.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logging_hook = CallbackLoggingHook(log_paths={0: str(log_path)}, wandb_run=wandb_run, constraint_names=constraint_names)
    one_update = partial(pipeline.one_update, logging_hook=logging_hook)
    cfg = policy.cfg

    def run_full(runtime_state: RuntimeState) -> tuple[RuntimeState, Metrics]:
        (next_state, all_metrics) = jax.lax.scan(one_update, runtime_state, jnp.arange(cfg.total_updates, dtype=jnp.int32))
        return next_state, all_metrics
    return CompiledTrainingPlan(init_runtime_state=pipeline.init_runtime_state, prepare_runtime_state=pipeline.prepare_runtime_state, run_full=run_full, compile_fingerprint=build_compile_fingerprint(cfg, compile_identity_base=cfg.compile_identity_base, env_signature={'obs_size': policy.cfg.obs_size, 'action_size': policy.cfg.action_size}))
