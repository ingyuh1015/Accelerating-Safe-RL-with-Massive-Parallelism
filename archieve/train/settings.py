from __future__ import annotations
from archieve import AlgoTrainConfig, ConstraintSpec, TaskEnvConfig, get_dtype
import math
from typing import Any
import jax.numpy as jnp

def load_bool_value(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ('1', 'true', 'yes', 'on'):
            return True
        if lowered in ('0', 'false', 'no', 'off'):
            return False
    raise ValueError(f'Expected boolean-like value, got: {value!r}')

def _build_tightening_setup(cfg: 'AlgoTrainConfig', *, episode_length: int, gamma: float, discounted: bool, num_constraints: int) -> tuple[str, Any, Any, Any]:
    mode = str(getattr(cfg, 'constraint_tightening_mode', 'off'))
    if mode not in ('off', 'on'):
        raise ValueError(f"constraint_tightening_mode must be 'off' or 'on', got {mode!r}")
    if mode == 'off':
        return ('off', None, None, None)
    if num_constraints < 1:
        raise ValueError("constraint_tightening_mode='on' requires at least one constraint")
    if str(cfg.get('policy_type')) != 'on_policy':
        raise ValueError("constraint_tightening_mode='on' is on-policy only")
    if str(cfg.get('rollout_mode')) != 'staggered':
        raise ValueError("constraint_tightening_mode='on' requires rollout_mode='staggered'")
    bc_user = getattr(cfg, 'tightening_bc', None)
    if bc_user is None:
        bc = float(episode_length) if not discounted or abs(gamma - 1.0) < 1e-09 else (1.0 - gamma ** episode_length) / (1.0 - gamma)
    else:
        bc = float(bc_user)
        if not (math.isfinite(bc) and bc > 0):
            raise ValueError(f'tightening_bc must be positive finite, got {bc}')
    delta_conf = float(getattr(cfg, 'tightening_delta_conf', 0.05))
    if not 0.0 < delta_conf < 1.0:
        raise ValueError(f'tightening_delta_conf must be in (0, 1), got {delta_conf}')
    tightening_max = float(getattr(cfg, 'tightening_max', float('inf')))
    return ('on', bc, delta_conf, tightening_max)

def get_collection_shape(cfg: AlgoTrainConfig, scale: int=1) -> tuple[int, int, int]:
    episode_length = int(cfg.episode_length)
    if episode_length <= 0:
        raise ValueError(f'episode_length must be positive, got {episode_length}')
    rollout_steps_per_update = int(cfg.rollout_steps_per_update) * scale
    if rollout_steps_per_update <= 0:
        raise ValueError(f'rollout_steps_per_update must be positive, got {rollout_steps_per_update}')
    if str(getattr(cfg, 'rollout_mode', 'standard')) == 'staggered':
        scale_axis = str(getattr(cfg, 'staggered_scale_axis', 'env') or 'env')
        num_envs = int(cfg.num_envs) * (scale if scale_axis == 'env' else 1)
        if num_envs <= 0:
            raise ValueError(f'num_envs must be positive, got {num_envs}')
        if rollout_steps_per_update % num_envs != 0:
            raise ValueError(f'staggered rollout requires rollout_steps_per_update ({rollout_steps_per_update}) to be divisible by num_envs ({num_envs})')
        rollout_horizon = rollout_steps_per_update // num_envs
        if rollout_horizon > episode_length or episode_length % rollout_horizon != 0:
            raise ValueError(f'staggered rollout_horizon ({rollout_horizon}) must divide episode_length ({episode_length})')
        return (rollout_steps_per_update, num_envs, rollout_horizon)
    override = getattr(cfg, 'rollout_horizon_override', None)
    rollout_horizon = episode_length if override is None else int(override)
    if rollout_horizon <= 0:
        raise ValueError(f'rollout_horizon_override must be positive, got {rollout_horizon}')
    if rollout_steps_per_update % rollout_horizon != 0:
        raise ValueError(f'standard rollout requires rollout_steps_per_update ({rollout_steps_per_update}) to be divisible by rollout_horizon ({rollout_horizon})')
    return (rollout_steps_per_update, rollout_steps_per_update // rollout_horizon, rollout_horizon)

def build_algo_train_config(args: Any, *, seed: int) -> AlgoTrainConfig:
    from archieve.utils import COST_FIELD_DEFAULTS, build_algo_config, load_task_config
    task_name = str(args.task)
    task_config = load_task_config(task_name)
    constraints = task_config.get('constraints', [])
    redun = max(1, int(getattr(args, 'redun', None) or 1))
    if constraints and redun > 1:
        constraints = [dict(item) for item in constraints for _ in range(redun)]
        task_config = dict(task_config)
        task_config['constraints'] = constraints
    if constraints:
        names = [item['name'] for item in constraints]
        limits = [float(item['cost_limit']) for item in constraints]
        sorted_pairs = sorted(zip(names, limits), key=lambda pair: pair[0])
        sorted_names, sorted_limits = zip(*sorted_pairs)
        constraint_spec = ConstraintSpec(constraint_names=tuple(sorted_names), cost_limits=tuple(float(value) for value in sorted_limits))
    else:
        constraint_spec = None
    preset = getattr(args, 'preset', None)
    component_overrides = {'preset': str(preset)} if preset is not None else None
    algo_name = str(args.algo).lower()
    cfg = build_algo_config(algo_name, component_overrides=component_overrides)
    base_updates: dict[str, Any] = {'algo_name': algo_name, 'task_name': task_name, 'seed': int(seed), 'seeds': (int(seed),), 'use_wandb': load_bool_value(getattr(args, 'wandb', None), default=False), 'use_logging': load_bool_value(getattr(args, 'logging', None), default=True), 'constraint_spec': constraint_spec, 'task_config': task_config}
    arg_overrides: tuple[tuple[str, str, Any], ...] = (('num_envs', 'num_envs', int), ('rollout_horizon_override', 'rollout_horizon_override', int), ('backend', 'backend', str), ('total_steps', 'total_steps', int), ('lagrange_lr', 'lagrange_lr', float), ('lagrange_multiplier_init', 'lagrange_multiplier_init', float), ('update_epochs', 'update_epochs', int), ('update_batch_size', 'update_batch_size', int), ('staggered_scale_axis', 'staggered_scale_axis', str))
    for (arg_name, cfg_key, coerce) in arg_overrides:
        value = getattr(args, arg_name, None)
        if value is not None:
            base_updates[cfg_key] = coerce(value)
    for _key in ('critic_lr', 'critic_activation', 'critic_weight_init', 'critic_norm_coef'):
        _val = getattr(args, _key, None)
        if _val is not None:
            base_updates[_key] = _val
    for _key in ('critic_layer_norm', 'use_critic_norm'):
        _val = getattr(args, _key, None)
        if _val is not None:
            base_updates[_key] = load_bool_value(_val, default=False)
    _critic_hs = getattr(args, 'critic_hidden_sizes', None)
    if _critic_hs:
        base_updates['critic_hidden_sizes'] = tuple(int(tok) for tok in str(_critic_hs).split(',') if tok.strip())
    tighten = str(getattr(args, 'tighten', 'off') or 'off')
    base_updates['constraint_tightening_mode'] = 'on' if tighten == 'on' else 'off'
    base_updates['tighten_terms'] = str(getattr(args, 'tighten_terms', 'both') or 'both')
    base_updates['episode_length'] = int(task_config['episode_length'])
    cfg = cfg.with_updates(base_updates)
    rollout_horizon_override = getattr(cfg, 'rollout_horizon_override', None)
    if rollout_horizon_override is not None:
        rollout_horizon_override = int(rollout_horizon_override)
        if rollout_horizon_override <= 0:
            raise ValueError(f'rollout_horizon_override must be positive, got {rollout_horizon_override}')
        cfg = cfg.with_updates({'rollout_steps_per_update': int(cfg.num_envs) * rollout_horizon_override})
    elif getattr(args, 'num_envs', None) is not None and (not str(getattr(cfg, 'rollout_mode', 'standard')) == 'staggered'):
        cfg = cfg.with_updates({'rollout_steps_per_update': int(cfg.num_envs) * int(cfg.episode_length)})
    scale_value = getattr(args, 'scale', None)
    scale = int(scale_value) if scale_value is not None and int(scale_value) >= 2 else 1
    total_steps = int(getattr(cfg, 'total_steps', 0)) * scale
    log_interval_updates = max(1, int(cfg.log_interval))
    (rollout_steps_per_update, num_envs, rollout_horizon) = get_collection_shape(cfg, scale)
    raw_cost_limits = constraint_spec.cost_limits if constraint_spec is not None else ()
    discounted = getattr(cfg, 'discounted', COST_FIELD_DEFAULTS['discounted'])
    episode_length = int(cfg.episode_length)
    gamma = float(cfg.gamma)
    if discounted and abs(gamma - 1.0) >= 1e-09:
        cost_scale = (1.0 - gamma ** episode_length) / (1.0 - gamma) / episode_length
    else:
        cost_scale = 1.0
    cost_signal_source = str(getattr(cfg, 'cost_signal_source', 'rollout'))
    if cost_signal_source not in ('rollout', 'terminal_recent_raw'):
        raise ValueError(f"cost_signal_source must be 'rollout' or 'terminal_recent_raw', got {cost_signal_source!r}")
    if cost_signal_source == 'terminal_recent_raw' and (
        str(getattr(cfg, 'rollout_mode', 'standard')) != 'standard'
        or discounted or bool(getattr(cfg, 'cost_surrogate', False))
    ):
        raise ValueError("cost_signal_source='terminal_recent_raw' requires rollout_mode='standard', discounted=False, and cost_surrogate=False")
    if not discounted and bool(getattr(cfg, 'cost_surrogate', False)) and abs(gamma - 1.0) >= 1e-09:

        raise ValueError(
            f'discounted=False with cost_surrogate=True and gamma={gamma} compares a '
            'discounted critic value against an undiscounted cost limit; the constraint '
            'would never bind. Set discounted=True, or cost_surrogate=False, or gamma=1.0.')
    derived_cost_limits = tuple((float(limit) * cost_scale for limit in raw_cost_limits))
    num_constraints = len(raw_cost_limits) if getattr(cfg, 'use_cost', False) else 0
    (tightening_mode, tightening_bc, _delta_conf, tightening_max) = _build_tightening_setup(cfg, episode_length=episode_length, gamma=gamma, discounted=discounted, num_constraints=num_constraints)
    user_zeta = getattr(cfg, 'con_zeta', None)
    if user_zeta is None and derived_cost_limits:
        con_zeta_value = float(min(derived_cost_limits))
    elif user_zeta is None:
        con_zeta_value = 0.0
    else:
        con_zeta_value = float(user_zeta)
    common_updates: dict[str, Any] = {'raw_cost_limits': tuple(raw_cost_limits), 'derived_cost_limits': tuple(derived_cost_limits), 'cost_limits': jnp.asarray(derived_cost_limits, dtype=get_dtype()).reshape(-1)[:num_constraints], 'discounted': discounted, 'cost_signal_source': cost_signal_source, 'num_constraints': num_constraints, 'num_envs': num_envs, 'rollout_horizon': rollout_horizon, 'rollout_steps_per_update': rollout_steps_per_update, 'update_batch_size': int(cfg.update_batch_size) * scale, 'total_steps': total_steps, 'total_updates': max(1, total_steps // rollout_steps_per_update), 'log_interval_updates': log_interval_updates, 'tightening_mode': tightening_mode, 'con_zeta': con_zeta_value}
    if tightening_mode == 'on':
        common_updates['tightening_bc'] = tightening_bc
        common_updates['tightening_max'] = tightening_max
        common_updates['tightening_delta_conf'] = _delta_conf
        common_updates['tightening_cmax'] = float(getattr(cfg, 'tightening_cmax', 1.0))
        common_updates['tighten_terms'] = str(getattr(cfg, 'tighten_terms', 'both'))
    policy_type = str(cfg.get('policy_type'))
    compile_identity_base = (algo_name, task_name, policy_type, str(str(getattr(cfg, 'rollout_mode', 'standard')) == 'staggered'), ','.join((str(n) for n in getattr(constraint_spec, 'constraint_names', ()))), ','.join((str(v) for v in raw_cost_limits)))
    return cfg.with_updates({**common_updates, 'compile_identity_base': compile_identity_base})

def build_task_env_config(algo_train_config: AlgoTrainConfig) -> TaskEnvConfig:
    task_config = algo_train_config.task_config
    return TaskEnvConfig(task_name=algo_train_config.task_name, task_config=dict(task_config), episode_length=int(task_config['episode_length']), num_envs=int(algo_train_config.num_envs), gamma=float(algo_train_config.gamma))
