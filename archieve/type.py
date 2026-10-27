from __future__ import annotations
from dataclasses import MISSING, dataclass, field, fields
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Callable, NamedTuple
import os
import jax
import jax.numpy as jnp
from flax import struct
from mujoco import mjx
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState as RolloutBufferState

_COMPUTE_DTYPE = jnp.bfloat16 if os.environ.get('ARCHIEVE_DTYPE', 'float32') == 'bfloat16' else jnp.float32

EPS: float = 1e-08
Metrics = dict[str, jax.Array]

def get_dtype():
    return _COMPUTE_DTYPE

@struct.dataclass
class OnPolicyState:
    actor_params: Any
    rng: jax.Array
    critic_params: Any
    critic_opt: Any
    obs_rms: RunningMeanStd

    pidr_f_params: Any = None
    pidr_ratio_params: Any = None
    pidr_f_opt: Any = None
    pidr_ratio_opt: Any = None

    pidr_cost_baseline: Any = None

@struct.dataclass
class LagrangeOnPolicyState(OnPolicyState):
    lagrange: Any = None
    lagrange_opt: Any = None

@struct.dataclass
class OnBatchState:
    obs: Any
    actions: Any
    rewards: Any
    costs: Any
    dones: Any
    terminated: Any
    next_obs: Any
    discounted_rewards: Any
    discounted_costs: Any
    logp: Any = None
    steps: Any = None
    policy_mean: Any = None
    policy_log_std: Any = None
    pre_tanh: Any = None
    returns: Any = None
    advantages: Any = None
    values: Any = None
    next_values: Any = None
    bootstrap: Any = None
    cost_signal: Any = None
    cost_limit: Any = None

    @classmethod
    def build_zero(cls, num_envs: int, obs_dim: int, act_dim: int, nc: int) -> 'OnBatchState':
        dtype = get_dtype()
        return cls(obs=jnp.zeros((num_envs, obs_dim), dtype=dtype), actions=jnp.zeros((num_envs, act_dim), dtype=dtype), rewards=jnp.zeros((num_envs,), dtype=dtype), costs=jnp.zeros((num_envs, nc), dtype=dtype), dones=jnp.zeros((num_envs,), dtype=dtype), terminated=jnp.zeros((num_envs,), dtype=dtype), next_obs=jnp.zeros((num_envs, obs_dim), dtype=dtype), discounted_rewards=jnp.zeros((num_envs,), dtype=dtype), discounted_costs=jnp.zeros((num_envs, nc), dtype=dtype), logp=jnp.zeros((num_envs,), dtype=dtype), steps=jnp.zeros((num_envs,), dtype=dtype), policy_mean=jnp.zeros((num_envs, act_dim), dtype=dtype), policy_log_std=jnp.zeros((num_envs, act_dim), dtype=dtype), pre_tanh=jnp.zeros((num_envs, act_dim), dtype=dtype))

    @classmethod
    def from_step(cls, obs: Any, action: Any, next_env: Any, steps: Any, sample_stats: Any) -> 'OnBatchState':
        (logp, act_mean, act_lstd, pre_tanh) = sample_stats
        info = next_env.info
        return cls(obs=obs, actions=action, rewards=next_env.reward, costs=info['step_cost'], dones=next_env.done.astype(get_dtype()), terminated=next_env.terminated.astype(get_dtype()), next_obs=info.get('terminal_obs', next_env.obs), discounted_rewards=info['discounted_rewards'], discounted_costs=info['discounted_costs'], logp=logp, steps=steps.astype(get_dtype()), policy_mean=act_mean, policy_log_std=act_lstd, pre_tanh=pre_tanh)

@dataclass(frozen=True)
class ConstraintConfig:
    name: str
    cost_limit: float = 25.0
    compute_fn: Any = None
    params: Mapping[str, Any] = field(default_factory=dict)
    cost_type: str = 'binary'

    def compute(self, data: Any, info: dict[str, Any]) -> None:
        self.compute_fn(data, info, **self.params)

@struct.dataclass
class EnvState:
    data: mjx.Data
    obs: jax.Array
    reward: jax.Array
    cost_vector: jax.Array
    done: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    info: dict[str, Any]

    @classmethod
    def build_zero(cls, *, data: mjx.Data, obs: jax.Array, num_constraints: int, info: dict[str, Any]) -> 'EnvState':
        dtype = get_dtype()
        zero = jnp.zeros_like(data.time, dtype=dtype)
        return cls(data=data, obs=obs, reward=zero, cost_vector=jnp.zeros(zero.shape + (num_constraints,), dtype=dtype), done=zero, terminated=zero, truncated=zero, info=info)

@struct.dataclass
class RunningMeanStd:
    mean: jax.Array
    var: jax.Array
    count: jax.Array

    @classmethod
    def create(cls, obs_dim: int) -> 'RunningMeanStd':
        return cls(mean=jnp.zeros((obs_dim,), dtype=get_dtype()), var=jnp.ones((obs_dim,), dtype=get_dtype()), count=jnp.array(1e-06, dtype=get_dtype()))

    def update(self, obs_batch: jax.Array) -> 'RunningMeanStd':
        obs_batch = jnp.atleast_2d(obs_batch)
        batch_mean = jnp.mean(obs_batch, axis=0)
        batch_var = jnp.var(obs_batch, axis=0)
        batch_size = jnp.array(obs_batch.shape[0], dtype=get_dtype())
        new_count = self.count + batch_size
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_size / new_count
        new_var = (self.var * self.count + batch_var * batch_size + jnp.square(delta) * self.count * batch_size / new_count) / new_count
        return RunningMeanStd(mean=new_mean, var=jnp.maximum(new_var, 1e-12), count=new_count)

    def normalize(self, obs: jax.Array) -> jax.Array:
        normalized = (obs - self.mean) / jnp.sqrt(self.var + 1e-08)
        return jnp.clip(normalized, -10.0, 10.0)

@dataclass(frozen=True)
class ConstraintSpec:
    constraint_names: tuple[str, ...]
    cost_limits: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.constraint_names) != len(self.cost_limits):
            raise ValueError(f'length mismatch: names={len(self.constraint_names)}, limits={len(self.cost_limits)}')

@dataclass(frozen=True, init=False)
class AlgoTrainConfig:
    algo_name: str = ''
    task_name: str = ''
    seed: int = 0
    seeds: tuple[int, ...] = field(default_factory=tuple)
    backend: str | None = None
    use_logging: bool = True
    use_wandb: bool = False
    num_envs: int = 0
    obs_size: int = 0
    action_size: int = 0
    rollout_steps_per_update: int = 0
    update_batch_size: int = 0
    total_steps: int = 0
    total_updates: int = 0
    log_interval_updates: int = 0
    lagrange_lr: float | None = None
    lagrange_multiplier_init: float | None = None
    constraint_spec: Any = None
    task_config: Mapping[str, Any] | None = None
    raw_cost_limits: tuple[Any, ...] = field(default_factory=tuple)
    derived_cost_limits: tuple[Any, ...] = field(default_factory=tuple)
    cost_limits: Any = field(default_factory=tuple)
    discounted: bool = False
    num_constraints: int = 0
    update_epochs: int = 0
    compile_identity_base: tuple[Any, ...] = field(default_factory=tuple)
    _extras: dict[str, Any] = field(default_factory=dict, repr=False)
    _explicit_keys: set[str] = field(default_factory=set, repr=False)

    def __init__(self, **kwargs: Any) -> None:
        explicit_keys = set(kwargs)
        extras: dict[str, Any] = {}
        dataclass_fields = {field_info.name: field_info for field_info in fields(type(self))}
        for (name, field_info) in dataclass_fields.items():
            if name.startswith('_'):
                continue
            if field_info.default is not MISSING:
                default = deepcopy(field_info.default)
            elif field_info.default_factory is not MISSING:
                default = field_info.default_factory()
            else:
                raise ValueError(f'Field {field_info.name} has no default')
            value = kwargs.pop(name, default)
            object.__setattr__(self, name, value)
        for (key, value) in kwargs.items():
            extras[str(key)] = deepcopy(value)
        object.__setattr__(self, '_extras', extras)
        object.__setattr__(self, '_explicit_keys', explicit_keys | set(extras))

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> 'AlgoTrainConfig':
        normalized: dict[str, Any] = {}
        for (key, value) in values.items():
            if isinstance(value, list) and key in {'raw_cost_limits', 'derived_cost_limits', 'seeds', 'compile_identity_base'}:
                normalized[key] = tuple(value)
            else:
                normalized[key] = value
        return cls(**normalized)

    def to_dict(self, *, exclude: frozenset[str]=frozenset()) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key in self._explicit_keys:
            if key in exclude:
                continue
            if key in self._extras:
                result[key] = deepcopy(self._extras[key])
            elif hasattr(self, key):
                result[key] = deepcopy(getattr(self, key))
        return result

    def copy(self) -> 'AlgoTrainConfig':
        return type(self).from_dict(self.to_dict())

    def with_updates(self, updates: Mapping[str, Any]) -> 'AlgoTrainConfig':
        merged = self.to_dict()
        for (key, value) in updates.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return type(self).from_dict(merged)

    def get(self, key: str, default: Any=None) -> Any:
        if key in {field_info.name for field_info in fields(type(self)) if not field_info.name.startswith('_')}:
            return getattr(self, key)
        return self._extras.get(key, default)

    def __getattr__(self, name: str) -> Any:
        if name in self._extras:
            return self._extras[name]
        raise AttributeError(name)

    def __contains__(self, key: object) -> bool:
        field_names = {field_info.name for field_info in fields(type(self)) if not field_info.name.startswith('_')}
        return isinstance(key, str) and (key in field_names or key in self._extras)

    def __repr__(self) -> str:
        return f'AlgoTrainConfig({self.to_dict()!r})'

@dataclass(frozen=True)
class TaskEnvConfig:
    task_name: str
    task_config: Mapping[str, Any]
    episode_length: int
    num_envs: int
    gamma: float

@struct.dataclass
class RuntimeState:
    policy_state: OnPolicyState
    env_state: EnvState
    rollout_buf_state: RolloutBufferState | None
    seed_idx: jax.Array | None = None

@dataclass(frozen=True)
class CompiledTrainingPlan:
    init_runtime_state: Callable[[jax.Array], RuntimeState] | None = None
    prepare_runtime_state: Callable[[RuntimeState], RuntimeState] | None = None
    run_full: Callable[[RuntimeState], tuple[RuntimeState, Metrics]] | None = None
    compile_fingerprint: str = ''

@dataclass(frozen=True)
class ExecutionResult:
    compile_fingerprint: str
    prepare_time_seconds: float
    compile_time_seconds: float
    training_time_seconds: float
    total_time_seconds: float

@dataclass(frozen=True)
class RunContext:
    experiment_name: str
    out_dir: Path
    progress_csv: Path
    meta_path: Path
    wandb_project: str
    wandb_run: Any | None

class RawPhaseMixtureTightening(NamedTuple):
    surrogate: jnp.ndarray
    beta: jnp.ndarray
    tightened: jnp.ndarray
    coverage_min_count: jnp.ndarray
    coverage_complete: jnp.ndarray

class PIDRWeights(NamedTuple):

    weights: jnp.ndarray
    raw_weights: jnp.ndarray
    ratio_mean: jnp.ndarray
    ratio_std: jnp.ndarray
    ratio_min: jnp.ndarray
    ratio_max: jnp.ndarray
    ratio_ess: jnp.ndarray
    tv: jnp.ndarray
    chi2: jnp.ndarray
    tv_by_phase: jnp.ndarray
    chi2_by_phase: jnp.ndarray
    ess_by_phase: jnp.ndarray
    norm_error: jnp.ndarray
    f_loss: jnp.ndarray
    r_loss: jnp.ndarray

class PIDRUpper(NamedTuple):

    surrogate: jnp.ndarray
    base_beta: jnp.ndarray
    correction_mean: jnp.ndarray
    correction_uncertainty: jnp.ndarray
    epsilon: jnp.ndarray
    upper: jnp.ndarray
