from __future__ import annotations
from collections.abc import Mapping
import os
import re
from pathlib import Path
from typing import Any, cast
import importlib
import optax
import yaml
from massive_saferl import AlgoTrainConfig
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_DATA_ROOT = _PACKAGE_ROOT / 'yaml'
_ALGO_CONFIG_DIR = _DATA_ROOT / 'algorithms'
_PRESETS_ROOT = _DATA_ROOT / 'presets'
_TASKS_ROOT = _DATA_ROOT / 'tasks'

COST_FIELD_DEFAULTS: dict[str, Any] = {'discounted': False}
ALGO_MAP: dict[str, str] = {'cpo': 'massive_saferl.algos.cpo.CPO', 'crpo': 'massive_saferl.algos.crpo.CRPO', 'trpolag': 'massive_saferl.algos.trpo_lag.TRPOLagrange'}


def deep_merge(base: dict[str, object], delta: dict[str, object]) -> dict[str, object]:
    result = base.copy()
    for (key, value) in delta.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(cast(dict[str, object], result[key]), cast(dict[str, object], value))
        else:
            result[key] = value
    return result

def _load_yaml_file(path: Path) -> dict[str, object]:
    with path.open('r', encoding='utf-8') as file_handle:
        raw = cast(object, yaml.safe_load(file_handle))
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(f'YAML root must be a mapping: {path}')
    return {str(key): value for key, value in raw.items()}

def _find_algo_yaml(algo_name: str) -> Path:
    direct = _ALGO_CONFIG_DIR / f'{algo_name}.yaml'
    if direct.exists():
        return direct
    lower = algo_name.lower()
    for p in _ALGO_CONFIG_DIR.glob('*.yaml'):
        if p.stem.lower() == lower:
            return p
    raise FileNotFoundError(f"No algorithm YAML found for '{algo_name}' in {_ALGO_CONFIG_DIR}")

def _load_algo_yaml_raw(algo_name: str) -> dict[str, object]:
    algo_path = _find_algo_yaml(algo_name)
    raw = _load_yaml_file(algo_path)
    base_name_ext = raw.pop('_extends', None)
    if base_name_ext is not None:
        base_raw = _load_algo_yaml_raw(str(base_name_ext))
        raw = deep_merge(cast(dict[str, object], base_raw), cast(dict[str, object], raw))
    return raw

def _validate_algo_yaml_layout(algo_name: str, raw: Mapping[str, object]) -> None:
    algo_block_keys = frozenset({'train', 'network', 'estimator', 'algo', 'cost', 'lagrange', 'logging', 'meta'})
    invalid_keys = sorted((key for key in raw if key not in algo_block_keys))
    if invalid_keys:
        raise ValueError(f"Algorithm YAML '{algo_name}' uses unsupported top-level keys {invalid_keys}. Expected only {sorted(algo_block_keys)}.")
    non_mapping_blocks = sorted((key for (key, value) in raw.items() if not isinstance(value, Mapping)))
    if non_mapping_blocks:
        raise ValueError(f"Algorithm YAML '{algo_name}' must use mapping blocks for {non_mapping_blocks}.")

def available_training_algorithms() -> tuple[str, ...]:
    return tuple(sorted((p.stem for p in _ALGO_CONFIG_DIR.glob('*.yaml'))))

def _preset_roots() -> list[Path]:

    roots = [_PRESETS_ROOT]
    extra = os.environ.get('MASSIVE_SAFERL_PRESET_DIRS', '')
    roots.extend(Path(p).expanduser() for p in extra.split(':') if p.strip())
    return [r for r in roots if r.is_dir()]

def available_presets() -> list[str]:
    return sorted({p.stem for r in _preset_roots() for p in r.glob('*.yaml')})

def _find_preset_path(preset_name: str) -> Path:
    roots = _preset_roots()
    matches = [p for r in roots for p in r.glob(f'{preset_name}.yaml')]
    if not matches:
        raise FileNotFoundError(f'Preset not found: {preset_name!r} under {roots}')
    if len(matches) > 1:
        raise FileNotFoundError(f'Preset name is ambiguous ({preset_name!r}): {matches}')
    return matches[0]


def build_algo_config(algo_name: str, *, component_overrides: dict[str, str] | None=None, **overrides: Any) -> AlgoTrainConfig:
    raw = _load_algo_yaml_raw(algo_name)
    _validate_algo_yaml_layout(algo_name, raw)
    preset_name = (component_overrides or {}).get('preset')
    flat: dict[str, object] = {}
    for (key, value) in raw.items():
        if isinstance(value, dict):
            flat.update(cast(dict[str, object], value))
        else:
            flat[key] = value
    if preset_name is not None:
        flat.update(_load_yaml_file(_find_preset_path(preset_name)))
    advantage = flat.pop('advantage', None)
    if isinstance(advantage, dict):
        if 'type' in advantage:
            flat.setdefault('advantage_estimator', advantage['type'])
        if 'reward_lambda' in advantage:
            flat.setdefault('gae_lambda', advantage['reward_lambda'])
        if 'cost_lambda' in advantage:
            flat.setdefault('cost_gae_lambda', advantage['cost_lambda'])
    flat.update(overrides)
    flat.setdefault('algo_name', str(algo_name))
    return AlgoTrainConfig.from_dict(flat)


def load_task_config(task_name: str) -> dict[str, Any]:
    stem = str(task_name).split('-v')[0].lower()
    path = next((p for p in _TASKS_ROOT.glob('*.yaml') if p.stem.lower() == stem), None)
    if path is not None:
        return dict(_load_yaml_file(path))
    navigation = re.fullmatch(r'safety(point|car|doggo)(goal|circle|button|push)([0-2])', stem)
    velocity = re.fullmatch(r'safety(hopper|walker2d|halfcheetah|ant|humanoid|swimmer)velocity([01])', stem)
    if navigation and (navigation[1] == 'doggo' or navigation[2] in ('button', 'push')):
        agent, family, level = navigation[1], navigation[2], int(navigation[3])
        cost_names = {
            'goal': ((), ('hazards',), ('hazards', 'vases')),
            'circle': ((), ('out_of_boundary',), ('out_of_boundary',)),
            'button': ((), ('buttons', 'hazards', 'gremlins'), ('buttons', 'hazards', 'gremlins')),
            'push': ((), ('hazards',), ('hazards', 'pillars')),
        }[family][level]
    elif velocity:
        agent = 'half_cheetah' if velocity[1] == 'halfcheetah' else velocity[1]
        family, level, cost_names = 'velocity', int(velocity[2]), ('velocity',)
    else:
        raise ValueError(f'Unknown task: {task_name}')
    return {
        'source': 'mjx_safetygym', 'agent_type': agent, 'task_type': family, 'level': level,
        'episode_length': 500 if family == 'circle' else 1000,
        'constraints': [{'name': name, 'cost_limit': 25.0, 'source_index': index}
                        for index, name in enumerate(cost_names)],
    }


def get_algo_class(key: str) -> type:
    k = str(key).lower()
    if k not in ALGO_MAP:
        raise ValueError(f'Unknown algorithm key: {key!r}. Available: {sorted(ALGO_MAP)}')
    return _resolve(ALGO_MAP[k])


def build_optimizer(*, lr: float, max_grad_norm: float | None=40.0) -> optax.GradientTransformation:
    optimizer = optax.adam(lr)
    if max_grad_norm is None:
        return optimizer
    return optax.chain(optax.clip_by_global_norm(max_grad_norm), optimizer)


def _resolve(path: str) -> type:
    (module_path, cls_name) = path.rsplit('.', 1)
    return getattr(importlib.import_module(module_path), cls_name)
