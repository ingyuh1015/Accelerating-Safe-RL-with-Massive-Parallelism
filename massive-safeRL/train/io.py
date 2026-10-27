from __future__ import annotations
from datetime import datetime, timezone
import csv
import logging
import pickle
import os
from pathlib import Path
import platform as py_platform
import subprocess
import sys
import time
from typing import Any, TYPE_CHECKING
import uuid
import yaml
import numpy as np

if TYPE_CHECKING:
    from massive_saferl import RunContext

_start: float | None = None

_logger = logging.getLogger(__name__)

def mark_start() -> None:
    global _start
    if _start is None:
        _start = time.monotonic()

def elapsed() -> float:
    return time.monotonic() - (_start if _start is not None else time.monotonic())


class CallbackLoggingHook:

    def __init__(self, *, log_paths: dict[int, str] | None=None, wandb_run: Any | None=None, wandb_queues: dict[int, Any] | None=None, constraint_names: tuple[str, ...] | None=None) -> None:
        self._handles = {int(idx): open(path, 'a', encoding='utf-8') for idx, path in (log_paths or {}).items()}
        self._constraint_names = constraint_names or ()
        self._n_names = len(self._constraint_names)
        self._unique_names = len(set(self._constraint_names)) == self._n_names
        self._wandb_run = wandb_run
        self._wandb_queues = wandb_queues or {}

    def emit(self, raw_scalars: dict[str, Any], debug: Any | None=None, seed_idx: Any | None=None) -> None:
        import jax
        jax.debug.callback(self._emit_host, (raw_scalars, debug, seed_idx), ordered=False)

    def _emit_host(self, args) -> None:
        (raw_scalars, update_index, seed_idx) = args
        sid = int(seed_idx) if seed_idx is not None else 0
        scalars: dict[str, float] = {}
        for (k, v) in raw_scalars.items():
            arr = np.asarray(v)
            if arr.ndim == 0:
                scalars[k] = float(arr)
            elif arr.ndim == 1 and arr.shape[0] > 0:
                use_names = self._n_names > 0 and self._unique_names and arr.shape[0] == self._n_names
                for i in range(arr.shape[0]):
                    suffix = self._constraint_names[i] if use_names else i
                    scalars[f'{k}_{suffix}'] = float(arr[i])
        step_num = int(update_index) + 1 if update_index is not None else 0
        log_lines = self._format_epoch_log(scalars)
        prefix = f'#{step_num} '
        indent = ' ' * len(prefix)
        if log_lines:
            body = ('\n' + indent).join([prefix + log_lines[0]] + log_lines[1:])
            width = max([len(prefix + log_lines[0]), *(len(indent + ln) for ln in log_lines[1:])])
            sep = '-' * width
            block = f'{sep}\n{body}\n{sep}'
        else:
            block = prefix
        if sid == 0:
            print(block, flush=True)
        handle = self._handles.get(sid)
        if handle is not None:
            handle.write(block + '\n')
            handle.flush()
        step = int(update_index) if update_index is not None else None
        q = (self._wandb_queues or {}).get(sid) if self._wandb_queues else None
        if q is not None:
            try:
                q.put_nowait((step, dict(scalars)))
            except Exception:
                pass
        elif self._wandb_run is not None and sid == 0:
            self._wandb_run.log(scalars, step=step)

    @staticmethod
    def _format_epoch_log(metrics: dict[str, Any]) -> list[str]:
        def _fmt(v: Any) -> str:
            if isinstance(v, (int, float)):
                return f'{float(v):.4f}'
            return str(v)

        cost_col_stems = ('Ep/r_cost', 'Ep/d_cost', 'Epd/r_cost', 'Epd/d_cost', 'tightening_beta', 'tightening_epsilon', 'tightening_delta', 'cost_critic_loss', 'cost_value_mean', 'cost_grad_norm')
        reward_col_order = ('Ep/reward', 'Ep/d_reward', 'Epd/reward', 'Epd/d_reward', 'reward_critic_loss', 'reward_value_mean', 'reward_grad_norm', 'actor_grad_norm')
        cost_pairs: dict[str, tuple[float, float]] = {}
        for k in sorted(metrics):
            if k.startswith('cost_signal'):
                suffix = k[len('cost_signal'):]
                cost_pairs[suffix] = (float(metrics.get(k, 0.0)), float(metrics.get(f'cost_limit{suffix}', 0.0)))
        reward_col: list[tuple[str, str]] = []
        for key in reward_col_order:
            if key in metrics:
                reward_col.append((key, _fmt(metrics[key])))
        cost_cols: list[list[tuple[str, str]]] = []
        for suffix in sorted(cost_pairs):
            (sig, lim) = cost_pairs[suffix]
            col: list[tuple[str, str]] = [(f'cost/limit{suffix}', f'{sig:.1f}/{lim:.1f}')]
            for stem in cost_col_stems:
                key = f'{stem}{suffix}'
                if key in metrics:
                    col.append((key, _fmt(metrics[key])))
            cost_cols.append(col)
        columns = [c for c in [reward_col, *cost_cols] if c]
        if not columns:
            return []
        col_kw = [max((len(k) for (k, _) in c)) for c in columns]
        col_vw = [max((len(v) for (_, v) in c)) for c in columns]
        num_rows = max((len(c) for c in columns))
        lines: list[str] = []
        for i in range(num_rows):
            cells: list[str] = []
            for (j, col) in enumerate(columns):
                if i < len(col):
                    (k, v) = col[i]
                    cells.append(f'{k:<{col_kw[j]}} {v:>{col_vw[j]}}')
                else:
                    cells.append(' ' * (col_kw[j] + 1 + col_vw[j]))
            lines.append(' | '.join(cells))
        return lines

def _extra_wandb_tags() -> list[str]:
    raw = os.environ.get('WANDB_TAGS_EXTRA', '')
    return [t.strip() for t in raw.split(',') if t.strip()]

def _to_yaml_safe(obj: Any) -> Any:
    if isinstance(obj, (tuple, list)):
        return [_to_yaml_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_yaml_safe(v) for (k, v) in obj.items()}
    return obj

def _build_saved_config(algo_train_config: Any) -> dict[str, Any]:
    config_exclude_keys = frozenset({'constraint_spec', 'compile_identity_base', 'seeds', 'cost_limits'})
    raw = algo_train_config.to_dict(exclude=config_exclude_keys)
    cs = getattr(algo_train_config, 'constraint_spec', None)
    if cs is not None and hasattr(cs, 'constraint_names'):
        raw['constraint_names'] = list(getattr(cs, 'constraint_names', ()))
        raw['constraint_cost_limits'] = [float(x) for x in getattr(cs, 'cost_limits', ())]
    return {'saved_config_schema_version': 2, **raw}


def build_run_context(algo_train_config: Any, *, seed: int, run_id: str | None=None, preset: str | None=None, skip_wandb_init: bool=False, execution_metadata: dict[str, Any] | None=None) -> RunContext:
    from massive_saferl import RunContext
    if run_id is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M')
        hex4 = uuid.uuid4().hex[:4]
        run_id = f'{ts}_{hex4}'
    algo_name = str(algo_train_config.algo_name)
    task_name = str(algo_train_config.task_name)
    policy_type = str(algo_train_config.policy_type)
    experiment_name = str(Path(task_name) / algo_name / f'{run_id}_s{seed}')
    _constraint_spec = getattr(algo_train_config, 'constraint_spec', None)
    resolved_constraint = ','.join((str(n) for n in getattr(_constraint_spec, 'constraint_names', ()))) or 'none'
    override = os.environ.get('MASSIVE_SAFERL_RESULTS_DIR', '').strip()
    results_root = Path(override).expanduser().resolve() if override else Path('results').resolve()
    out_dir = results_root / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'logs').mkdir(parents=True, exist_ok=True)
    progress_csv = out_dir / 'progress.csv'
    meta_path = out_dir / 'metadata.yaml'
    if not meta_path.exists():
        meta_path = save_metadata(out_dir, algo=algo_name, task=task_name, seed=int(seed), constraint=resolved_constraint, extra_meta={'env_id': task_name, 'policy_type': policy_type, **({'execution': execution_metadata} if execution_metadata is not None else {})})
    config_path = out_dir / 'config.yaml'
    if not config_path.exists():

        saved_config = {**_build_saved_config(algo_train_config), 'seed': int(seed)}
        with config_path.open('w', encoding='utf-8') as f:
            yaml.safe_dump(_to_yaml_safe(saved_config), f, default_flow_style=False, allow_unicode=True)
    wandb_project = _wandb_project(policy_type, task_name)
    wandb_group = os.environ.get('WANDB_RUN_GROUP')
    wandb_run = None
    if bool(getattr(algo_train_config, 'use_wandb', False)) and (not skip_wandb_init):
        wandb_run = init_wandb(project=wandb_project, name=f'{algo_name}_s{seed}_{experiment_name}', mode='online', group=wandb_group, dir=str(out_dir), config={'algo': algo_name, 'task': task_name, 'seed': int(seed), 'preset': preset or '', 'constraint': resolved_constraint, 'experiment_name': experiment_name, 'policy_type': policy_type}, tags=[f'algo:{algo_name}', f'task:{task_name}', f'policy:{policy_type}', *([] if not preset else [f'preset:{preset}']), *_extra_wandb_tags()])
    return RunContext(experiment_name=experiment_name, out_dir=out_dir, progress_csv=progress_csv, meta_path=meta_path, wandb_project=wandb_project, wandb_run=wandb_run)

def save_run_metadata(run_context: RunContext, execution_result: Any, algo_train_config: Any=None) -> None:
    if run_context.wandb_run is not None:
        try:
            run_context.wandb_run.finish()
        except Exception:
            pass
    loaded_data = yaml.safe_load(run_context.meta_path.read_text(encoding='utf-8'))
    if loaded_data is not None and not isinstance(loaded_data, dict):
        raise ValueError('metadata YAML must decode to a mapping')
    meta_data = dict(loaded_data or {})
    meta_data['ended_at'] = datetime.now(timezone.utc).isoformat()
    meta_data['exit_status'] = 0 if execution_result is not None else 1
    if run_context.wandb_run is not None and getattr(run_context.wandb_run, 'id', None) is not None:
        meta_data['wandb_run_id'] = str(run_context.wandb_run.id)
        meta_data['wandb_project'] = run_context.wandb_project
    else:
        meta_data['wandb_run_id'] = ''
        meta_data['wandb_project'] = ''
    if execution_result is not None:
        meta_data['static_compile_fingerprint'] = execution_result.compile_fingerprint
        meta_data['prepare_time_seconds'] = float(execution_result.prepare_time_seconds)
        meta_data['compile_time_seconds'] = float(execution_result.compile_time_seconds)
        meta_data['training_time_seconds'] = float(execution_result.training_time_seconds)
        meta_data['total_time_seconds'] = float(execution_result.total_time_seconds)
    if algo_train_config is not None:
        meta_data['total_updates'] = int(algo_train_config.total_updates)
    run_context.meta_path.write_text(yaml.safe_dump(_to_yaml_safe(meta_data), default_flow_style=False, allow_unicode=True), encoding='utf-8')


def _get_git_info() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
    except Exception:
        commit = 'unknown'
    try:
        status = subprocess.check_output(['git', 'status', '--porcelain'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
        dirty = bool(status)
    except Exception:
        dirty = False
    return (commit, dirty)

def _get_key_packages() -> dict[str, str]:
    packages = ['jax', 'jaxlib', 'mujoco', 'dm-tree', 'optax', 'flax']
    versions = {}
    for pkg in packages:
        try:
            mod = __import__(pkg.replace('-', '_'))
            versions[pkg] = getattr(mod, '__version__', 'unknown')
        except Exception:
            versions[pkg] = 'not installed'
    return versions

def save_metadata(exp_dir: Path, algo: str, task: str, seed: int, constraint: str, extra_meta: dict[str, Any] | None=None) -> Path:
    (git_commit, git_dirty) = _get_git_info()
    meta = {'created': datetime.now(timezone.utc).isoformat(), 'ended_at': None, 'git_commit': git_commit, 'git_dirty': git_dirty, 'command': ' '.join(sys.argv), 'hostname': py_platform.node(), 'python_version': py_platform.python_version(), 'key_packages': _get_key_packages(), 'wandb_run_id': '', 'wandb_project': '', 'notes': '', 'algo': algo, 'task': task, 'seed': seed, 'constraint': constraint, 'exit_status': None}
    if extra_meta:
        meta.update(extra_meta)
    meta_path = Path(exp_dir) / 'metadata.yaml'
    with meta_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(_to_yaml_safe(meta), f, default_flow_style=False, allow_unicode=True)
    return meta_path

def init_wandb(project: str, name: str, config: dict[str, Any], mode: str='online', entity: str | None=None, tags: list[str] | None=None, group: str | None=None, dir: str | None=None) -> Any:
    try:
        from wandb.sdk.wandb_init import init as _wandb_init
        if dir is not None:
            os.makedirs(dir, exist_ok=True)
        return _wandb_init(project=project, name=name, config=config, mode=mode, entity=entity, tags=tags or [], group=group, reinit=True, dir=dir)
    except Exception as e:
        _logger.warning('wandb init failed: %s', e)
        return None

def spawn_wandb_writers(run_contexts: list[Any], seeds: tuple[int, ...], cfg: Any, shared_run_id: str, preset: str | None=None) -> tuple[dict[int, Any], list[Any]]:
    import multiprocessing as mp
    algo_name = str(cfg.algo_name)
    task_name = str(cfg.task_name)
    policy_type = str(cfg.policy_type)
    project = _wandb_project(policy_type, task_name)
    group = os.environ.get('WANDB_RUN_GROUP') or f'{algo_name}_{shared_run_id}'
    extra_tags = _extra_wandb_tags()
    ctx = mp.get_context('spawn')
    queues: dict[int, Any] = {}
    processes: list[Any] = []
    readies: list[Any] = []
    for (i, (rc, seed)) in enumerate(zip(run_contexts, seeds)):
        name = f'{algo_name}_s{seed}'
        config = {'algo': algo_name, 'task': task_name, 'seed': int(seed), 'preset': preset or '', 'policy_type': policy_type}
        tags = [f'algo:{algo_name}', f'task:{task_name}', f'seed:{seed}', *([] if not preset else [f'preset:{preset}']), *extra_tags]
        q = ctx.Queue(maxsize=0)
        ready = ctx.Event()
        p = ctx.Process(target=_wandb_writer_worker, args=(project, name, group, config, tags, q, ready, str(rc.out_dir)), daemon=False)
        p.start()
        queues[i] = q
        processes.append(p)
        readies.append(ready)
    for r in readies:
        r.wait(timeout=120)
    return (queues, processes)

def shutdown_wandb_writers(queues: dict[int, Any], processes: list[Any]) -> None:
    for q in queues.values():
        try:
            q.put(None)
        except Exception:
            pass
    for p in processes:
        try:
            p.join(timeout=60)
        except Exception:
            pass

def _wandb_writer_worker(project: str, name: str, group: str, config: dict[str, Any], tags: list[str], queue: Any, ready: Any, dir: str | None=None) -> None:
    from wandb.sdk.wandb_init import init as wandb_init
    if dir is not None:
        os.makedirs(dir, exist_ok=True)
    run = wandb_init(project=project, name=name, group=group, config=config, tags=tags, dir=dir)
    try:
        ready.set()
        while True:
            msg = queue.get()
            if msg is None:
                break
            (step, scalars) = msg
            run.log(scalars, step=step)
    finally:
        try:
            run.finish()
        except Exception:
            pass

def _wandb_project(policy_type: str, task_name: str) -> str:
    default = f'massive_saferl-{policy_type}-{task_name}' if task_name else f'massive_saferl-{policy_type}'
    return os.environ.get('WANDB_PROJECT') or default


def save_progress_csv(progress_path: str, all_metrics: Any, train_seconds: float | None=None) -> None:
    if not progress_path or not all_metrics:
        return
    fields = {k: np.asarray(v) for (k, v) in all_metrics.items()}
    n = None
    cols: dict[str, np.ndarray] = {}
    for (key, arr) in fields.items():
        if arr.ndim == 1:
            cols[key] = arr
            n = len(arr) if n is None else n
        elif arr.ndim == 2:
            for j in range(arr.shape[1]):
                cols[f'{key}_{j}'] = arr[:, j]
            n = arr.shape[0] if n is None else n
    if not cols or n is None:
        return
    if train_seconds is not None and n > 0:
        cols['time_s'] = np.linspace(train_seconds / n, train_seconds, n, dtype=np.float64)
    out_path = Path(progress_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_column_order = ('time_s', 'Ep/reward', 'Ep/d_reward', 'Ep/r_cost', 'Ep/d_cost', 'Epd/reward', 'Epd/d_reward', 'Epd/r_cost', 'Epd/d_cost', 'cost_signal', 'cost_limit', 'actor_loss', 'reward_critic_loss', 'cost_critic_loss', 'reward_value_mean', 'cost_value_mean', 'reward_grad_norm', 'cost_grad_norm', 'actor_grad_norm', 'entropy', 'ratio_mean', 'lagrange', 'lagrange_delta')
    ordered: list[str] = []
    remaining = set(cols)
    for prefix in csv_column_order:
        for k in sorted(remaining):
            if k == prefix or k.startswith(prefix + '_'):
                ordered.append(k)
        remaining -= set(ordered)
    ordered.extend(sorted(remaining))
    ordered_keys = ordered
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=ordered_keys)
        writer.writeheader()
        for i in range(n):
            writer.writerow({k: float(v[i]) for (k, v) in cols.items()})
    print(f'[progress] Saved {n} updates to {out_path}', flush=True)

def save_final_model(out_dir: Path, policy_state: Any) -> None:
    import jax
    state = {'actor_params': policy_state.actor_params, 'critic_params': policy_state.critic_params}
    if policy_state.obs_rms is not None:
        state['obs_rms'] = policy_state.obs_rms
    path = Path(out_dir) / 'model'
    path.mkdir(parents=True, exist_ok=True)
    leaves, treedef = jax.tree_util.tree_flatten(state)
    np.savez(path / 'params.npz', *(np.asarray(leaf) for leaf in leaves))
    with (path / 'treedef.pkl').open('wb') as handle:
        pickle.dump(treedef, handle)


def upload_wandb(run_context, cfg, seed) -> None:
    if not bool(getattr(cfg, "use_wandb", False)):
        return
    try:
        import wandb
    except Exception as e:
        print(f"[wandb] import failed, skip upload: {e}", flush=True)
        return
    name = os.environ.get("WANDB_NAME") or f"{cfg.algo_name}_s{seed}_{cfg.task_name}"
    project = os.environ.get("WANDB_PROJECT") or "massive_saferl"
    group = os.environ.get("WANDB_RUN_GROUP")
    csv_path = str(run_context.progress_csv)
    if not os.path.exists(csv_path):
        print(f"[wandb] no progress.csv for seed {seed}, skip", flush=True)
        return
    run = wandb.init(project=project, name=name, group=group,
                     dir=str(run_context.out_dir), reinit=True,
                     config={"seed": int(seed)})
    with open(csv_path, encoding="utf-8") as f:
        for step, row in enumerate(csv.DictReader(f)):
            rec = {}
            for k, v in row.items():
                try:
                    rec[k] = float(v)
                except (TypeError, ValueError):
                    continue
            if rec:
                run.log(rec, step=step)
    run.finish()
    print(f"[wandb] seed {seed} -> run '{name}' uploaded ({project})", flush=True)
