from __future__ import annotations

from copy import copy
from datetime import datetime
from pathlib import Path


def run_evaluation(args) -> dict:
    requested = str(getattr(args, 'backend', 'gpu')).lower()
    if requested not in ('auto', 'gpu'):
        raise ValueError('eval_warp supports only --backend gpu (or auto).')
    graph_mode = str(getattr(args, 'warp_graph_mode', 'warp'))
    if graph_mode != 'warp':
        raise ValueError('eval_warp requires graph mode warp to preserve contact and constraint outputs.')
    capacities = {}
    for name in ('naconmax', 'njmax'):
        value = getattr(args, f'warp_{name}', None)
        if value is not None and (isinstance(value, bool) or int(value) != value or value < 1):
            raise ValueError(f'--warp-{name} must be a positive integer.')
        capacities[name] = None if value is None else int(value)

    from massive_saferl.train_warp.compat import configure_wsl_xla
    configure_wsl_xla()
    from massive_saferl.eval.run import run_evaluation as run_shared_evaluation

    args = copy(args)
    args.backend = 'gpu'
    rendering = bool(getattr(args, 'render', False) or getattr(args, 'render_grid', None)
                     or getattr(args, 'render_path', None))
    if not getattr(args, 'output', None):
        if getattr(args, 'render_path', None):
            args.output = str(Path(args.render_path).with_suffix('.json'))
        else:
            directory = Path(args.run_dir).expanduser() / 'eval_warp'
            args.output = str(directory / f'evaluation_{datetime.now():%Y%m%d_%H%M%S_%f}.json')
    if rendering and not getattr(args, 'render_path', None):
        args.render_path = str(Path(args.output).with_suffix('.mp4'))
    def build_environment(**kwargs):
        from massive_saferl.train_warp.environment import build_env
        return build_env(**kwargs, graph_mode=graph_mode, **capacities)

    return run_shared_evaluation(args, env_builder=build_environment)
