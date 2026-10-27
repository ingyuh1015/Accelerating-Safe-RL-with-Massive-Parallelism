from __future__ import annotations

from copy import copy
from functools import partial
from importlib.metadata import PackageNotFoundError, version


def _check_runtime() -> None:
    for package, expected in (('mujoco-mjx', '3.6.0'), ('warp-lang', '1.11.1')):
        try:
            installed = version(package)
        except PackageNotFoundError as exc:
            raise ImportError(
                'Warp dependencies are missing. Install the GPU configuration described in README.md.') from exc
        if installed != expected:
            raise RuntimeError(
                f'Warp execution requires {package}=={expected}; found {installed}. '
                'Install the GPU configuration described in README.md.')

    from massive_saferl.train_warp.compat import configure_wsl_xla
    configure_wsl_xla()
    import jax
    devices = jax.local_devices()
    if not devices or any(device.platform != 'gpu' or 'cuda' not in
                          device.client.platform_version.lower() for device in devices):
        raise RuntimeError('Warp execution requires JAX on NVIDIA CUDA GPUs.')
    driver_hint = (
        'Check NVIDIA driver discovery and CUDA_VISIBLE_DEVICES. On WSL, start a '
        'new process with LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH '
        'before the Warp command.')
    try:
        import warp
        warp.init()
        warp_devices = warp.get_cuda_devices()
    except Exception as exc:
        raise RuntimeError(f'Warp CUDA initialization failed. {driver_hint}') from exc
    requested = {int(getattr(device, 'local_hardware_id', device.id)) for device in devices}
    available = {int(device.ordinal) for device in warp_devices}
    if not requested.issubset(available):
        addresses = lambda ordinals: ', '.join(f'cuda:{i}' for i in sorted(ordinals)) or 'none'
        raise RuntimeError(
            f'JAX uses {len(requested)} CUDA device(s) [{addresses(requested)}], '
            f'but Warp exposes {len(available)} [{addresses(available)}]. {driver_hint}')
    if any(int(device.id) != int(device.local_hardware_id) for device in devices):
        from massive_saferl.train_warp.compat import install_local_device_mapping
        install_local_device_mapping(warp, devices, process_index=jax.process_index())


def run_training(args, algo_train_config) -> None:
    requested = str(getattr(args, 'backend', 'gpu')).lower()
    if requested not in ('auto', 'gpu'):
        raise ValueError('train_warp supports only --backend gpu (or auto).')
    graph_mode = str(getattr(args, 'warp_graph_mode', 'warp'))
    if graph_mode != 'warp':
        raise ValueError('train_warp requires graph mode warp to preserve contact and constraint outputs.')
    capacities = {}
    for name in ('naconmax', 'njmax'):
        value = getattr(args, f'warp_{name}', None)
        if value is not None and (isinstance(value, bool) or int(value) != value or value < 1):
            raise ValueError(f'--warp-{name} must be a positive integer.')
        capacities[name] = None if value is None else int(value)

    from massive_saferl.utils import backend
    backend.bootstrap_backend('gpu')
    _check_runtime()
    from massive_saferl.train.run import run_training as run_shared_training
    from massive_saferl.train_warp.environment import build_env

    args = copy(args)
    args.backend = 'gpu'
    config = algo_train_config.with_updates({
        'backend': 'gpu', 'physics_backend': 'warp', 'warp_graph_mode': graph_mode,
        **{f'warp_{name}': value for name, value in capacities.items()},
    })
    run_shared_training(args, config, env_builder=partial(
        build_env, graph_mode=graph_mode, **capacities))
