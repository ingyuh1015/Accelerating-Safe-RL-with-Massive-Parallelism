from __future__ import annotations
import os
import tempfile
import sys
from pathlib import Path

def _canonical_backend(value: str) -> str:
    candidate = value.strip().lower()
    if candidate.startswith('tpu'):
        return 'tpu'
    if candidate.startswith(('gpu', 'cuda', 'rocm')):
        return 'gpu'
    return 'cpu'

def init_compilation_cache() -> str:

    cache_root = os.environ.get('XDG_CACHE_HOME', '').strip() or tempfile.gettempdir()
    cache_dir = os.environ.get('JAX_COMPILATION_CACHE_DIR', '') or str(Path(cache_root) / 'archieve' / 'jax_compilation_cache')
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ['JAX_COMPILATION_CACHE_DIR'] = cache_dir
    os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES', '-1')
    os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS', '0')
    import jax
    jax.config.update('jax_compilation_cache_dir', cache_dir)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.0)
    return cache_dir

def init_backend(backend: str='auto') -> str:
    valid_backends = {'auto', 'cpu', 'gpu', 'tpu'}
    requested = backend.strip().lower()
    if requested not in valid_backends:
        raise ValueError(f"Unsupported backend '{backend}'. Expected one of {valid_backends}.")
    if requested == 'auto':
        platforms = os.environ.get('JAX_PLATFORMS', '').strip()
        detected = _canonical_backend(platforms.split(',', maxsplit=1)[0]) if platforms else 'auto'
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'true' if detected == 'tpu' else 'false')
        return detected
    jax_platform = 'cuda' if requested == 'gpu' else requested
    if jax_platform == 'tpu':
        os.environ['JAX_PLATFORMS'] = 'tpu,cpu'
    elif jax_platform == 'cuda':
        os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
    else:
        os.environ['JAX_PLATFORMS'] = jax_platform
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'true' if requested == 'tpu' else 'false')
    module = sys.modules.get('jax')
    if module is not None:
        module.config.update('jax_platforms', os.environ['JAX_PLATFORMS'])
    return requested

def get_backend() -> str:
    import jax
    return _canonical_backend(jax.default_backend())

def bootstrap_backend(backend: str) -> str:
    active_backend = str(init_backend(backend))
    init_xla_flags(active_backend)

    init_compilation_cache()
    return active_backend

def initialize_distributed() -> None:
    import jax
    if not jax.distributed.is_initialized():
        jax.distributed.initialize()

def configure_host_device_count(backend: str, num_devices: int) -> None:
    if backend not in ('auto', 'cpu') or num_devices <= 1:
        return
    existing_flags = [flag for flag in os.environ.get('XLA_FLAGS', '').split() if not flag.startswith('--xla_force_host_platform_device_count=')]
    host_device_flag = f'--xla_force_host_platform_device_count={num_devices}'
    os.environ['XLA_FLAGS'] = ' '.join([*existing_flags, host_device_flag])

def init_xla_flags(profile: str) -> None:
    valid_matmul_precisions = {'default', 'high', 'highest'}

    platform_xla_flags = {'gpu': '--xla_gpu_enable_triton_gemm=false --xla_gpu_enable_latency_hiding_scheduler=true', 'cpu': '--xla_cpu_multi_thread_eigen=true'}
    selected = profile.strip().lower()
    defaults = ' '.join(platform_xla_flags.values()) if selected == 'auto' else platform_xla_flags.get(selected, '')
    existing = os.environ.get('XLA_FLAGS', '').strip()
    existing_names = {flag.split('=', 1)[0] for flag in existing.split()}
    missing = [flag for flag in defaults.split() if flag.split('=', 1)[0] not in existing_names]
    os.environ['XLA_FLAGS'] = ' '.join([existing, *missing]).strip()
    raw_precision = (os.environ.get('MJX_SAFETY_MATMUL_PRECISION') or '').strip().lower()
    if raw_precision and raw_precision in valid_matmul_precisions:
        os.environ['JAX_DEFAULT_MATMUL_PRECISION'] = raw_precision
        module = sys.modules.get('jax')
        if module is not None:
            module.config.update('jax_default_matmul_precision', raw_precision)
