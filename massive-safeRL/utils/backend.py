from __future__ import annotations
import os
import subprocess
import tempfile
import sys
from pathlib import Path

_MJX_CONFIGURED = False
_GL_BACKEND = None

_GL_PROBE = """
import mujoco
model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><light pos="0 0 3"/>'
    '<geom type="sphere" size="0.1" rgba="0.2 0.6 0.9 1"/>'
    '</worldbody></mujoco>')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
with mujoco.Renderer(model, height=16, width=16) as renderer:
    renderer.update_scene(data)
    image = renderer.render()
    if image.shape != (16, 16, 3) or not image.any():
        raise RuntimeError('OpenGL produced an empty image.')
"""


def _gl_environment(backend: str) -> dict:
    environment = os.environ.copy()
    environment['MUJOCO_GL'] = backend
    if backend in ('egl', 'osmesa'):
        environment['PYOPENGL_PLATFORM'] = backend
    else:
        environment.pop('PYOPENGL_PLATFORM', None)
    return environment


def _probe_gl(backend: str) -> str | None:
    try:
        result = subprocess.run(
            [sys.executable, '-c', _GL_PROBE], env=_gl_environment(backend),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)
    if result.returncode == 0:
        return None
    output = (result.stderr or result.stdout).split('Exception ignored in:', 1)[0].strip().splitlines()
    return output[-1] if output else f'process exited with code {result.returncode}'


def configure_gl(render: bool = False) -> None:
    global _GL_BACKEND
    if not render:
        os.environ['MUJOCO_GL'] = 'disabled'
        os.environ.pop('PYOPENGL_PLATFORM', None)
        return
    requested = os.environ.get('MUJOCO_GL', '').strip().lower()
    aliases = {'glx': 'glfw', 'wgl': 'glfw', 'cgl': 'glfw'}
    requested = aliases.get(requested, requested)
    if requested in ('disabled', 'disable', 'off', 'false', '0'):
        requested = 'auto'
    if requested not in ('', 'auto', 'enable', 'enabled', 'on', 'true', '1', 'egl', 'osmesa', 'glfw'):
        raise ValueError(f'Unsupported MUJOCO_GL backend: {requested}')
    defaults = ['egl', 'osmesa'] if sys.platform == 'linux' else ['glfw']
    if sys.platform == 'linux' and (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        defaults.append('glfw')
    candidates = list(dict.fromkeys(([requested] if requested in ('egl', 'osmesa', 'glfw') else []) + defaults))
    loaded_context = next((sys.modules[name] for name in
                           ('mujoco.gl_context', 'mujoco.rendering.classic.gl_context')
                           if name in sys.modules), None)
    loaded_platform = sys.modules.get('OpenGL.platform')
    platform_type = type(getattr(loaded_platform, 'PLATFORM', None))
    platform_name = aliases.get(platform_type.__module__.rsplit('.', 1)[-1],
                                platform_type.__module__.rsplit('.', 1)[-1])
    locked = loaded_context is not None or loaded_platform is not None
    if loaded_context is not None:
        context_type = getattr(loaded_context, 'GLContext', None)
        loaded = context_type.__module__.rsplit('.', 1)[-1] if context_type is not None else None
        loaded = aliases.get(loaded, loaded)
    else:
        loaded = platform_name
    if locked:
        if loaded not in candidates or (loaded in ('egl', 'osmesa') and
                                        loaded_platform is not None and platform_name != loaded):
            raise RuntimeError('OpenGL was imported before its renderer was selected. Start rendering in a new process.')
        candidates = [loaded]
    errors = []
    for candidate in candidates:
        error = None if candidate == _GL_BACKEND else _probe_gl(candidate)
        if error is None:
            environment = _gl_environment(candidate)
            os.environ['MUJOCO_GL'] = environment['MUJOCO_GL']
            if 'PYOPENGL_PLATFORM' in environment:
                os.environ['PYOPENGL_PLATFORM'] = environment['PYOPENGL_PLATFORM']
            else:
                os.environ.pop('PYOPENGL_PLATFORM', None)
            _GL_BACKEND = candidate
            return
        errors.append(f'{candidate}: {error}')
    action = 'Start rendering in a new process before importing MuJoCo or OpenGL.' if locked else 'Run the GPU or TPU setup command in README.md.'
    raise RuntimeError(f'No working OpenGL renderer. {action} ' + '; '.join(errors))


def configure_mjx() -> None:
    global _MJX_CONFIGURED
    if _MJX_CONFIGURED:
        return
    import jax.numpy as jnp
    import numpy as np
    from mujoco.mjx._src import collision_convex
    original = collision_convex.jp

    class Float32Finfo:
        def __getattr__(self, name):
            return self.finfo if name == 'finfo' else getattr(original, name)

        @staticmethod
        def finfo(dtype):
            return original.finfo(jnp.float32 if dtype is float or np.dtype(dtype) == np.float64 else dtype)

    collision_convex.jp = Float32Finfo()
    _MJX_CONFIGURED = True


def _canonical_backend(value: str) -> str:
    candidate = value.strip().lower()
    if candidate.startswith('tpu'):
        return 'tpu'
    if candidate.startswith(('gpu', 'cuda', 'rocm')):
        return 'gpu'
    return 'cpu'


def init_compilation_cache() -> str:

    cache_root = os.environ.get('XDG_CACHE_HOME', '').strip() or tempfile.gettempdir()
    cache_dir = os.environ.get('JAX_COMPILATION_CACHE_DIR', '') or str(Path(cache_root) / 'massive_saferl' / 'jax_compilation_cache')
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

def bootstrap_backend(backend: str, *, render: bool = False) -> str:
    configure_gl(render)
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
