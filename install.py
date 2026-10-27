from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
GRAPHICS_LIBRARIES = {
    'EGL': 'libegl1',
    'GL': 'libgl1',
    'OSMesa': 'libosmesa6',
    'glfw': 'libglfw3',
}


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def check_graphics() -> tuple[dict[str, str | None] | None, str]:
    code = '''
import importlib.util
import json
import os
import sys
spec = importlib.util.spec_from_file_location('massive_backend', sys.argv[1])
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)
backend.configure_gl(render=True)
print(json.dumps({name: os.environ.get(name) for name in ('MUJOCO_GL', 'PYOPENGL_PLATFORM')}))
'''
    result = subprocess.run(
        [sys.executable, '-c', code, str(ROOT / 'massive-safeRL/utils/backend.py')],
        capture_output=True, text=True, timeout=150,
    )
    if result.returncode:
        return None, result.stderr.strip() or result.stdout.strip()
    return json.loads(result.stdout.splitlines()[-1]), ''


def install_graphics() -> bool:
    missing = []
    for library, package in GRAPHICS_LIBRARIES.items():
        path = ctypes.util.find_library(library)
        try:
            if path is None:
                raise OSError(library)
            ctypes.CDLL(path)
        except OSError:
            missing.append(package)
    if not missing:
        return False
    command = ['apt-get', 'install', '-y', '--no-install-recommends', '--no-upgrade', *missing]
    if not shutil.which('apt-get'):
        raise RuntimeError('Install the EGL, OpenGL, OSMesa, and GLFW shared libraries with your system package manager, then run this command again.')
    if os.geteuid() == 0:
        prefix = []
    elif shutil.which('sudo'):
        prefix = ['sudo']
    else:
        raise RuntimeError('Run as administrator: ' + ' '.join(command))
    run([*prefix, 'apt-get', 'update'])
    run([*prefix, *command])
    return True


def check_video(graphics: dict[str, str | None]) -> None:
    code = '''
import tempfile
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 3"/><geom type="sphere" size="0.2" rgba="0.8 0.2 0.1 1"/></worldbody></mujoco>')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
with mujoco.Renderer(model, height=64, width=64) as renderer:
    renderer.update_scene(data)
    frame = renderer.render()
assert frame.shape == (64, 64, 3) and frame.dtype == np.uint8 and frame.max() > frame.min()
with tempfile.TemporaryDirectory() as directory:
    video = Path(directory) / 'render.mp4'
    with imageio.get_writer(video, fps=30, codec='libx264', quality=8) as writer:
        for _ in range(3):
            writer.append_data(frame)
    with imageio.get_reader(video) as reader:
        frames = list(reader.iter_data())
    assert len(frames) == 3 and all(item.shape == frame.shape for item in frames)
'''
    environment = os.environ.copy()
    for name, value in graphics.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value
    subprocess.run([sys.executable, '-c', code], env=environment, check=True, timeout=60)


def main() -> None:
    parser = argparse.ArgumentParser(description='Install Massive SafeRL and configure rendering.')
    parser.add_argument('accelerator', choices=('gpu', 'tpu'))
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error('Python 3.10 or newer is required.')
    if sys.platform != 'linux':
        parser.error('Run this installer on Linux or WSL2.')
    run([sys.executable, '-m', 'pip', 'install', '-e', f'{ROOT}[{args.accelerator}]'])
    run([sys.executable, '-m', 'pip', 'check'])
    graphics, failure = check_graphics()
    if graphics is None and install_graphics():
        graphics, failure = check_graphics()
    if graphics is None:
        raise RuntimeError('No working OpenGL backend.\n' + failure)
    check_video(graphics)
    print(f"Installed {args.accelerator}: navigation, locomotion, evaluation, and rendering ({graphics['MUJOCO_GL']}).")


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        sys.exit(str(error))
