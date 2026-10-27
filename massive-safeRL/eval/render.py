from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from uuid import uuid4

import numpy as np


class GridVideoRecorder:


    @staticmethod
    def parse_grid(value: str) -> tuple[int, int]:
        match = re.fullmatch(r'\s*([1-9]\d*)\s*[xX×]\s*([1-9]\d*)\s*', str(value))
        if match is None:
            raise ValueError('Render grid must be positive ROWSxCOLUMNS, for example 4x8')
        return int(match[1]), int(match[2])

    def __init__(self, env, path, grid=(1, 1), width=None, height=None, fps=25):
        self._renderer = None
        self._writer = None
        self._closed = False
        self._failed = False
        self.frame_count = 0
        self.path = Path(path).expanduser().resolve()
        self._partial_path = None
        if isinstance(grid, str):
            grid = self.parse_grid(grid)
        if len(grid) != 2 or any(int(n) != n or n < 1 for n in grid):
            raise ValueError('Render grid must contain two positive integers')
        self.rows, self.columns = map(int, grid)
        self.num_envs = self.rows * self.columns
        if self.num_envs != env.batch_size:
            raise ValueError('Render grid size must match the evaluation environment count')
        width = 1920 if width is None else width
        height = 1080 if height is None else height
        if any(int(n) != n or n < 2 or n % 2 for n in (width, height)):
            raise ValueError('Video width and height must be positive even integers')
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Render FPS must be finite and positive')
        if self.path.suffix.lower() != '.mp4':
            raise ValueError('Evaluation video output must use the .mp4 extension')
        self.width, self.height = int(width) // self.columns, int(height) // self.rows
        self.width -= (self.columns * self.width) % 2
        self.height -= (self.rows * self.height) % 2
        if min(self.width, self.height) < 2:
            raise ValueError('Video resolution must provide at least two pixels per cell dimension')
        self.task = env.task
        import imageio.v2 as imageio
        import imageio_ffmpeg
        import mujoco
        from PIL import Image, ImageDraw, ImageFont
        self._mujoco, self._Image, self._ImageDraw = mujoco, Image, ImageDraw
        self.model = copy.copy(env.mj_model)
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, self.width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, self.height)
        self.data = mujoco.MjData(self.model)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.azimuth = 270
        self.camera.elevation = -45
        self.camera.distance = math.sqrt(8)
        self._navigation = self.task.family not in ('velocity', 'locomotion')
        if self.task.family == 'locomotion':
            root_joints = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
            if root_joints.size != 1:
                raise ValueError('Locomotion rendering requires one floating robot root')
            self._agent = int(self.model.jnt_bodyid[root_joints[0]])
        else:
            body_name = 'agent' if self._navigation else 'torso'
            self._agent = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if self._agent < 0:
                raise ValueError(f'Evaluation model has no {body_name} body')
        if not self._navigation:
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.camera.fixedcamid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'track')
            if self.camera.fixedcamid < 0:
                raise ValueError('Evaluation model has no track camera')
            self.model.light_castshadow[:] = False
        self._floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        if self._navigation and self._floor >= 0:
            self.model.geom_rgba[self._floor] = (1.0, 1.0, 1.0, 1.0)
            self.model.geom_size[self._floor, :2] = 3.5
        self._option = mujoco.MjvOption()
        self._option.geomgroup[:] = 1
        self._identity = np.eye(3).reshape(-1)
        font_size = max(10, min(32, round(self.width / 24)))
        try:
            self._font = ImageFont.truetype('DejaVuSans.ttf', font_size)
        except OSError:
            self._font = ImageFont.load_default()
        self._line_height = self._font.getbbox('Env 0123')[3] + 4
        if self._navigation:
            self._visual_model = mujoco.MjModel.from_xml_string('''
                <mujoco>
                  <asset>
                    <texture type="skybox" fileright="star_hd.right.png" fileleft="star_hd.left.png"
                             fileback="star_hd.back.png" filefront="star_hd.front.png"
                             filedown="star_hd.bottom.png" fileup="star_hd.top.png"/>
                    <texture type="2d" name="matplane" builtin="checker"
                             rgb1="0.859 0.843 0.765" rgb2="0.949 0.914 0.855" width="100" height="100"/>
                    <material name="matplane" texture="matplane" texrepeat="10 10"
                              reflectance="0.1" shininess="0.1" specular="0.1"/>
                  </asset>
                  <worldbody>
                    <light name="top" pos="0 0 0.5" dir="0 0 -1" directional="true"
                           cutoff="100" exponent="1" diffuse="1 1 1" specular="0 0 0" castshadow="false"/>
                    <geom type="plane" size="3.5 3.5 0.1" material="matplane" rgba="1 1 1 1"/>
                  </worldbody>
                </mujoco>
            ''', assets={path.name: path.read_bytes() for path in (Path(__file__).parent / 'assets').glob('*.png')})
            self._visual_model.vis.global_.offwidth = self.model.vis.global_.offwidth
            self._visual_model.vis.global_.offheight = self.model.vis.global_.offheight
            self._visual_data = mujoco.MjData(self._visual_model)
            mujoco.mj_forward(self._visual_model, self._visual_data)
        else:
            self._visual_model, self._visual_data = self.model, self.data
        try:
            imageio_ffmpeg.get_ffmpeg_exe()
            self._renderer = mujoco.Renderer(self._visual_model, height=self.height, width=self.width,
                                             max_geom=max(256, self.model.ngeom + int(getattr(self.task, 'n_hazards', 0)) + 16))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._partial_path = self.path.with_name(f'.{self.path.stem}.{uuid4().hex}.partial.mp4')
            self._writer = imageio.get_writer(str(self._partial_path), format='FFMPEG', mode='I',
                                              fps=float(fps), codec='libx264', pixelformat='yuv420p',
                                              quality=8, macro_block_size=1, ffmpeg_log_level='error')
        except BaseException:
            self._failed = True
            try:
                self.close()
            except Exception:
                pass
            raise

    def _geom(self, kind, size, position, color):
        scene = self._renderer.scene
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError('Evaluation scene exceeds its geometry capacity')
        geom = scene.geoms[scene.ngeom]
        self._mujoco.mjv_initGeom(geom, kind, np.asarray(size, dtype=float),
                                 np.asarray(position, dtype=float),
                                 self._identity, np.asarray(color, dtype=np.float32))
        geom.category = self._mujoco.mjtCatBit.mjCAT_DECOR
        geom.transparent = color[3] < 1
        scene.ngeom += 1
        return geom

    def _task_geometry(self, frame, index):
        geom_type = self._mujoco.mjtGeom
        if self.task.family in ('goal', 'button', 'push'):
            x, y = frame['goal_xy'][index]
            if self.task.family == 'button':
                native = getattr(self.task, 'native_task', self.task)
                radius, alpha = 2 * native.button_size, 1.0
            else:
                radius, alpha = self.task.goal_size, 0.25
            self._geom(geom_type.mjGEOM_CYLINDER, (radius, radius / 2, 0),
                       (x, y, radius / 2 + 0.01), (0, 1, 0, alpha))
            for x, y in frame.get('hazards_xy', np.empty((self.num_envs, 0, 2)))[index]:
                self._geom(geom_type.mjGEOM_CYLINDER, (self.task.hazard_size, 0.01, 0),
                           (x, y, 0.02), (0, 0, 1, 0.25))
        elif self.task.family == 'circle':
            self._geom(geom_type.mjGEOM_CYLINDER, (self.task.circle_radius, 0.01, 0),
                       (0, 0, 0.01), (0, 1, 0, 0.1))
            for wall in self.task.objects.get('sigwalls', {}).get('walls', ()):
                x_wall = wall['axis'] == 'x'
                size = (0.05, 3.5, 0.3) if x_wall else (3.5, 0.05, 0.3)
                position = (wall['coord'], 0, 0.25) if x_wall else (0, wall['coord'], 0.25)
                self._geom(geom_type.mjGEOM_BOX, size, position, (1, 1, 0, 0.1))
        agent_pos = self.data.xpos[self._agent]
        if self.task.family != 'locomotion' and np.any(frame.get('step_costs', frame['costs'])[index] > 0):
            self._geom(geom_type.mjGEOM_SPHERE, (0.25, 0.25, 0.25), agent_pos, (0.5, 0, 0, 0.5))

    def write(self, frame):
        if self._closed:
            raise RuntimeError('The evaluation video recorder is closed')
        try:
            values = {key: np.asarray(value) for key, value in frame.items()}
            shapes = {'qpos': (self.num_envs, self.model.nq), 'qvel': (self.num_envs, self.model.nv),
                      'reward': (self.num_envs,), 'costs': (self.num_envs, self.task.n_costs),
                      'step': (self.num_envs,), 'done': (self.num_envs,)}
            if self.task.family in ('goal', 'button', 'push'):
                shapes['goal_xy'] = (self.num_envs, 2)
                if getattr(self.task, 'n_hazards', 0):
                    shapes['hazards_xy'] = (self.num_envs, self.task.n_hazards, 2)
            for key, shape in shapes.items():
                if key not in values or values[key].shape != shape:
                    raise ValueError(f'Render field {key} must have shape {shape}')
                if not np.all(np.isfinite(values[key])):
                    raise ValueError(f'Render field {key} contains a nonfinite value')
            for key, shape in (('mocap_pos', (self.num_envs, self.model.nmocap, 3)),
                               ('mocap_quat', (self.num_envs, self.model.nmocap, 4)),
                               ('step_costs', shapes['costs'])):
                if key in values and (values[key].shape != shape or not np.all(np.isfinite(values[key]))):
                    raise ValueError(f'Render field {key} must be finite with shape {shape}')
            canvas = np.empty((self.rows * self.height, self.columns * self.width, 3), dtype=np.uint8)
            for index in range(self.num_envs):
                self.data.qpos[:] = values['qpos'][index]
                self.data.qvel[:] = values['qvel'][index]
                for name in ('mocap_pos', 'mocap_quat'):
                    if name in values:
                        getattr(self.data, name)[:] = values[name][index]
                self._mujoco.mj_kinematics(self.model, self.data)
                self._mujoco.mj_comPos(self.model, self.data)
                self._mujoco.mj_camlight(self.model, self.data)
                self.camera.lookat[:] = self.data.xpos[self._agent]
                scene = self._renderer.scene
                self._mujoco.mjv_updateScene(self.model, self.data, self._option, None, self.camera,
                                             self._mujoco.mjtCatBit.mjCAT_ALL, scene)
                if self._navigation:
                    self._mujoco.mjv_makeLights(self._visual_model, self._visual_data, scene)
                    for geom in scene.geoms[:scene.ngeom]:
                        if geom.objtype == self._mujoco.mjtObj.mjOBJ_GEOM and geom.objid == self._floor:
                            geom.matid = 0
                            geom.reflectance = 0.1
                            geom.shininess = geom.specular = 0.1
                self._task_geometry(values, index)
                panel = self._Image.fromarray(self._renderer.render())
                draw = self._ImageDraw.Draw(panel, 'RGBA')
                row, column = divmod(index, self.columns)
                violation = bool(np.any(values.get('step_costs', values['costs'])[index] > 0))
                status = '  DONE' if values['done'][index] else ''
                draw.text((5, 3), f'Env {index + 1:03d}   Step {int(values["step"][index])}{status}',
                          font=self._font, fill=(255, 255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0, 255))
                costs = ', '.join(f'{value:.1f}' for value in values['costs'][index]) or '0.0'
                draw.text((5, self._line_height + 3), f'Reward {values["reward"][index]:.1f}   Cost {costs}',
                          font=self._font, fill=(255, 64, 64, 255) if violation else (255, 255, 255, 255),
                          stroke_width=1, stroke_fill=(0, 0, 0, 255))
                canvas[row * self.height:(row + 1) * self.height,
                       column * self.width:(column + 1) * self.width] = np.asarray(panel)
            self._writer.append_data(canvas)
            self.frame_count += 1
        except BaseException:
            self._failed = True
            try:
                self.close()
            except Exception:
                pass
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        failure = None
        for resource in (self._writer, self._renderer):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    failure = failure or exc
        if self._partial_path is not None:
            try:
                if not self._failed and failure is None and self.frame_count:
                    if not self._partial_path.exists() or self._partial_path.stat().st_size == 0:
                        raise RuntimeError('The video encoder produced an empty output')
                    self._partial_path.replace(self.path)
                else:
                    self._partial_path.unlink(missing_ok=True)
            except Exception as exc:
                failure = failure or exc
                self._partial_path.unlink(missing_ok=True)
        if failure is not None:
            raise failure

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._failed = self._failed or exc_type is not None
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except Exception:
                pass
