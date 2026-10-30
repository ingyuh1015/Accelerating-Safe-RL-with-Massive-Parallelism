# Accelerating Safe Reinforcement Learning with Massive Parallelism

Official Github Repository for "Accelerating Safe Reinforcement Learning with Massive Parallelism". (NeurIPS 2026)

## Setup

Run from the repository root on Linux or WSL2:

```bash
conda create -n massive-saferl python=3.10 -y
conda activate massive-saferl
```

GPU (MJX and MuJoCo Warp):

```bash
python install.py gpu
```

Cloud TPU (MJX):

```bash
python install.py tpu
```

Both install navigation, locomotion, training, evaluation, and rendering. The installer checks rendering and installs missing graphics libraries when needed. EGL, OSMesa, or GLFW is selected automatically.

## Train

MJX:

```bash
python -m massive_saferl train \
  --algo CPO --task SafetyPointGoal1 \
  --preset staggered_nstep --tighten on --wandb false
```

MuJoCo Warp:

```bash
python -m massive_saferl train_warp \
  --algo CRPO --task Laikago \
  --preset staggered_nstep --tighten on \
  --seeds 0,1,2,3,4 --sequential --wandb false
```

Use `--preset standard --tighten off` for standard training. `--scale 2` and `--scale 4` increase the parallel environment scale. Algorithm, task, and preset configurations are in [`massive-safeRL/yaml`](massive-safeRL/yaml).

Runs are saved under `results/<task>/<algo>/<run_id>_s<seed>/`, including `config.yaml`, `progress.csv`, and `model/`.

## Evaluate and Render

Set `RUN_DIR` to a saved training run:

```bash
RUN_DIR='results/Laikago/crpo/<run_id>_s0'
```

MJX:

```bash
python -m massive_saferl eval \
  --run-dir "$RUN_DIR" --render-grid 2x2 --output evaluation.json
```

MuJoCo Warp:

```bash
python -m massive_saferl eval_warp \
  --run-dir "$RUN_DIR" --render-grid 2x2
```

Warp saves evaluation JSON and MP4 files under `$RUN_DIR/eval_warp/`. MJX saves the video under `$RUN_DIR/eval/`. To evaluate without rendering, replace `--render-grid 2x2` with `--episodes 100 --num-envs 100`.

## License

[MIT License](LICENSE). Third-party code and assets retain their existing licenses.
