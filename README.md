# Reinforcement Learning for Super Mario Bros. 1-1

A PPO agent trained to complete World 1-1 of Super Mario Bros. autonomously,
using `gym-super-mario-bros` (Gymnasium-native) and Stable-Baselines3.

## Results

See [`artifacts/results.md`](artifacts/results.md) for the evaluation table,
[`artifacts/training_curve.png`](artifacts/training_curve.png) for learning
curves across seeds, and `artifacts/demo.mp4` for the trained agent playing
the level.

## Setup

Requires Python >= 3.13 (`nes-py` 9.x ships wheels for CPython 3.13/3.14 only)
and an NVIDIA GPU with a CUDA 12.8 build of PyTorch.

```bash
uv venv --python 3.14 .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install gym-super-mario-bros stable-baselines3 \
    opencv-python-headless imageio imageio-ffmpeg matplotlib tensorboard pandas
```

## Usage

```bash
python src/benchmark.py                 # measure env throughput
powershell -File run_all.ps1            # launch the three training runs
python src/status.py                    # check progress across runs
python src/report.py                    # build all submission artifacts
```

## Method

**Environment.** `SuperMarioBros-1-1-v0`, single life per episode, so an
episode ends on death or at the flagpole.

**Observation pipeline.** Actions are taken every 4th frame, with the last two
frames max-pooled — the NES flickers sprites across alternate frames, so a
single sampled frame can be missing an enemy entirely. Frames are resized to
84x84 and converted to grayscale (colour carries no task information here),
then 4 consecutive frames are stacked so the agent can perceive velocity.
Final observation: `(84, 84, 4)` uint8.

**Action space.** `SIMPLE_MOVEMENT` (7 actions) for the two primary runs.
A third run uses `RIGHT_ONLY` (5 actions) as an ablation: 1-1 requires no
backtracking, so removing left-moving actions shrinks the exploration problem
and is expected to reach the flag in fewer steps.

**Reward.** The environment's built-in reward (rightward velocity, a small
clock penalty, and a death penalty) is used unchanged, divided by 10. The
scaling is numerical hygiene rather than shaping: raw rewards spike to ±15,
which makes PPO's value loss dominate the policy loss. A +50 (pre-scaling)
bonus on reaching the flag makes the reward curve step visibly on success;
it contributes little to the gradient, since the flag is rare until the level
is nearly solved.

**Algorithm.** PPO with a Nature-CNN policy.

| hyperparameter | value | rationale |
|---|---|---|
| learning rate | 1e-4 | stable for long Mario runs |
| n_steps | 512 | per-env rollout length |
| batch size | 256 | |
| epochs | 10 | |
| gamma | 0.9 | low by Atari standards, standard for Mario: frame-skip 4 shortens the useful credit-assignment horizon |
| gae_lambda | 0.95 | |
| clip range | 0.2 | |
| entropy coef | 0.01 | sustains exploration past early plateaus |

**Parallelism.** Throughput is bound by CPU-side NES emulation, not the GPU —
the policy is ~1.7M parameters and leaves the GPU largely idle. Benchmarking
(`src/benchmark.py`) measured 1,000 / 1,271 / 1,681 / 1,598 steps-per-second
at 4 / 8 / 12 / 16 workers, so 12 workers is the throughput ceiling on this
machine and 16 is actively worse. Three runs therefore train concurrently at
4 workers each.

**Seeds.** PPO on Mario has high seed variance, so three independent runs are
trained in parallel rather than one longer run. The best-performing policy is
used for the demo, and per-seed results are reported.

**Checkpointing.** Every run saves periodic checkpoints and separately keeps a
best-ever model, tracked by rolling mean training reward. This is insurance
against late-training policy collapse, which PPO on Mario is prone to once the
level is mastered. Evaluation deliberately does not run during training: a
separate eval environment would steal CPU from the rollout workers, so the
rigorous 20-episode evaluation runs afterwards in `report.py`.

## Comparison to prior work

[yumouwei/super-mario-bros-reinforcement-learning](https://github.com/yumouwei/super-mario-bros-reinforcement-learning)
solves the same task with PPO, and the differences are informative.

| | That implementation | This one |
|---|---|---|
| observation | RAM-derived 13x16 tile grid (memory 0x0500-0x069F) | 84x84 grayscale pixels |
| policy | `MlpPolicy`, two Dense(64) layers | `CnnPolicy` (Nature CNN) |
| hyperparameters | SB3 defaults + linear LR annealing | gamma=0.9, lr=1e-4, 10 epochs |
| steps to clear 1-1 | ~10M | ~0.4M (see `artifacts/results.md`) |

The RAM-based representation hands the agent a pre-parsed tile map, removing
the perception problem entirely; the pixel-based agent here has to learn it.
The faster convergence in this implementation comes from Mario-specific
hyperparameters rather than the representation -- in particular gamma=0.9,
which suits the short credit-assignment horizon that frame-skip 4 produces,
where SB3's default gamma=0.99 does not.

Two of that author's findings bear directly on results observed here.

**Frame stacking prevents stalling.** They report that a single-frame model
"failed, getting stuck", and that temporal information helps the agent make
jump decisions. This matches a behaviour observed in this project: under a
*deterministic* (greedy) policy the trained agent runs to x~2370 and then
stands motionless until the in-game clock expires -- 90% of the episode
spent stationary, with only 10% of steps showing any movement. The same
policy *sampled* rather than argmaxed clears the level. Stalling in front of
an obstacle is evidently a narrow failure of action selection rather than of
the learned representation, and both temporal context and action-sampling
mitigate it.

**Layout memorisation.** They report that the agent transfers poorly to 1-2,
4-1 and 5-1, having learned specific tile patterns rather than general
mechanics. This project therefore scopes its claim to World 1-1 explicitly
and does not assert generalisation.

## Layout

```
src/mario_env.py   environment construction and wrappers
src/train.py       PPO training for one seed
src/benchmark.py   throughput measurement
src/status.py      health check across concurrent runs
src/record.py      render a policy to mp4
src/report.py      evaluation, curves, results table, demo video
run_all.ps1        launches the three training runs
runs/              per-run checkpoints, monitor logs, tensorboard
artifacts/         submission outputs
```

Checkpoints and TensorBoard logs (~400 MB per run) are not committed. What is
committed per run -- `monitor.csv`, `config.json`, `first_clear.json` -- is
enough to regenerate the training curves and results table without retraining.
The best model of the winning run (`runs/simple_seed0/best_model.zip`) is
included so the demo can be reproduced directly:

```bash
python src/record.py --model runs/simple_seed0/best_model.zip --out artifacts/demo.mp4 --episodes 5
```
