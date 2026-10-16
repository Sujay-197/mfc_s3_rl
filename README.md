# Reinforcement Learning for Super Mario Bros.

A PPO agent trained to complete Super Mario Bros. from raw pixels, using
`gym-super-mario-bros` (Gymnasium-native) and Stable-Baselines3. Worlds 1-1 and
1-2 are complete and reported; World 1-3 is implemented to the same extent and
awaiting its training run (see [World 1-3](#world-1-3)). The pipeline itself is
level-generic: `src/levels.py` configures one stage from each of the game's
eight worlds, so every level-aware script accepts `--level` for any of them.

## Results

See [`artifacts/results.md`](artifacts/results.md) for the 1-1 evaluation table,
[`artifacts/training_curve.png`](artifacts/training_curve.png) for learning
curves across seeds, and `artifacts/demo.mp4` for the trained agent playing
the level. The 1-2 equivalents are written alongside as `results_1_2.md`,
`training_curve_1_2.png`, and `demo_1_2.mp4`.

## Levels

`src/levels.py` is the single source of truth; every level-aware script
(`smoke_test.py`, `train.py`, `report.py`, `build_page.py`, ...) takes its
choices from it. Per-level artifacts are suffixed (`results_2_1.md`,
`demo_5_1.mp4`, ...) so sets coexist rather than overwrite each other.

| level | world | env id | actions | status |
|---|---|---|---|---|
| 1-1 | 1 | `SuperMarioBros-1-1-v0` | simple + right_only ablation | trained, reported |
| 1-2 | 1 | `SuperMarioBros-1-2-v0` | complex (exit pipe) | trained, reported |
| 1-3 | 1 | `SuperMarioBros-1-3-v0` | simple | implemented, smoke-tested, awaiting run |
| 2-1 | 2 | `SuperMarioBros-2-1-v0` | simple | configured, env validated |
| 3-1 | 3 | `SuperMarioBros-3-1-v0` | simple | configured, env validated |
| 4-1 | 4 | `SuperMarioBros-4-1-v0` | simple | configured, env validated |
| 5-1 | 5 | `SuperMarioBros-5-1-v0` | simple | configured, env validated |
| 6-1 | 6 | `SuperMarioBros-6-1-v0` | simple | configured, env validated |
| 7-1 | 7 | `SuperMarioBros-7-1-v0` | simple | configured, env validated |
| 8-1 | 8 | `SuperMarioBros-8-1-v0` | simple | configured, env validated |

"env validated" means the env id resolves in the installed emulator and a
live reset reports the world/stage StageGuard expects, with no guard misfire.
Each level should still get a full `smoke_test.py --level X-Y` pass before a
long run is committed to it.


## Setup

Requires Python >= 3.13 (`nes-py` 9.x ships wheels for CPython 3.13/3.14 only)
and an NVIDIA GPU with a CUDA 12.8 build of PyTorch.

```bash
uv venv --python 3.14 .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

`opencv-python-headless` is not optional despite no direct `cv2` import:
`gymnasium.wrappers.ResizeObservation` imports it internally.

## Usage

```bash
python src/smoke_test.py --level 1-2         # pre-flight check before a long run
python src/benchmark.py                      # measure env throughput
powershell -File run_all.ps1                 # 1-1, the original three runs
powershell -File run_all.ps1 -Level 1-3    # 1-3, the next level (simple, 2 seeds)
python src/status.py                         # check progress across runs
python src/report.py                         # build 1-1 submission artifacts
python src/report.py --level 1-2             # build 1-2 submission artifacts
python src/build_page.py --level 1-2         # standalone HTML page
```


## Batch files

The three scripts below are the Windows automation for this project; the engine
stays `run_all.ps1` + the Python entry points.

| File | What it does |
|---|---|
| `setup.bat` | Creates `.venv` if missing, then installs torch (CUDA build) + `requirements.txt` |
| `train.bat [level] [seeds] [nenvs] [maxhours] [p]` | Trains one level, one seed at a time (`p` = parallel); logs under `logs/` |
| `outputs.bat <level>` | Regenerates the report, training curve, demo and HTML page for a level |
## Method

**Environment.** `SuperMarioBros-1-1-v0`, single life per episode, so an
episode ends on death or at the flagpole.

**Observation pipeline.** Actions are taken every 4th frame, with the last two
frames max-pooled â€” the NES flickers sprites across alternate frames, so a
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
scaling is numerical hygiene rather than shaping: raw rewards spike to Â±15,
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

**Parallelism.** Throughput is bound by CPU-side NES emulation, not the GPU â€”
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

## World 1-2

1-2 uses the identical pipeline, selected by `--level 1-2`. Two things are
genuinely different from 1-1, and both were found by the pre-flight smoke test
rather than assumed.

**The level ends at a pipe, so the action set changes.** 1-2 is underground and
finishes by walking right into the exit pipe, which takes Mario to an outdoor
area, up a staircase, and onto a flagpole. `SIMPLE_MOVEMENT` and `RIGHT_ONLY`
contain no `down` action, so neither can enter a pipe; only `COMPLEX_MOVEMENT`
(12 actions) can. 1-2 is therefore configured with `complex` in `src/levels.py`.
Measured, not assumed: a constant-input policy reaches xâ‰ˆ180-198 in the first
Goomba pair under `simple`, and the same policy under `complex` reaches xâ‰ˆ675
before dying at the first Piranha Plant pipe.

**The level contains warp pipes into worlds 2, 3 and 4.** An agent that takes
one leaves the level, and the single-stage environment does not end the episode
on that transition â€” so it could keep playing, reach another level's flagpole,
and be recorded as having cleared 1-2. `StageGuard` (`src/mario_env.py`) ends
the episode the moment the game's own world/stage registers leave `(1, 2)` and
reports it through `info["left_stage"]`, which `report.py` counts as `escapes`
so warp escapes are never folded into the clear rate.

**Flagpole position is measured, not hardcoded.** 1-1's `3160` is a level
constant in `src/levels.py`. 1-2's began as `None` and was measured from the
run's own `monitor.csv` by `report.py`: the furthest x reached by an episode
that actually touched the flag, 3161. That measured value is now recorded in
`src/levels.py`, so the results table, the curve and the HTML page all report
the same number. 1-3 stays `None` until a run clears it: until then the
progress plot draws no reference line, because a guessed one would read as a
target the agent had actually been measured against.

**Running it.** 1-2 is harder and longer than 1-1, so give it a real budget:

```bash
powershell -File run_all.ps1 -Level 1-2 -PrimaryActions complex -NEnvs 2 -Seeds 2
```

`-NEnvs` and `-Seeds` exist because of memory, not tuning. A run's footprint is
the PPO parent (â‰ˆ1.1 GB: torch plus the CUDA context) plus its emulator workers
(â‰ˆ0.39 GB each), so 4 workers and 3 concurrent runs want roughly 10 GB of
commit. Measured here: 2.4 GB for one 2-env run. On a machine with a couple of
GB of headroom, add `-Sequential` to run the seeds one at a time, or lower
`-MaxHours`. The BLAS thread limits in `run_all.ps1` matter for the same
reason â€” a thread pool per emulator worker is what exhausted the commit limit.

Two bugs were found and fixed while standing this up, both of which also
affected 1-1:

- `train.py` imported `stable_baselines3.PPO` at module scope. On Windows every
  `SubprocVecEnv` worker re-imports that module, so each emulator worker loaded
  the full torch and CUDA runtime it never uses; the rollout buffer then failed
  to allocate under 7 MB. The PPO import now happens inside `main()`, after the
  workers are spawned.
- `StageGuard` only set `info["left_stage"]` when it fired, but `VecMonitor`
  reads every registered `info_keywords` entry on every step, so training died
  with `KeyError: 'left_stage'`. It is now always present.

## World 1-3

The next stage after 1-2, selected by `--level 1-3`. 1-3 is above ground: it
hops between platforms over pits and ends at a flagpole rather than an exit
pipe, so it keeps the `simple` action set used for 1-1 -- there is no pipe to
hold `down` for. Configured in `src/levels.py` as `SuperMarioBros-1-3-v0` with
StageGuard target `(1, 3)`; `flagpole_x` stays `None` until a run clears it
and `report.py` measures it.

The smoke test (`python src/smoke_test.py --level 1-3`) confirms the guard
side: no StageGuard misfires, no warp escapes, every episode staying in
world/stage `(1, 3)`. It cannot demo a clear, though: constant-input policies
die at the first platform gap around x 300, because the level needs timed
jumps rather than a button the action set lacks. That matches 1-2, where
constant input never touched the flag either, and the trained agent went on to
clear the level at 65%.

```bash
python src/smoke_test.py --level 1-3         # pre-flight check before a long run
powershell -File run_all.ps1 -Level 1-3      # two simple seeds, 4 envs each
powershell -File run_all.ps1 -Level 1-3 -Seeds 3   # adds a right_only ablation run
python src/report.py --level 1-3             # results_1_3.md, curve, demo
python src/build_page.py --level 1-3         # page_1_3.html
```

Run directories are tagged `lvl13_...` by `train.py`, so they live beside the
1-1 and 1-2 runs in `runs/` without colliding with them.


## Layout

```
src/levels.py      per-level config: env id, target, flagpole, action set
src/mario_env.py   environment construction and wrappers
src/smoke_test.py  pre-flight check for a level, before a long run
src/train.py       PPO training for one seed
src/benchmark.py   throughput measurement
src/status.py      health check across concurrent runs
src/record.py      render a policy to mp4
src/diagnose.py    x-position trajectory and greedy-vs-sampled spread
src/report.py      evaluation, curves, results table, demo video
src/build_page.py  standalone HTML page from a template
run_all.ps1        launches the training runs
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
