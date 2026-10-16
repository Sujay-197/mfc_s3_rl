"""Measure environment throughput to size the overnight runs.

Reports agent-steps/sec (each agent step is 4 emulated NES frames) for a
range of worker counts, so we can pick the point where adding processes
stops helping. Emulation is CPU-bound, so this is the number that decides
how long training takes -- not the GPU.
"""

import sys, time, argparse
sys.path.insert(0, "src")
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from levels import DEFAULT_LEVEL, LEVELS
from mario_env import make_mario_env


def build(n_envs, level=DEFAULT_LEVEL, action_set=None):
    fns = [(lambda i=i: make_mario_env(level=level, action_set=action_set))
           for i in range(n_envs)]
    return VecFrameStack(SubprocVecEnv(fns, start_method="spawn"), 4)


def bench(n_envs, seconds=20.0, level=DEFAULT_LEVEL, action_set=None):
    venv = build(n_envs, level, action_set)
    venv.reset()
    steps = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        venv.step(np.array([venv.action_space.sample() for _ in range(n_envs)]))
        steps += n_envs
    dt = time.perf_counter() - t0
    venv.close()
    return steps / dt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="+", default=[4, 8, 12, 16])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--level", choices=sorted(LEVELS), default=DEFAULT_LEVEL)
    args = ap.parse_args()

    print(f"level {args.level}")
    print(f"{'workers':>8} {'steps/s':>10} {'frames/s':>10}  {'1M steps in':>12}")
    for n in args.workers:
        sps = bench(n, args.seconds, args.level)
        mins = 1e6 / sps / 60
        print(f"{n:>8} {sps:>10.0f} {sps*4:>10.0f}  {mins:>10.1f} min")
