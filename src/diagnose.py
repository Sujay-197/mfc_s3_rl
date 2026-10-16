"""Diagnose a policy: x-position trajectory and stochastic-vs-greedy spread.

    python src/diagnose.py <model.zip> <action_set> [level]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from levels import DEFAULT_LEVEL
from mario_env import make_mario_env

model_path, action_set = sys.argv[1], sys.argv[2]
level = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_LEVEL
venv = VecFrameStack(DummyVecEnv(
    [lambda: make_mario_env(level=level, action_set=action_set)]), 4)
model = PPO.load(model_path, device="cpu")

# 1. greedy episode, logging x each agent step
obs = venv.reset(); done = False; xs = []; total = 0.0
while not done:
    a, _ = model.predict(obs, deterministic=True)
    obs, r, dones, infos = venv.step(a)
    xs.append(int(infos[0].get("x_pos", 0))); total += float(r[0]); done = bool(dones[0])
xs = np.array(xs)
print(f"greedy: {len(xs)} agent steps, reward {total:.1f}, x {xs.min()}->{xs.max()}")
print(f"  final x {xs[-1]}, time left {infos[0].get('time')}")
# where does it stop making progress?
peak_i = int(np.argmax(xs))
print(f"  reached max x={xs.max()} at step {peak_i} ({peak_i/len(xs):.0%} through episode)")
print(f"  steps spent after peak: {len(xs)-peak_i} ({(len(xs)-peak_i)/len(xs):.0%} of video)")
moved = np.abs(np.diff(xs)) > 0
print(f"  steps with any x movement: {moved.sum()}/{len(moved)} ({moved.mean():.0%})")
print("  x every 10% of episode:", [int(xs[int(f*(len(xs)-1))]) for f in np.linspace(0,1,11)])

# 2. stochastic spread
print("\nstochastic (sampled) episodes:")
for ep in range(3):
    obs = venv.reset(); done = False; tot = 0.0; xm = 0
    while not done:
        a, _ = model.predict(obs, deterministic=False)
        obs, r, dones, infos = venv.step(a)
        tot += float(r[0]); xm = max(xm, int(infos[0].get("x_pos_max", 0))); done = bool(dones[0])
    print(f"  ep{ep+1}: reward={tot:.1f} x_max={xm}")
venv.close()
