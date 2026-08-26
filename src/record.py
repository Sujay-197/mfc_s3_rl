"""Record a trained (or untrained) policy playing 1-1 to an mp4.

Kept separate from report.py so the encode path can be smoke-tested before
any training starts -- discovering a missing encoder on deadline morning is
not a risk worth taking.
"""

import sys, os, argparse, pathlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import imageio.v2 as imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from mario_env import make_mario_env


def grab_frame(env):
    """Return the current RGB framebuffer as a fresh array.

    Deliberately NOT env.render(): in this wrapper stack render() returns a
    byte-identical frame on every call, which silently produces videos that
    are one still image repeated. The emulator's own screen buffer updates
    correctly, so it is read directly -- and copied, because it is a live
    buffer that would otherwise alias across frames.
    """
    return np.array(env.unwrapped.screen, copy=True)


def record(model_path, out_path, action_set="simple", episodes=3, fps=30,
           deterministic=False, best_only=True, stagnation_steps=120):
    """Play `episodes` episodes; write the best one (or all) to `out_path`.

    Two recording-specific choices, neither of which affects evaluation:

    * The policy is *sampled*, not greedy. The environment is deterministic,
      so a greedy policy replays one identical trajectory -- and if that
      trajectory stalls, every episode stalls identically. Sampling also
      measurably outperforms greedy here.
    * An episode is cut short after `stagnation_steps` agent steps without
      forward progress. A stalled agent otherwise stands still until the
      in-game clock expires, which produced demo videos that were ~90%
      motionless. This trims the video only; it does not alter any
      reported metric.
    """
    base = make_mario_env(action_set, render_mode="rgb_array")
    venv = VecFrameStack(DummyVecEnv([lambda: base]), 4)
    model = PPO.load(model_path, device="cpu") if model_path else None

    runs = []
    for ep in range(episodes):
        obs = venv.reset()
        frames, total_r, done = [], 0.0, False
        info = {}
        best_x, stalled = 0, 0
        while not done:
            if model is None:
                action = np.array([venv.action_space.sample()])
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, dones, infos = venv.step(action)
            frames.append(grab_frame(base))
            total_r += float(reward[0]); info = infos[0]; done = bool(dones[0])
            x = int(info.get("x_pos", 0))
            if x > best_x:
                best_x, stalled = x, 0
            else:
                stalled += 1
                if stalled >= stagnation_steps:
                    print(f"    (stalled at x={best_x}, cutting recording)")
                    break
        runs.append({"frames": frames, "reward": total_r,
                     "flag": bool(info.get("flag_get", False)),
                     "x": int(info.get("x_pos_max", 0))})
        print(f"  episode {ep+1}: reward={total_r:.1f} x_max={runs[-1]['x']} "
              f"flag={runs[-1]['flag']}")

    venv.close()
    # Prefer a clearing episode; otherwise the one that got furthest.
    runs.sort(key=lambda r: (r["flag"], r["x"]), reverse=True)
    chosen = runs[:1] if best_only else runs
    frames = [f for r in chosen for f in r["frames"]]

    # Guard: a video whose frames are all identical is a capture bug, not a
    # stalled agent. Fail loudly rather than shipping a still image as footage.
    distinct = sum(1 for f in frames[1:] if not np.array_equal(f, frames[0]))
    if len(frames) > 1 and distinct == 0:
        raise RuntimeError(
            f"refusing to write {out_path}: all {len(frames)} frames are "
            "identical -- the frame capture is broken, not the agent")

    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps, macro_block_size=1)
    print(f"  frame check: {distinct}/{len(frames)-1} frames differ from the first")
    print(f"wrote {out_path}  ({len(frames)} frames, "
          f"flag={chosen[0]['flag']}, x_max={chosen[0]['x']})")
    return chosen[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="omit for a random policy")
    ap.add_argument("--out", default="artifacts/demo.mp4")
    ap.add_argument("--action-set", choices=["simple", "right_only"], default="simple")
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()
    record(args.model, args.out, args.action_set, args.episodes)
