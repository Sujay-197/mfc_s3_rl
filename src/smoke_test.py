"""Pre-flight check for a level, run before committing hours to training.

It answers three questions cheaply, using the real wrapper stack:

  * where do episodes actually end -- the flag, a death, or the in-game clock?
  * does StageGuard fire when it should not? (A false fire truncates every
    episode at the first step and would silently ruin a run, so the observed
    world/stage values are printed rather than assumed.)
  * how far do scripted rightward policies get, and does one touch the flag?
    That measures the flagpole x and shows whether the configured action set can
    physically finish the level at all. A set with no "down" cannot enter a pipe,
    so if the flag needs one, this is where it shows up.

    python src/smoke_test.py --level 1-2 --steps 20000

Action indices 0-6 are identical in the simple, right_only and complex sets, so
the scripted policies below mean the same thing whichever set is configured.
"""

import sys, os, argparse, json, pathlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from levels import DEFAULT_LEVEL, LEVELS
from mario_env import ACTION_SETS, make_mario_env

# index -> label, valid for every action set
SCRIPTED = {"noop": 0, "right": 1, "right+A": 2, "right+A+B": 4}


def play(env, action, steps_left, label):
    """Play one episode, stopping early if it ends. Returns a result row."""
    obs, info = env.reset()
    steps, reward, x_max = 0, 0.0, 0
    stages, terminated, truncated = set(), False, False
    while steps < steps_left:
        obs, r, terminated, truncated, info = env.step(action)
        steps += 1
        reward += float(r)
        x_max = max(x_max, int(info.get("x_pos_max", 0)))
        stages.add((info.get("world"), info.get("stage"), info.get("area")))
        if terminated or truncated:
            break
    outcome = ("flag" if info.get("flag_get") else
               "left-stage" if info.get("left_stage") else
               "died" if terminated else
               "time-ran-out" if truncated else "still-playing")
    return {"policy": label, "steps": steps, "reward": round(reward, 1),
            "x_max": x_max, "outcome": outcome,
            "stages": sorted(s for s in stages if s[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=sorted(LEVELS), default=DEFAULT_LEVEL)
    ap.add_argument("--action-set", choices=sorted(ACTION_SETS), default=None)
    ap.add_argument("--steps", type=int, default=20000,
                    help="total agent steps to spend per scripted policy")
    ap.add_argument("--episodes", type=int, default=3, help="random-policy episodes")
    args = ap.parse_args()

    spec = LEVELS[args.level]
    action_set = args.action_set or spec["action_set"]
    print(f"level {args.level} ({spec['env_id']})")
    print(f"action set {action_set}: {len(ACTION_SETS[action_set])} actions")
    print(f"target world/stage {spec['target']}\n")

    env = make_mario_env(level=args.level, action_set=action_set)

    # 1. random policy: do bare episodes behave, and does StageGuard misfire?
    #    A misfire shows up as an episode ending after one step with left-stage.
    print(f"random policy, up to {args.episodes} episodes:")
    random_log = [play(env, env.action_space.sample(), 2000, f"random#{i + 1}")
                  for i in range(args.episodes)]
    for row in random_log:
        print(f"  {row['outcome']:<14} steps {row['steps']:>5}  "
              f"x_max {row['x_max']:>5}  stages {row['stages']}")

    # 2. scripted rightward policies: how far does a constant input get?
    print(f"\nscripted policies, up to {args.steps} steps and 20 episodes each:")
    scripted = []
    for label in ("right", "right+A", "right+A+B"):
        rows, left = [], args.steps
        while left > 0 and len(rows) < 20:
            row = play(env, SCRIPTED[label], left, label)
            left -= row["steps"]
            rows.append(row)
        best = max(rows, key=lambda r: r["x_max"])
        outcomes = {}
        for r in rows:
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        scripted.append(best)
        print(f"  {label:<11} {len(rows):>3} eps  best x_max {best['x_max']:>5}  "
              f"outcomes {outcomes}")
    env.close()

    every = random_log + scripted
    cleared = [r for r in every if r["outcome"] == "flag"]
    escaped = [r for r in every if r["outcome"] == "left-stage"]
    best_x = max((r["x_max"] for r in every), default=0)
    print(f"\nsummary")
    print(f"  furthest x reached     {best_x}")
    print(f"  flag touched           "
          f"{'yes -> flagpole x = ' + str(best_x) if cleared else 'no'}")
    print(f"  left-stage episodes    {len(escaped)}")
    if escaped:
        print("                         (warp pipes are being used; the guard is "
              "doing its job)")
    print("\nIf nothing touched the flag, the level may need a pipe ('down' is "
          "absent from the simple and right_only sets) or timing that a constant "
          "input cannot produce. That argues for --action-set complex -- worth "
          "knowing before a multi-hour run rather than after it.")


if __name__ == "__main__":
    main()
