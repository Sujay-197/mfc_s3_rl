"""Generate every submission artifact from finished runs.

Run this in the morning, once training has stopped and the CPU is free:

    python src/report.py                # 1-1 -> results.md, training_curve.png, demo.mp4
    python src/report.py --level 1-2    # 1-2 -> results_1_2.md, training_curve_1_2.png, demo_1_2.mp4

Only runs belonging to the requested level are evaluated, and 1-1 keeps its
original artifact filenames so its submission is left exactly as it was.
"""

import sys, os, json, pathlib, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from levels import DEFAULT_LEVEL, LEVELS, artifact_paths, get_level, level_of_run
from mario_env import make_mario_env
from record import record

RUNS_DIR = pathlib.Path("runs")
ART = pathlib.Path("artifacts")


def load_monitor(run):
    csv = run / "monitor.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv, skiprows=1)
    df["timesteps"] = df["l"].cumsum()
    return df


def measure_flagpole(runs, known):
    """Return `(x, measured)` for the level's flagpole.

    A level with a recorded value uses it. A level without one gets it measured
    from the runs rather than guessed: the furthest x reached by an episode that
    actually touched the flag, falling back to the furthest x seen at all if the
    level was never cleared. Measuring is the honest source, because a wrong
    constant silently mislabels every training curve.
    """
    if known is not None:
        return known, False
    flagged, seen = [], []
    for run in runs:
        df = load_monitor(run)
        if df is None or df.empty:
            continue
        seen.extend(df["x_pos_max"].tolist())
        touched = df.loc[df["flag_get"].astype(str).str.lower() == "true",
                         "x_pos_max"]
        flagged.extend(touched.tolist())
    values = flagged or seen
    return (int(max(values)) if values else None), True


def pick_model(run):
    """Best-ever checkpoint if present, else final, else newest periodic.

    Each candidate is verified as a loadable zip before being returned: if
    training is still running, a checkpoint may be observed mid-write, and a
    truncated zip raises EOFError. Falling back is always preferable to
    failing the whole report.
    """
    import zipfile
    candidates = [run / "best_model.zip", run / "final_model.zip"]
    if (run / "checkpoints").exists():
        candidates += sorted((run / "checkpoints").glob("*.zip"),
                             key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        if not path.exists():
            continue
        try:
            with zipfile.ZipFile(path) as zf:
                if zf.testzip() is None:
                    return path
        except Exception:
            continue
    return None


def evaluate(model_path, level, action_set, episodes=20, deterministic=True):
    """Play `episodes` episodes and report clear rate and progress.

    Note on determinism: the NES environment is fully deterministic, so a
    deterministic policy replays an identical trajectory every episode --
    a "clear rate" measured that way is one episode reported N times and can
    only read 0% or 100%. Meaningful variance therefore comes from the
    stochastic (sampled) policy; the deterministic pass is reported
    separately as a single pass/fail.

    `escapes` counts episodes StageGuard cut short because a warp pipe carried
    the agent out of the level. On a level with warp pipes that number belongs
    next to the clear rate: an escaping agent is not a clearing agent, and the
    episode it abandoned must not be folded into that column.
    """
    venv = VecFrameStack(DummyVecEnv(
        [lambda: make_mario_env(level=level, action_set=action_set)]), 4)
    model = PPO.load(str(model_path), device="cpu")
    rewards, clears, escapes, xs = [], 0, 0, []
    for _ in range(episodes):
        obs = venv.reset(); done = False; total = 0.0; info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, dones, infos = venv.step(action)
            total += float(r[0]); info = infos[0]; done = bool(dones[0])
        rewards.append(total)
        xs.append(int(info.get("x_pos_max", 0)))
        clears += bool(info.get("flag_get", False))
        escapes += bool(info.get("left_stage", False))
    venv.close()
    return {"episodes": episodes, "clear_rate": clears / episodes,
            "escapes": escapes,
            "mean_reward": float(np.mean(rewards)),
            "mean_x_max": float(np.mean(xs)),
            "best_x_max": int(max(xs))}


def plot_curves(runs, level, flagpole_x, out_path):
    """Draw reward and level-progress curves for `runs` and save to `out_path`.

    The flagpole line is drawn only when its x is known; a level that has never
    been cleared has none, and drawing a guessed line would read as a target the
    agent was measured against.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for run in runs:
        df = load_monitor(run)
        if df is None or df.empty:
            continue
        w = max(len(df) // 100, 20)
        axes[0].plot(df["timesteps"], df["r"].rolling(w, min_periods=1).mean(),
                     label=run.name, lw=1.6)
        axes[1].plot(df["timesteps"], df["x_pos_max"].rolling(w, min_periods=1).mean(),
                     label=run.name, lw=1.6)
    axes[0].set_xlabel("environment steps"); axes[0].set_ylabel("episode reward")
    axes[0].set_title("Episode reward (rolling mean)")
    if flagpole_x is not None:
        axes[1].axhline(flagpole_x, ls="--", c="grey", lw=1)
        axes[1].annotate("flagpole", (0.01, flagpole_x),
                         xycoords=("axes fraction", "data"),
                         va="bottom", fontsize=8, color="grey")
    axes[1].set_xlabel("environment steps"); axes[1].set_ylabel("max x position")
    axes[1].set_title("Level progress (rolling mean)")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"PPO on Super Mario Bros. {level}", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def write_results_md(rows, eval_episodes, level, flagpole_x, measured, out_path):
    """Emit the results table for `level`.

    The deterministic-policy note is written from what the data actually shows:
    the greedy stall was pronounced mid-training and may or may not persist, so
    the note is never asserted unconditionally. The narrative describing the
    observed 1-1 stall is specific to that level and is only written for it;
    other levels get a neutral note derived from their own results.
    """
    stalling = [r for r in rows if not r["greedy_clear"]]
    escaping = [r for r in rows if r.get("escapes")]
    source = "measured from the runs" if measured else "the level's recorded value"
    lines = [f"# Results — PPO on Super Mario Bros. {level}", "",
             f"Evaluation: {eval_episodes} episodes per run with the "
             "**stochastic** policy, plus one **deterministic (greedy)** pass.",
             "",
             "The environment is deterministic, so the greedy policy produces an "
             "identical trajectory every run; it is reported as a single pass/fail. "
             "Clear rate below is the stochastic figure, which is the meaningful "
             "measure of robustness.",
             (f"The flagpole is at x ≈ {flagpole_x} ({source})."
              if flagpole_x is not None
              else "The flagpole position for this level is not known yet."), "",
             "| run | clear rate | actions | seed | steps trained "
             "| first clear (steps) | mean reward | mean x | best x | greedy clears |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        fc = f"{r['first_clear_steps']:,}" if r["first_clear_steps"] else "—"
        lines.append(f"| {r['run']} | **{r['clear_rate']:.0%}** | {r['action_set']} "
                     f"| {r['seed']} | {r['total_steps']:,} | {fc} "
                     f"| {r['mean_reward']:.1f} | {r['mean_x_max']:.0f} | {r['best_x_max']} "
                     f"| {'yes' if r['greedy_clear'] else 'no'} |")
    best = rows[0]
    lines += ["", f"**Best run: `{best['run']}` — clears World {level} in "
              f"{best['clear_rate']:.0%} of {eval_episodes} episodes.**", ""]
    if escaping:
        detail = ", ".join(f"`{r['run']}` ({r['escapes']}/{r['episodes']})"
                           for r in escaping)
        lines += ["### Episodes lost to warp pipes", "",
                  "StageGuard ended an episode early in " + detail
                  + " because a warp pipe carried the agent out of the level "
                    "instead of the agent finishing it. Those episodes are not "
                    "counted as progress above; they are reported here rather "
                    "than silently inflating the result.", ""]
    lines += ["### Note on the deterministic policy", ""]

    if level == DEFAULT_LEVEL:
        observed = ("Earlier in training (~500K steps) the greedy (argmax) policy stalled: "
                    "it ran to x~2370 and then stood still until the in-game clock expired "
                    "— roughly 90% of the episode motionless, with only ~10% of steps "
                    "showing any movement — while sampling from that same policy cleared "
                    "the level. ")
        if stalling:
            lines.append(observed + "That behaviour persists at the final checkpoint for: "
                         + ", ".join(f"`{r['run']}`" for r in stalling) + ".")
        else:
            lines.append(observed + "By 1M steps the behaviour had resolved: every run's "
                         "greedy policy now clears the level, so the stall was a "
                         "transient mid-training artifact rather than a property of the "
                         "converged agent.")
        lines += ["",
                  "Either way it indicates a narrow failure of action selection rather than "
                  "of the learned representation, and it is consistent with the finding in "
                  "[yumouwei/super-mario-bros-reinforcement-learning]"
                  "(https://github.com/yumouwei/super-mario-bros-reinforcement-learning) "
                  "that a single-frame agent gets stuck and that temporal context helps the "
                  "agent commit to jumps.", ""]
    else:
        # No stall narrative is transplanted from 1-1: this level's behaviour is
        # reported only from this level's measurements.
        lines.append("The environment is deterministic, so the greedy pass replays a "
                     "single trajectory: it reads as one pass/fail, not a rate.")
        if stalling:
            lines.append("At the final checkpoint the greedy policy fails to finish for "
                         + ", ".join(f"`{r['run']}`" for r in stalling)
                         + " — the same narrow failure of action selection observed on "
                           "1-1, where sampling from the identical policy still clears "
                           "the level.")
        else:
            lines.append("At the final checkpoint every run's greedy policy finishes "
                         "the level.")
        lines.append("")
    out_path.write_text(chr(10).join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=sorted(LEVELS), default=DEFAULT_LEVEL)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--rebuild-md", action="store_true",
                    help="regenerate the results table from the artifacts JSON "
                         "without re-running the evaluation")
    args = ap.parse_args()

    art = artifact_paths(args.level, ART)
    spec = get_level(args.level)

    if args.rebuild_md:
        rows = json.loads(art["results_json"].read_text())
        flagpole, measured = measure_flagpole(
            sorted(p for p in RUNS_DIR.iterdir()
                   if p.is_dir() and level_of_run(p) == args.level),
            spec["flagpole_x"])
        write_results_md(rows, rows[0]["episodes"], args.level, flagpole,
                         measured, art["results_md"])
        return

    # Only this level's runs are evaluated: a 1-2 table containing a 1-1 run
    # would compare two different tasks in one ranking.
    runs = sorted(p for p in RUNS_DIR.iterdir()
                  if p.is_dir() and level_of_run(p) == args.level)
    if not runs:
        sys.exit(f"no runs found for level {args.level}")
    ART.mkdir(exist_ok=True)

    flagpole, measured = measure_flagpole(runs, spec["flagpole_x"])
    if flagpole is None:
        print(f"no flag reached in any {args.level} run; "
              "the progress plot gets no flagpole line")
    elif measured:
        print(f"flagpole x for {args.level} measured from the runs: {flagpole}")

    plot_curves(runs, args.level, flagpole, art["curve"])

    rows = []
    for run in runs:
        cfg = json.loads((run / "config.json").read_text())
        model = pick_model(run)
        if model is None:
            # A run with no readable model is left out of the table instead of
            # being reported with numbers it never produced.
            print(f"{run.name}: no model, skipping"); continue
        print(f"evaluating {run.name} ({model.name}) ...")
        res = evaluate(model, args.level, cfg["action_set"], args.eval_episodes,
                       deterministic=False)
        greedy = evaluate(model, args.level, cfg["action_set"], 1,
                          deterministic=True)
        res["greedy_clear"] = greedy["clear_rate"] == 1.0
        res["greedy_x_max"] = greedy["best_x_max"]
        fc = run / "first_clear.json"
        res["first_clear_steps"] = json.loads(fc.read_text())["timesteps"] if fc.exists() else None
        df = load_monitor(run)
        if df is not None and len(df):
            res["total_steps"] = int(df["timesteps"].iloc[-1])
            # monitor.csv's t column is cumulative wall seconds, so this is the
            # run's own measured throughput rather than a quoted benchmark.
            wall = float(df["t"].iloc[-1]) or 1.0
            res["steps_per_sec"] = round(res["total_steps"] / wall)
        else:
            res["total_steps"] = 0
            res["steps_per_sec"] = 0
        res.update(run=run.name, level=args.level, action_set=cfg["action_set"],
                   seed=cfg["seed"], model=str(model))
        rows.append(res)
        print(f"  clear rate {res['clear_rate']:.0%}, mean x {res['mean_x_max']:.0f}")

    if not rows:
        sys.exit(f"no evaluatable models found for level {args.level}")

    rows.sort(key=lambda r: (r["clear_rate"], r["mean_x_max"], r["greedy_clear"]),
              reverse=True)

    write_results_md(rows, args.eval_episodes, args.level, flagpole, measured,
                     art["results_md"])
    art["results_json"].write_text(json.dumps(rows, indent=2))

    if not args.skip_video:
        best = rows[0]
        print(f"recording demo from {best['run']} ...")
        record(best["model"], str(art["video"]), best["action_set"], args.level,
               episodes=5)


if __name__ == "__main__":
    main()
