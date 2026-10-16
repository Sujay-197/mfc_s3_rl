"""One-glance health check across all overnight runs.

Prints what actually matters at 3am: is each run alive, is it stepping at
the expected rate, is reward trending up, has anything reached the flag,
and when was the last checkpoint written.
"""

import json, pathlib, time, sys

from levels import DEFAULT_LEVEL, get_level, level_of_run

EXPECTED_SPS = 300  # per run with 3 concurrent runs; below this = contention


def fmt_age(path):
    if not path.exists():
        return "none"
    age = time.time() - path.stat().st_mtime
    return f"{age/60:.1f}m ago" if age < 3600 else f"{age/3600:.1f}h ago"


def main():
    runs = sorted(p for p in pathlib.Path("runs").iterdir()
                  if p.is_dir() and p.name != "smoketest")
    if not runs:
        print("no runs found"); return

    print(f"{'run':<20}{'lvl':>5}{'steps':>11}{'sp/s':>7}{'rew':>9}{'x_max':>8}"
          f"{'flag%':>7}{'first clear':>14}{'ckpt':>10}")
    print("-" * 91)
    for run in runs:
        lvl = level_of_run(run) or DEFAULT_LEVEL
        sf = run / "status.json"
        if not sf.exists():
            print(f"{run.name:<20}{lvl:>5}{'(no status yet)':>11}")
            continue
        s = json.loads(sf.read_text())
        ckpts = list((run / "checkpoints").glob("*.zip")) if (run / "checkpoints").exists() else []
        latest = max(ckpts, key=lambda p: p.stat().st_mtime) if ckpts else run / "__none__"
        fc = s.get("first_clear_steps")
        stale = (time.time() - sf.stat().st_mtime) > 300
        flag = "  STALE" if stale else ""
        print(f"{run.name:<20}{lvl:>5}{s['timesteps']:>11,}{s['steps_per_sec']:>7.0f}"
              f"{s['mean_reward_100']:>9.1f}{s['mean_x_pos_max_100']:>8.0f}"
              f"{s['flag_rate_100']*100:>6.0f}%"
              f"{(f'{fc:,}' if fc else '-'):>14}{fmt_age(latest):>10}{flag}")

    print("\nhealth: steps/s well under "
          f"{EXPECTED_SPS} or a STALE marker means a run needs attention.")
    for level in sorted({level_of_run(r) or DEFAULT_LEVEL for r in runs}):
        spec = get_level(level)
        target = (f"x_max ~{spec['flagpole_x']} is the flagpole"
                  if spec["flagpole_x"] is not None
                  else "flagpole x not recorded; report.py measures it from the run")
        print(f"{level}: {target} ({spec['notes']})")


if __name__ == "__main__":
    main()
