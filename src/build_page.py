"""Build the standalone results page for a level.

Inlines the demo video and training curve as base64 data URIs so the page is a
single self-contained file that opens anywhere, with no server and no relative
asset paths.

    python src/build_page.py                # 1-1 -> artifacts/page.html
    python src/build_page.py --level 1-2    # 1-2 -> artifacts/page_1_2.html

Every headline figure is substituted from that level's results JSON, so the page
reports what the run actually did rather than numbers typed into the template.
"""

import base64, json, pathlib, argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from levels import DEFAULT_LEVEL, LEVELS, artifact_paths, get_level

ART = pathlib.Path("artifacts")


def headline(rows, level, flagpole):
    """Return the page's headline figures, taken from `rows` (best run first).

    Throughput is summed over the runs because they trained concurrently, which
    is the number that describes the machine. An unknown flagpole position is
    reported as unknown rather than replaced with the run's best x, which would
    read as a target the agent had been measured against.
    """
    best = rows[0]
    first = best.get("first_clear_steps")
    return {
        "__LEVEL__": level,
        "__RUNS__": str(len(rows)),
        "__CLEARRATE__": f"{best['clear_rate']:.0%}",
        "__EPISODES__": str(best["episodes"]),
        "__REWARD__": f"{best['mean_reward']:.1f}",
        "__XMAX__": str(best["best_x_max"]),
        "__FLAGPOLE__": str(flagpole) if flagpole is not None else "unknown",
        "__STEPS__": f"{best['total_steps']:,}",
        "__STEPSM__": f"{best['total_steps'] / 1e6:.1f}M",
        "__SPS__": f"{sum(r.get('steps_per_sec', 0) for r in rows):,}",
        "__FIRSTCLEAR__": f"{first:,}" if first else "never",
        "__ESCAPES__": str(sum(r.get("escapes", 0) for r in rows)),
        "__RESULT__": "FLAG REACHED" if best["clear_rate"] > 0 else "NOT YET CLEARED",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=sorted(LEVELS), default=DEFAULT_LEVEL)
    args = ap.parse_args()

    art = artifact_paths(args.level, ART)
    for path in (art["template"], art["results_json"], art["video"], art["curve"]):
        if not path.exists():
            raise SystemExit(f"missing {path} -- run src/report.py --level "
                             f"{args.level} first")

    rows = json.loads(art["results_json"].read_text(encoding="utf-8"))
    values = headline(rows, args.level, get_level(args.level)["flagpole_x"])

    html = art["template"].read_text(encoding="utf-8")
    for token, value in values.items():
        html = html.replace(token, value)
    html = html.replace("__VID__", base64.b64encode(art["video"].read_bytes()).decode())
    html = html.replace("__PNG__", base64.b64encode(art["curve"].read_bytes()).decode())

    leftover = [t for t in list(values) + ["__VID__", "__PNG__"] if t in html]
    if leftover:
        raise SystemExit(f"placeholder substitution failed: {', '.join(leftover)}")

    art["page"].write_text(html, encoding="utf-8")
    print(f"wrote {art['page']} ({round(len(html) / 1024)} KB)")


if __name__ == "__main__":
    main()
