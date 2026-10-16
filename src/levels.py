"""Per-level configuration shared by every stage of the pipeline.

Linear Super Mario Bros. levels are identified by their (world, stage) target.
Everything that differs between two levels lives here rather than being
scattered through the pipeline, so adding a level is a config entry instead of
a hunt for hardcoded numbers.

Coverage: all three of world 1's early stages (1-1, 1-2, 1-3) plus the first
stage of each later world (2-1 through 8-1), so one level from every one of
the game's eight worlds is available to every stage of the pipeline.

`flagpole_x` is the approximate x position of the level's flagpole. It is used
only for the reference line on the progress plot and for the status legend. A
level whose value is None has it measured from the run's own logs instead,
which is more honest than guessing: the value becomes the furthest x reached by
an episode that actually touched the flag.
"""

import json
import pathlib

# the level the project was originally built and reported around
DEFAULT_LEVEL = "1-1"

LEVELS = {
    "1-1": {
        "env_id": "SuperMarioBros-1-1-v0",
        "target": (1, 1),
        "flagpole_x": 3160,
        "action_set": "simple",
        "notes": "above ground; ~700 is the first pipe cluster",
    },
    "1-2": {
        "env_id": "SuperMarioBros-1-2-v0",
        "target": (1, 2),
        # 3161 was measured from the runs by report.py: the furthest x reached
        # by an episode that actually touched the flag. Recorded here so the
        # results table, the curve and the HTML page all report the same number.
        "flagpole_x": 3161,
        # 1-2 ends by walking right into the exit pipe and then holding down.
        # 'simple' has no 'down', so the level cannot be completed with it;
        # 'complex' is the only action set that can enter a pipe.
        "action_set": "complex",
        "notes": ("underground; ends at the exit pipe to an outdoor area, then "
                  "a staircase and flagpole; contains warp pipes into worlds 2-4"),
    },
    "1-3": {
        "env_id": "SuperMarioBros-1-3-v0",
        "target": (1, 3),
        "flagpole_x": None,  # measured from the run's logs
        # 1-3 ends at a flagpole, not an exit pipe, so 'simple' carries every
        # button that matters (right, jump, sprint); the 'down' that forces
        # 1-2 onto 'complex' is not needed here.
        "action_set": "simple",
        # smoke_test.py on this level: guard clean (no misfires, no escapes),
        # but constant-input policies die at the first platform gap (~x 300) --
        # the level needs timed jumps, which is what training is for.
        "notes": "above ground; platform-hopping over pits, ends at the flagpole",
    },
    # One stage from each of the game's worlds 2-8. Every X-1 ends at a
    # flagpole, so 'simple' is sufficient -- no exit pipe needs a 'down' the
    # way 1-2 does. flagpole_x stays None until report.py measures it from a
    # run that actually clears the level; smoke-test each before training.
    "2-1": {
        "env_id": "SuperMarioBros-2-1-v0",
        "target": (2, 1),
        "flagpole_x": None,  # measured from the run's logs
        "action_set": "simple",
        "notes": "first stage of world 2; ends at a flagpole",
    },
    "3-1": {
        "env_id": "SuperMarioBros-3-1-v0",
        "target": (3, 1),
        "flagpole_x": None,  # measured from the run's logs
        "action_set": "simple",
        "notes": "first stage of world 3; ends at a flagpole",
    },
    "4-1": {
        "env_id": "SuperMarioBros-4-1-v0",
        "target": (4, 1),
        "flagpole_x": None,  # measured from the run's logs
        "action_set": "simple",
        "notes": "first stage of world 4; ends at a flagpole",
    },
    "5-1": {
        "env_id": "SuperMarioBros-5-1-v0",
        "target": (5, 1),
        "flagpole_x": None,  # measured from the run's logs
        "action_set": "simple",
        "notes": "first stage of world 5; ends at a flagpole",
    },
    "6-1": {
        "env_id": "SuperMarioBros-6-1-v0",
        "target": (6, 1),
        "flagpole_x": None,  # measured from the run's logs
        "action_set": "simple",
        "notes": "first stage of world 6; ends at a flagpole",
    },
    "7-1": {
        "env_id": "SuperMarioBros-7-1-v0",
        "target": (7, 1),
        "flagpole_x": None,  # measured from the run's logs
        "action_set": "simple",
        "notes": "first stage of world 7; ends at a flagpole",
    },
    "8-1": {
        "env_id": "SuperMarioBros-8-1-v0",
        "target": (8, 1),
        "flagpole_x": None,  # measured from the run's logs
        "action_set": "simple",
        "notes": "first stage of world 8; ends at a flagpole",
    },
}


def get_level(level):
    """Return the configuration for `level`, or exit with a readable message."""
    try:
        return LEVELS[level]
    except KeyError:
        known = ", ".join(sorted(LEVELS))
        raise SystemExit(f"unknown level {level!r}; known levels: {known}")


def level_of_run(run_dir):
    """Return the level a run directory belongs to.

    Runs written before levels were parameterised carry no "level" key in their
    config; every one of those is a 1-1 run, so that is the fallback.
    """
    config = pathlib.Path(run_dir) / "config.json"
    if not config.exists():
        return None
    try:
        return json.loads(config.read_text(encoding="utf-8")).get(
            "level", DEFAULT_LEVEL)
    except (ValueError, OSError):
        return DEFAULT_LEVEL


def artifact_paths(level, art_dir):
    """Return the output paths for `level`.

    1-1 keeps the original unsuffixed filenames so the existing submission is
    left exactly as it is; every other level is suffixed with its own tag, so
    two artifact sets coexist rather than overwriting each other.
    """
    tag = "" if level == DEFAULT_LEVEL else "_" + level.replace("-", "_")
    art = pathlib.Path(art_dir)
    return {
        "results_md": art / f"results{tag}.md",
        "results_json": art / f"results{tag}.json",
        "curve": art / f"training_curve{tag}.png",
        "video": art / f"demo{tag}.mp4",
        "page": art / f"page{tag}.html",
        "template": art / f"page_template{tag}.html",
    }
