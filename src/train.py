"""PPO training for Super Mario Bros. 1-1.

One process trains one seed. Three of these run in parallel overnight
(see run_all.py); each gets its own subdirectory under runs/.

Design notes:
  * No eval environment. NES emulation is the bottleneck, and a separate
    eval env would steal CPU from the rollout workers all night. "Best"
    is tracked from the rolling mean of training episode reward, which is
    free, and a rigorous evaluation is run in the morning by report.py.
  * gamma=0.9 is low by Atari standards but standard for Mario: with a
    frame-skip of 4 the useful credit-assignment horizon is short, and
    higher discounts make the value function much harder to fit here.
  * Checkpoints are frequent and cheap; a late policy collapse must never
    cost us a working agent.
"""

import sys, os, time, argparse, json, pathlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from mario_env import make_mario_env

INFO_KEYWORDS = ("flag_get", "x_pos_max")


class BestModelCallback(BaseCallback):
    """Save the model whenever rolling mean episode reward hits a new high.

    Insurance against late-training policy collapse: whatever happens by
    morning, the best policy the run ever reached is on disk.
    """

    def __init__(self, save_path, window=40, min_episodes=40, check_freq=1000):
        super().__init__()
        self.save_path = save_path
        self.window = window
        self.min_episodes = min_episodes
        self.check_freq = check_freq  # in rollout steps, not timesteps
        self.best = -np.inf

    def _on_step(self):
        if self.n_calls % self.check_freq:
            return True
        buf = self.model.ep_info_buffer
        if buf is None or len(buf) < self.min_episodes:
            return True
        mean_r = float(np.mean([ep["r"] for ep in list(buf)[-self.window:]]))
        if mean_r > self.best:
            self.best = mean_r
            # Save atomically: a reader (report.py/status) must never observe
            # a half-written zip, which raises EOFError on load. The temp name
            # must not carry its own suffix -- SB3 only appends ".zip" when the
            # path has none, so "best_model.tmp" would be written verbatim.
            # A bookkeeping failure must never end the run, hence the guard.
            try:
                tmp = self.save_path + "_tmp"
                self.model.save(tmp)
                os.replace(tmp + ".zip", self.save_path + ".zip")
            except Exception as exc:
                print(f"[warn] best-model save failed, continuing: {exc!r}",
                      flush=True)
            self.logger.record("train/best_mean_reward", self.best)
        return True


class ProgressCallback(BaseCallback):
    """Log flag rate and level progress, and write a status file for status.py."""

    def __init__(self, run_dir, log_freq=500):
        super().__init__()
        self.run_dir = pathlib.Path(run_dir)
        self.log_freq = log_freq  # in rollout steps: x n_envs for timesteps
        self.t0 = time.time()
        self.first_clear_steps = None

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if info.get("flag_get") and self.first_clear_steps is None:
                self.first_clear_steps = self.num_timesteps
                try:
                    (self.run_dir / "first_clear.json").write_text(
                        json.dumps({"timesteps": self.num_timesteps,
                                    "wall_seconds": round(time.time() - self.t0, 1)}))
                except Exception as exc:
                    print(f"[warn] first_clear write failed: {exc!r}", flush=True)
        if self.n_calls % self.log_freq:
            return True
        buf = list(self.model.ep_info_buffer or [])[-100:]
        if buf:
            flag_rate = float(np.mean([ep.get("flag_get", 0) for ep in buf]))
            x_max = float(np.mean([ep.get("x_pos_max", 0) for ep in buf]))
            mean_r = float(np.mean([ep["r"] for ep in buf]))
            self.logger.record("mario/flag_rate_100", flag_rate)
            self.logger.record("mario/x_pos_max_100", x_max)
            elapsed = time.time() - self.t0
            self._write_status({
                "timesteps": self.num_timesteps,
                "elapsed_min": round(elapsed / 60, 1),
                "steps_per_sec": round(self.num_timesteps / max(elapsed, 1e-9), 1),
                "mean_reward_100": round(mean_r, 2),
                "flag_rate_100": round(flag_rate, 3),
                "mean_x_pos_max_100": round(x_max, 1),
                "first_clear_steps": self.first_clear_steps,
                "updated": time.strftime("%H:%M:%S"),
            })
        return True

    def _write_status(self, payload):
        """Status reporting is diagnostics; never let it kill training."""
        try:
            (self.run_dir / "status.json").write_text(json.dumps(payload))
        except Exception as exc:
            print(f"[warn] status write failed: {exc!r}", flush=True)


class WallClockCallback(BaseCallback):
    """Hard stop so the run ends on its own, well before the deadline."""

    def __init__(self, max_hours):
        super().__init__()
        self.deadline = time.time() + max_hours * 3600

    def _on_step(self):
        return time.time() < self.deadline


def keep_system_awake():
    """Ask Windows not to sleep while this process runs.

    A process-scoped request (SetThreadExecutionState), not a change to the
    power scheme: it lapses automatically when training exits, so the
    machine's normal sleep behaviour is left untouched. Without this the
    default 60-minute AC idle timeout would end the run overnight.
    """
    if sys.platform != "win32":
        return
    import ctypes
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action-set", choices=["simple", "right_only"], default="simple")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--total-steps", type=int, default=20_000_000)
    ap.add_argument("--max-hours", type=float, default=7.5)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    keep_system_awake()

    run_name = args.run_name or f"{args.action_set}_seed{args.seed}"
    run_dir = pathlib.Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    venv = SubprocVecEnv(
        [(lambda: make_mario_env(args.action_set)) for _ in range(args.n_envs)],
        start_method="spawn")
    venv = VecMonitor(venv, filename=str(run_dir / "monitor.csv"),
                      info_keywords=INFO_KEYWORDS)
    venv = VecFrameStack(venv, 4)
    venv.seed(args.seed)

    model = PPO(
        "CnnPolicy", venv,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.9,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
        device="cuda",
    )

    callbacks = [
        BestModelCallback(str(run_dir / "best_model")),
        CheckpointCallback(save_freq=max(50_000 // args.n_envs, 1),
                           save_path=str(run_dir / "checkpoints"),
                           name_prefix="ppo_mario"),
        ProgressCallback(run_dir),
        WallClockCallback(args.max_hours),
    ]
    try:
        model.learn(total_timesteps=args.total_steps, callback=callbacks,
                    progress_bar=False)
    finally:
        model.save(str(run_dir / "final_model"))
        venv.close()
        print(f"[{run_name}] saved final model")


if __name__ == "__main__":
    main()
