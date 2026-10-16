"""Environment construction for the Super Mario Bros. PPO agent.

Wrapper order matters and is deliberate:

    JoypadSpace     Discrete(256) raw NES buttons -> 5, 7 or 12 meaningful actions
    StageGuard      end the episode if a warp pipe takes the agent out of the level
    SkipFrame       act once every 4 frames, max-pool the last 2 (NES sprite flicker)
    RewardShaping   scale built-in reward by 1/10, add a flag bonus
    Resize          84x84, the standard Nature-CNN input
    Grayscale       colour carries no task information here (keeps a channel dim)
    (VecFrameStack) applied on the vectorised env -> (84, 84, 4)

Frame-stacking is applied at the vector level by SB3, not here, so the
per-env observation stays (84, 84, 1).

The level is a parameter (see levels.py); nothing here is specific to 1-1.
"""

import gymnasium as gym
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import (
    COMPLEX_MOVEMENT,
    RIGHT_ONLY,
    SIMPLE_MOVEMENT,
)
from nes_py.wrappers import JoypadSpace

from levels import DEFAULT_LEVEL, get_level

ACTION_SETS = {
    "simple": SIMPLE_MOVEMENT,
    "right_only": RIGHT_ONLY,
    "complex": COMPLEX_MOVEMENT,
}


class SkipFrame(gym.Wrapper):
    """Repeat an action for `skip` frames, returning the max of the last two.

    Max-pooling exists because the NES flickers sprites across alternate
    frames; a single sampled frame can be missing an enemy entirely.
    """

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        frames = []
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            frames.append(obs)
            if terminated or truncated:
                break
        obs = np.max(np.stack(frames[-2:]), axis=0) if len(frames) > 1 else frames[-1]
        return obs, total_reward, terminated, truncated, info


class StageGuard(gym.Wrapper):
    """End the episode if the agent leaves the target stage.

    Levels containing warp pipes let the agent travel into another world. The
    single-stage environment does not end the episode on such a jump, so an
    escaping agent could keep playing, reach *another* level's flagpole, and be
    recorded as having cleared this one. That would be a false positive in the
    headline metric, so the episode is stopped the moment the game's own
    world/stage registers leave the target, and the event is reported through
    `info["left_stage"]` so escapes can be counted rather than hidden.

    Only world and stage are compared: they stay fixed while the agent moves
    between areas of the same level (1-1's underground coin room, for example),
    so legitimate pipe use inside a level is not mistaken for an escape.
    """

    def __init__(self, env, target):
        super().__init__(env)
        self.target = tuple(target)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # `left_stage` is registered in train.py's VecMonitor info_keywords, and
        # VecMonitor reads every registered keyword on every step, so the key
        # must exist even when no escape has happened.
        info["left_stage"] = False
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        world, stage = info.get("world"), info.get("stage")
        # world/stage read as 0 during transient screens; those are not escapes
        left_stage = bool(world and stage and (world, stage) != self.target)
        if left_stage:
            terminated = True
        info["left_stage"] = left_stage
        return obs, reward, terminated, truncated, info


class RewardShaping(gym.Wrapper):
    """Scale the built-in reward and add a terminal bonus for the flagpole.

    The built-in reward (x-velocity + clock penalty - death penalty) already
    points the right way; it is only rescaled so PPO's value loss does not
    dominate the policy loss. The flag bonus is small in gradient terms -- it
    is there so the reward curve visibly steps up once the level is solved.
    """

    def __init__(self, env, scale=10.0, flag_bonus=50.0):
        super().__init__(env)
        self.scale = scale
        self.flag_bonus = flag_bonus

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward) / self.scale
        if info.get("flag_get", False):
            reward += self.flag_bonus / self.scale
        return obs, reward, terminated, truncated, info


def make_mario_env(level=DEFAULT_LEVEL, action_set=None, render_mode=None,
                   seed=None):
    """Build a single fully-wrapped Mario environment for `level`.

    `action_set` defaults to the level's configured set. Passing an unknown
    level exits with a readable message rather than silently falling back to
    1-1, because a silent fallback would train the wrong level for hours.
    """
    spec = get_level(level)
    if action_set is None:
        action_set = spec["action_set"]
    env = gym_super_mario_bros.make(spec["env_id"], render_mode=render_mode)
    env = JoypadSpace(env, ACTION_SETS[action_set])
    env = StageGuard(env, spec["target"])
    env = SkipFrame(env, skip=4)
    env = RewardShaping(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
    if seed is not None:
        env.reset(seed=seed)
    return env
