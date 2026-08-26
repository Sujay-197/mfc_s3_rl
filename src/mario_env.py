"""Environment construction for the Super Mario Bros. PPO agent.

Wrapper order matters and is deliberate:

    JoypadSpace     Discrete(256) raw NES buttons -> 5 or 7 meaningful actions
    SkipFrame       act once every 4 frames, max-pool the last 2 (NES sprite flicker)
    RewardShaping   scale built-in reward by 1/10, add a flag bonus
    Resize          84x84, the standard Nature-CNN input
    Grayscale       colour carries no task information here (keeps a channel dim)
    (VecFrameStack) applied on the vectorised env -> (84, 84, 4)

Frame-stacking is applied at the vector level by SB3, not here, so the
per-env observation stays (84, 84, 1).
"""

import gymnasium as gym
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

ACTION_SETS = {"simple": SIMPLE_MOVEMENT, "right_only": RIGHT_ONLY}


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


def make_mario_env(action_set="simple", render_mode=None, seed=None):
    """Build a single fully-wrapped Mario environment."""
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode=render_mode)
    env = JoypadSpace(env, ACTION_SETS[action_set])
    env = SkipFrame(env, skip=4)
    env = RewardShaping(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
    if seed is not None:
        env.reset(seed=seed)
    return env
