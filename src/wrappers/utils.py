# -*- coding: utf-8 -*-
"""Utilities for the OpenAI Gym wrappers."""
import shutil
import time
from pathlib import Path
from typing import Any, Collection, List, Optional

import gym
from gym import Wrapper
from PIL import Image

from src.callbacks.base import Callback, CallbackList


class MyMonitor(Wrapper):
    """A simple monitor."""

    def __init__(self, env: gym.Env, directory: str, force: bool = False):
        """
        Initialize the environment.

        :param env: the environment.
        :param directory: the directory where to save elements.
        """
        super().__init__(env)

        self._directory = Path(directory)
        shutil.rmtree(directory, ignore_errors=force)
        self._directory.mkdir(exist_ok=False)

        self._current_step = 0
        self._current_episode = 0

    def _save_image(self):
        """Save a frame."""
        array = self.render(mode="rgb_array")
        image = Image.fromarray(array)
        episode_dir = f"{self._current_episode:05d}"
        filepath = f"{self._current_step:05d}.jpeg"
        (self._directory / episode_dir).mkdir(parents=True, exist_ok=True)
        image.save(str(self._directory / episode_dir / filepath))

    def reset(self, **kwargs):
        """Reset the environment."""
        result = super().reset(**kwargs)
        self._current_step = 0
        self._current_episode += 1
        self._save_image()
        return result

    def step(self, action):
        """Do a step in the environment, and record the frame."""
        result = super().step(action)
        self._current_step += 1
        self._save_image()
        return result


class StatsRecorder(gym.Wrapper):
    """Stats recorder."""

    def __init__(self, env: gym.Env, prefix: str = ""):
        """
        Initialize stats recorder.

        :param env: the environment to monitor.
        :param prefix: the prefix to add to statistics attributes.
        """
        super().__init__(env)
        self._prefix = prefix
        self._episode_lengths: List[int] = []
        self._episode_rewards: List[float] = []
        self._timestamps: List[float] = []
        self._random_seed: Optional[int] = None
        self._steps = None
        self._total_steps = 0
        self._rewards = None
        self._done = False
        self._set_attributes()

    def _set_attributes(self):
        """Set main attributes with the prefix."""
        setattr(self, self._prefix + "episode_lengths", self._episode_lengths)
        setattr(self, self._prefix + "episode_rewards", self._episode_rewards)
        setattr(self, self._prefix + "total_steps", self._total_steps)
        setattr(self, self._prefix + "timestamps", self._timestamps)
        setattr(self, self._prefix + "random_seed", self._random_seed)

    def step(self, action):
        """Do a step."""
        state, reward, done, info = super().step(action)
        self._steps += 1
        self._total_steps += 1
        self._rewards += reward
        self._done = done
        if done:
            self.save_complete()

        return state, reward, done, info

    def save_complete(self):
        """Save episode statistics."""
        if self._steps is not None:
            self._episode_lengths.append(self._steps)
            self._episode_rewards.append(float(self._rewards))
            self._timestamps.append(time.time())

    def reset(self, **kwargs):
        """Do reset."""
        result = super().reset(**kwargs)
        self._done = False
        self._steps = 0
        self._rewards = 0
        return result

    def seed(self, seed=None):
        """Set seed."""
        super().seed(seed)
        self._random_seed = seed
        setattr(self, self._prefix + "random_seed", self._random_seed)


class CallbackEnv(gym.Wrapper):
    """Call functions during the simulation with the environment."""

    def __init__(self, env: gym.Env, callbacks: Collection[Callback] = ()):
        """Initialize."""
        super().__init__(env)
        self.callback_list = CallbackList(callbacks)
        self.last_state: Optional[Any] = None
        self.episode = -1
        self.step_count = 0

    def reset(self, **kwargs):
        """Reset the environment."""
        result = super().reset()
        self.last_state = result
        self.episode += 1
        self.step_count = 0
        self.callback_list.on_episode_begin(self.episode)
        return result

    def step(self, action):
        """Do a step."""
        self.callback_list.on_step_begin(self.step_count, action)
        next_state, reward, done, info = super().step(action)
        agent_obs = (self.last_state, action, reward, next_state, done)
        self.callback_list.on_step_end(self.step_count, agent_obs)
        self.step_count += 1
        self.last_state = next_state
        return next_state, reward, done, info
