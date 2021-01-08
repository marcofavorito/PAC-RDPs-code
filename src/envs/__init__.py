# -*- coding: utf-8 -*-
"""This package contains Gym environments for Non-Narkovian RL."""

import gym

from .malfunction_mab import *  # noqa: ignore
from .rotating_mab import *  # noqa: ignore
from .sequential_mab import *  # noqa: ignore

gym.register(
    id="RotatingMAB-v0",
    entry_point="src:RotatingMAB",
    max_episode_steps=100,
)

gym.register(
    id="NonMarkovianRotatingMAB-v0",
    entry_point="src:NonMarkovianRotatingMAB",
    max_episode_steps=100,
)

gym.register(
    id="MalfunctionMAB-v0",
    entry_point="src:RotatingMAB",
    max_episode_steps=100,
)

gym.register(
    id="NonMarkovianMalfunctionMAB-v0",
    entry_point="src:NonMarkovianMalfunctionMAB",
    max_episode_steps=100,
)

gym.register(
    id="SequentialMAB-v0", entry_point="src:SequentialMAB", max_episode_steps=100
)


gym.register(
    id="NonMarkovianSequentialMAB-v0",
    entry_point="src:NonMarkovianSequentialMAB",
    max_episode_steps=100,
)
