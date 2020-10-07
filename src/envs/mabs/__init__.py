# -*- coding: utf-8 -*-
"""This package contains the implementations of several Multi-Armed Bandit variants."""
import gym

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
    id="SequentialMAB-v0", entry_point="src:SequentialMAB", max_episode_steps=100
)


gym.register(
    id="NonMarkovianSequentialMAB-v0",
    entry_point="src:NonMarkovianSequentialMAB",
    max_episode_steps=100,
)
