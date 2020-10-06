# -*- coding: utf-8 -*-
"""This package contains the implementations of several Multi-Armed Bandit variants."""
import gym

from .rotating_mab import *
from .sequential_mab import *

gym.register(
    id="RotatingMAB-v0", entry_point="nmrl_wa:RotatingMAB", max_episode_steps=100,
)

gym.register(
    id="NonMarkovianRotatingMAB-v0",
    entry_point="nmrl_wa:NonMarkovianRotatingMAB",
    max_episode_steps=100,
)

gym.register(
    id="SequentialMAB-v0", entry_point="nmrl_wa:SequentialMAB", max_episode_steps=100
)


gym.register(
    id="NonMarkovianSequentialMAB-v0",
    entry_point="nmrl_wa:NonMarkovianSequentialMAB",
    max_episode_steps=100,
)
