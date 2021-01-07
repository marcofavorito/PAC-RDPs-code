"""RotMAB experiments."""

import gym
from gym.wrappers import TimeLimit

from src import NonMarkovianRotatingMAB


def make_rotmab_env(winning_probs, max_episode_steps) -> gym.Env:
    """Make environment."""
    return TimeLimit(
        NonMarkovianRotatingMAB(winning_probs=winning_probs),
        max_episode_steps=max_episode_steps,
    )
