"""RotMAB experiments."""

import gym
from gym.wrappers import TimeLimit

from src import NonMarkovianRotatingMAB
from src.envs.driving_agent import NonMarkovianDrivingAgentEnv


def make_rotmab_env(winning_probs, max_episode_steps) -> gym.Env:
    """Make environment."""
    return TimeLimit(
        NonMarkovianRotatingMAB(winning_probs=winning_probs),
        max_episode_steps=max_episode_steps,
    )


def make_driving_agent_env(max_episode_steps: int) -> gym.Env:
    """Make DrivingAgent environment."""
    return TimeLimit(
        NonMarkovianDrivingAgentEnv(),
        max_episode_steps=max_episode_steps,
    )
