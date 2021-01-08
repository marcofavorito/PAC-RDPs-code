"""RotMAB experiments."""
from typing import List

import gym
from gym.wrappers import TimeLimit

from src import NonMarkovianCheatMAB
from src.envs.driving_agent import NonMarkovianDrivingAgentEnv
from src.envs.malfunction_mab import NonMarkovianMalfunctionMAB
from src.envs.rotating_mab import NonMarkovianRotatingMAB


def make_rotmab_env(winning_probs, max_episode_steps) -> gym.Env:
    """Make environment."""
    return TimeLimit(
        NonMarkovianRotatingMAB(winning_probs=winning_probs),
        max_episode_steps=max_episode_steps,
    )


def make_malfunctionmab_env(
    winning_probs, k: int, malfunctioning_arm: int = 0, max_episode_steps: int = 100
) -> gym.Env:
    """Make environment."""
    return TimeLimit(
        NonMarkovianMalfunctionMAB(
            winning_probs=winning_probs, k=k, malfunctioning_arm=malfunctioning_arm
        ),
        max_episode_steps=max_episode_steps,
    )


def make_cheatmab_env(
    nb_arms: int, pattern: List[int], max_episode_steps: int = 100
) -> gym.Env:
    """Make environment."""
    return TimeLimit(
        NonMarkovianCheatMAB(nb_arms=nb_arms, pattern=pattern),
        max_episode_steps=max_episode_steps,
    )


def make_driving_agent_env(max_episode_steps: int) -> gym.Env:
    """Make DrivingAgent environment."""
    return TimeLimit(
        NonMarkovianDrivingAgentEnv(),
        max_episode_steps=max_episode_steps,
    )
