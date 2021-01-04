# -*- coding: utf-8 -*-
"""Core module."""
from abc import ABC, abstractmethod
from typing import Any

import gym

from src.helpers.gym import space_size
from src.types import AgentObservation


class Agent(ABC):
    """An RL agent."""

    def __init__(
        self, observation_space: gym.Space, action_space: gym.Space, **_kwargs
    ):
        """
        Initialize.

        :param observation_space: the observation space.
        :param observation_space: the action space.
        :param kwargs: the algorithm arguments.
        """
        self.observation_space = observation_space
        self.action_space = action_space

        self.observation_space_size = space_size(self.observation_space)
        self.action_space_size = space_size(self.action_space)

    @abstractmethod
    def choose_action(self, state: Any):
        """Chose an action."""

    def on_training_begin(self, **kwargs) -> None:
        """On training begin event."""

    def on_training_end(self, **kwargs) -> None:
        """On training begin event."""

    def on_episode_begin(self, episode, **kwargs) -> None:
        """On episode begin event."""

    def on_episode_end(self, episode, **kwargs) -> None:
        """On episode end event."""

    def on_step_begin(self, step, action, **kwargs) -> None:
        """On step begin event."""

    def on_step_end(self, step, agent_observation: AgentObservation, **kwargs) -> None:
        """On step end event."""
