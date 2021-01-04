# -*- coding: utf-8 -*-
"""Utilities for the OpenAI Gym wrappers."""
from abc import ABC
from typing import Collection, Optional

import gym

from src.core import Agent
from src.types import AgentObservation


class BaseCallback(ABC):
    """
    The callback interface.

    It can listen to the following events:
    - on_training_begin
    - on_training_end
    - on_episode_begin
    - on_episode_end
    - on_step_begin
    - on_step_end
    """

    def on_training_begin(self, **kwargs) -> None:
        """On training begin event."""

    def on_training_end(self, **kwargs) -> None:
        """On training end event."""

    def on_episode_begin(self, episode, **kwargs) -> None:
        """On episode begin event."""

    def on_episode_end(self, episode, **kwargs) -> None:
        """On episode end event."""

    def on_step_begin(self, step, action, **kwargs) -> None:
        """On step begin event."""

    def on_step_end(self, step, agent_observation: AgentObservation, **kwargs) -> None:
        """On step end event."""


class Callback(BaseCallback, ABC):
    """Abstract class for callbacks."""

    agent: Optional[Agent] = None
    env: Optional[gym.Env] = None
    experiment_name: Optional[str] = None


class CallbackList(BaseCallback):
    """Callback list."""

    def __init__(self, callbacks: Collection[Callback]):
        """
        Initialize the callback list.

        :param callbacks: the list of callbacks.
        """
        self.callbacks = callbacks

    def _call_method(self, method_name: str, *args, **kwargs):
        """Call a method to all callbacks."""
        for callback in self.callbacks:
            method = getattr(callback, method_name)
            method(*args, **kwargs)

    def set_agent(self, agent: Optional[Agent]):
        """Set the agent to every callback."""
        for c in self.callbacks:
            c.agent = agent

    def set_environment(self, env: Optional[gym.Env]):
        """Set the environment to every callback."""
        for c in self.callbacks:
            c.env = env

    def set_experiment_name(self, experiment_name: Optional[str]):
        """Set the environment to every callback."""
        for c in self.callbacks:
            c.experiment_name = experiment_name

    def on_training_begin(self, **kwargs) -> None:
        """On training begin event."""
        self._call_method("on_training_begin", **kwargs)

    def on_training_end(self, **kwargs) -> None:
        """On training end event."""
        self._call_method("on_training_end", **kwargs)

    def on_episode_begin(self, episode, **kwargs) -> None:
        """On episode begin event."""
        self._call_method("on_episode_begin", episode, **kwargs)

    def on_episode_end(self, episode, **kwargs) -> None:
        """On episode end event."""
        self._call_method("on_episode_end", episode, **kwargs)

    def on_step_begin(self, step, action, **kwargs) -> None:
        """On step begin event."""
        self._call_method("on_step_begin", step, action, **kwargs)

    def on_step_end(self, step, agent_observation: AgentObservation, **kwargs) -> None:
        """On step end event."""
        self._call_method("on_step_end", step, agent_observation, **kwargs)
