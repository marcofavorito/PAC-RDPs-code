# -*- coding: utf-8 -*-
"""Core module."""
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Collection

import gym
import numpy as np

from src.callbacks.base import BaseCallback, Callback, CallbackList
from src.helpers.gym import space_size
from src.types import AgentObservation

Policy = Callable[["Agent", Any], int]


def greedy_policy(agent: "Agent", state: Any):
    """Greedy policy."""
    return agent.choose_best_action(state)


def random_policy(agent: "Agent", _state: Any):
    """Random actions."""
    return agent.action_space.sample()


def eps_greedy_policy(agent: "Agent", state: Any, epsilon: float = 0.1):
    """Epsilon-greedy policy."""
    if np.random.random() < epsilon:
        return agent.action_space.sample()
    return agent.choose_best_action(state)


def make_eps_greedy_policy(epsilon: float = 0.1) -> Policy:
    """Make epsilon-greedy policy."""
    return partial(eps_greedy_policy, epsilon=epsilon)


class Agent(ABC):
    """An RL agent."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        policy: Policy = greedy_policy,
        **_kwargs
    ):
        """
        Initialize.

        :param observation_space: the observation space.
        :param observation_space: the action space.
        :param policy: the policy to follow.
        :param kwargs: the algorithm arguments.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.policy = policy

        self.observation_space_size = space_size(self.observation_space)
        self.action_space_size = space_size(self.action_space)

    @abstractmethod
    def choose_best_action(self, state: Any):
        """Chose best action."""

    def take_action(self, state: Any):
        """Take an action."""
        return self.policy(self, state)

    def observe(self, agent_observation: AgentObservation):
        """Observe a transition."""

    def _play(
        self,
        env: gym.Env,
        callbacks: Collection[Callback] = (),
        nb_episodes: int = 1000,
        experiment_name: str = "",
        is_training: bool = False,
    ):
        context = Context(experiment_name, self, env, callbacks)
        context.on_training_begin()
        for episode in range(nb_episodes):
            state = env.reset()
            done = False
            step = 0
            context.on_episode_begin(episode)
            while not done:
                action = (
                    self.take_action(state)
                    if is_training
                    else self.choose_best_action(state)
                )
                context.on_step_begin(step, action)
                state2, reward, done, info = env.step(action)
                observation = (state, action, reward, state2, done)
                if is_training:
                    self.observe(observation)
                context.on_step_end(step, observation)
                state = state2
                step += 1
            context.on_episode_end(episode)
        context.on_training_end()

    def train(
        self,
        env: gym.Env,
        callbacks: Collection[Callback] = (),
        nb_episodes: int = 1000,
        experiment_name: str = "",
    ):
        """
        Train the agent.

        :param env: the environment.
        :param callbacks: list of callbacks.
        :param nb_episodes: number of episodes.
        :param experiment_name: the experiment name
        :return: None
        """
        self._play(
            env,
            callbacks,
            nb_episodes=nb_episodes,
            experiment_name=experiment_name,
            is_training=True,
        )

    def test(
        self,
        env: gym.Env,
        callbacks: Collection[Callback] = (),
        nb_episodes: int = 1000,
        experiment_name: str = "",
    ):
        """
        Test the agent.

        :param env: environment.
        :param callbacks: list of callbacks.
        :param nb_episodes: number of episodes.
        :param experiment_name: the experiment name
        :return: None
        """
        self._play(
            env,
            callbacks,
            nb_episodes=nb_episodes,
            experiment_name=experiment_name,
            is_training=False,
        )


class Context(BaseCallback):
    """Training context."""

    def __init__(
        self,
        experiment_name: str,
        agent: "Agent",
        env: gym.Env,
        callbacks: Collection[Callback] = (),
    ):
        """Initialize."""
        self.experiment_name = experiment_name
        self.agent: "Agent" = agent
        self.env: gym.Env = env
        self.callback_list = CallbackList(callbacks)

        self._old_agent_policy = agent.policy

    def on_training_begin(self, **kwargs) -> None:
        """On training begin."""
        self.callback_list.set_agent(self.agent)
        self.callback_list.set_environment(self.env)
        self.callback_list.set_experiment_name(self.experiment_name)
        self.callback_list.on_training_begin()

    def on_training_end(self, **kwargs) -> None:
        """On training end."""
        self.callback_list.on_training_end()
        self.callback_list.set_agent(None)
        self.callback_list.set_environment(None)
        self.callback_list.set_experiment_name(None)

    def on_episode_begin(self, episode, **kwargs) -> None:
        """On episode begin."""
        self.callback_list.on_episode_begin(episode)

    def on_episode_end(self, episode, **kwargs) -> None:
        """On episode end."""
        self.callback_list.on_episode_end(episode)

    def on_step_begin(self, step, action, **kwargs) -> None:
        """On step begin."""
        self.callback_list.on_step_begin(step, action)

    def on_step_end(self, step, agent_observation: AgentObservation, **kwargs) -> None:
        """On step end."""
        self.callback_list.on_step_end(step, agent_observation)
