# -*- coding: utf-8 -*-
"""Base class for RL agents."""
from functools import partial
from typing import Any, Callable, Collection, Optional

import gym
import numpy as np
from gym.wrappers import TimeLimit

from src.callbacks.base import BaseCallback, Callback, CallbackList
from src.core import Agent
from src.types import AgentObservation

Policy = Callable[[Agent, Any], int]


def greedy_policy(agent: Agent, state: Any):
    """Greedy policy."""
    return agent.choose_action(state)


def random_policy(agent: Agent, _state: Any):
    """Random actions."""
    return agent.action_space.sample()


def eps_greedy_policy(agent: Agent, state: Any, epsilon: float = 0.1):
    """Epsilon-greedy policy."""
    if np.random.random() < epsilon:
        return agent.action_space.sample()
    else:
        return agent.choose_action(state)


def make_eps_greedy_policy(epsilon: float = 0.1) -> Policy:
    """Make epsilon-greedy policy."""
    return partial(eps_greedy_policy, epsilon=epsilon)


class Context(BaseCallback):
    """Training context."""

    def __init__(
        self,
        experiment_name: str,
        agent: Agent,
        env: gym.Env,
        policy: Optional[Policy] = None,
        callbacks: Collection[Callback] = (),
    ):
        """Initialize."""
        self.experiment_name = experiment_name
        self.agent: Agent = agent
        self.env: gym.Env = env
        self.policy: Policy = policy or make_eps_greedy_policy()
        self.callback_list = CallbackList(callbacks)

    def on_training_begin(self, **kwargs) -> None:
        """On training begin."""
        self.agent.on_training_begin()
        self.callback_list.set_agent(self.agent)
        self.callback_list.set_environment(self.env)
        self.callback_list.set_experiment_name(self.experiment_name)
        self.callback_list.on_training_begin()

    def on_training_end(self, **kwargs) -> None:
        """On training end."""
        self.agent.on_training_end()
        self.callback_list.on_training_end()
        self.callback_list.set_agent(None)
        self.callback_list.set_environment(None)
        self.callback_list.set_experiment_name(None)

    def on_episode_begin(self, episode, **kwargs) -> None:
        """On episode begin."""
        self.agent.on_episode_begin(episode)
        self.callback_list.on_episode_begin(episode)

    def on_episode_end(self, episode, **kwargs) -> None:
        """On episode end."""
        self.agent.on_episode_end(episode)
        self.callback_list.on_episode_end(episode)

    def on_step_begin(self, step, action, **kwargs) -> None:
        """On step begin."""
        self.agent.on_step_begin(step, action)
        self.callback_list.on_step_begin(step, action)

    def on_step_end(self, step, agent_observation: AgentObservation, **kwargs) -> None:
        """On step end."""
        self.agent.on_step_end(step, agent_observation)
        self.callback_list.on_step_end(step, agent_observation)


def train(
    agent: Agent,
    env: gym.Env,
    policy: Optional[Policy] = None,
    callbacks: Collection[Callback] = (),
    nb_episodes: int = 1000,
    experiment_name: str = "",
):
    """Train."""
    policy = policy or make_eps_greedy_policy()
    context = Context(experiment_name, agent, env, policy, callbacks)
    context.on_training_begin()
    for episode in range(nb_episodes):
        state = env.reset()
        done = False
        step = 0
        context.on_episode_begin(episode)
        while not done:
            action = policy(agent, state)
            context.on_step_begin(step, action)
            state2, reward, done, info = env.step(action)
            context.on_step_end(step, (state, action, reward, state2, done))
            state = state2
            step += 1
        context.on_episode_end(episode)
    context.on_training_end()
    env.close()


def test(
    agent: Agent,
    env: gym.Env,
    nb_episodes: int = 1,
    max_steps: Optional[int] = None,
    policy: Policy = greedy_policy,
):
    """
    Do a test.

    :param agent: an agent.
    :param env: the OpenAI Gym environment.
    :param nb_episodes: the number of test episodes.
    :param max_steps: maximum number of steps per episodes.
    :param policy: a callable that takes the environment and the state and returns the action.
    :return: None
    """
    if max_steps:
        env = TimeLimit(env, max_episode_steps=max_steps)
    for _ in range(nb_episodes):
        state = env.reset()
        done = False
        while not done:
            action = policy(agent, state)
            state, reward, done, info = env.step(action)
