# -*- coding: utf-8 -*-
"""This module contains the main code for the algorithm."""
from typing import Collection, List, Optional

import gym
import numpy as np
from gym import Space
from pdfa_learning.pdfa import PDFA
from pdfa_learning.types import Word

from src.callbacks.base import Callback
from src.core import Context, random_policy
from src.helpers.gym import DiscreteEnv
from src.pac_rdp.base import BasePacRdpAgent
from src.types import AgentObservation


def stop_probability_from_D(d: int) -> float:
    """
    Compute the stop probability from the depth.

    :param d: upper bound on depth.
    :return: the stop probability.
    """
    return 1 / (10 * d + 1)


class PacRdpAgentSimple(BasePacRdpAgent):
    """Implementation of PAC-RDP agent."""

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        env: DiscreteEnv,
        epsilon: float = 0.1,
        delta: float = 0.1,
        gamma: float = 0.9,
        max_depth: int = 10,
        update_frequency: int = 1,
    ):
        """Initialize the agent."""
        super().__init__(
            observation_space,
            action_space,
            env,
            epsilon,
            delta,
            gamma,
        )
        self.max_depth = max_depth
        self.update_frequency = update_frequency
        self._reset()

    def get_upperbound(self) -> int:
        """Get upperbound on number of states."""
        return self.max_depth

    def get_stop_probability(self) -> float:
        """Get current stop probability."""
        return self.current_p

    def _reset(self):
        """Reset the state of the agent."""
        self.pdfa: Optional[PDFA] = None
        self.dataset: List[Word] = []
        self.current_p = stop_probability_from_D(self.max_depth)
        self._episode_reset()

    def _episode_reset(self):
        """Episode reset."""
        self.current_episode = []
        self.current_state = 0
        self.stop = False

    def done(self) -> bool:
        """Return true when either we should stop or a hard stop happened."""
        return self.stop

    def _should_stop(self) -> bool:
        """Check that next action should be the stop action."""
        return np.random.random() < self.current_p

    def observe(self, agent_observation: AgentObservation):
        """Observe a transition."""
        self.current_episode.append(agent_observation)
        self.stop = self._should_stop()

    def train(
        self,
        env: gym.Env,
        callbacks: Collection[Callback] = (),
        nb_episodes: int = 1000,
        experiment_name: str = "",
    ):
        """Train."""
        context = Context(experiment_name, self, env, callbacks)
        context.on_training_begin()
        for episode in range(nb_episodes):
            state = env.reset()
            done = False
            step = 0
            self._episode_reset()
            self.stop = self._should_stop()
            context.on_episode_begin(episode)
            while not done and not self.done():
                action = random_policy(self, state)
                context.on_step_begin(step, action)
                state2, reward, done, info = env.step(action)
                observation = (state, action, reward, state2, done)
                self.observe(observation)
                context.on_step_end(step, observation)
                state = state2
                step += 1
            self._add_trace()
            if episode % self.update_frequency == 0:
                self._learn_pdfa()
            context.on_episode_end(episode)
        context.on_training_end()

    def test(
        self,
        env: gym.Env,
        callbacks: Collection[Callback] = (),
        nb_episodes: int = 1000,
        experiment_name: str = "",
    ):
        """Test."""
        context = Context(experiment_name, self, env, callbacks)
        context.on_training_begin()
        for episode in range(nb_episodes):
            self.current_state = 0
            state = env.reset()
            done = False
            step = 0
            context.on_episode_begin(episode)
            while not done:
                action = self.choose_best_action(state)
                context.on_step_begin(step, action)
                state2, reward, done, info = env.step(action)
                observation = (state, action, reward, state2, done)
                self.do_pdfa_transition(observation)
                context.on_step_end(step, observation)
                state = state2
                step += 1
            context.on_episode_end(episode)
        context.on_training_end()
