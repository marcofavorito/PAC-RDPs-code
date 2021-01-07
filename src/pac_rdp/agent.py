# -*- coding: utf-8 -*-
"""This module contains the main code for the algorithm."""
import logging
import pprint
from math import log
from typing import Collection, Optional, cast

import gym
import numpy as np
from gym import Space
from pdfa_learning.learn_pdfa.base import learn_pdfa
from pdfa_learning.pdfa import PDFA

from src.algorithms.value_iteration import value_iteration
from src.callbacks.base import Callback
from src.core import Context, random_policy
from src.helpers.gym import DiscreteEnv
from src.pac_rdp.base import BasePacRdpAgent
from src.pac_rdp.helpers import RDPGenerator, mdp_from_pdfa
from src.types import AgentObservation


def stop_probability_from_l(l_value: int) -> float:
    """
    Compute the stop probability from the depth.

    :param l_value: the state upper bound.
    :return: the stop probability.
    """
    return 1 / (10 * l_value + 1)


def max_number_of_steps(l_value: int):
    """Compute the max number of steps."""
    p = stop_probability_from_l(l_value)
    return (2 / p) * l_value ** 5 * (l_value + 4 * log(l_value))


class PacRdpAgent(BasePacRdpAgent):
    """Implementation of PAC-RDP agent."""

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        env: DiscreteEnv,
        epsilon: float = 0.1,
        delta: float = 0.1,
        gamma: float = 0.9,
        max_l: int = 10,
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
        self.max_l = max_l
        self._reset()

    def _encode_reward(self, reward: float) -> int:
        """Encode the reward."""
        return self.env.rewards.index(reward)  # type: ignore

    def _reset(self):
        """Reset the state of the agent."""
        self.current_l = 1
        self.pdfa: Optional[PDFA] = None
        self._iteration_reset()
        self._episode_reset()

    def _episode_reset(self):
        """Episode reset."""
        self.current_episode = []
        self.current_state = 0

    def _iteration_reset(self):
        """Reset after the end of one iteration."""
        self.current_p = stop_probability_from_l(self.current_l)
        self.current_M = max_number_of_steps(self.current_l)
        self.current_i = 0
        self.hard_stop = False
        self.stop = False

    def done(self) -> bool:
        """Return true when either we should stop or a hard stop happened."""
        return self.stop or self.hard_stop

    def should_stop(self) -> bool:
        """Check if the next action should be the stop action."""

    def _should_stop(self) -> bool:
        """Check that next action should be the stop action."""
        return np.random.random() < self.current_p

    def _should_hard_stop(self) -> bool:
        """Check that next action should be a hard-stop."""
        return (self.current_M - self.current_i) <= len(self.current_episode)

    def observe(self, agent_observation: AgentObservation):
        """Observe a transition."""
        self.current_episode.append(agent_observation)
        self.stop = self._should_stop()
        self.hard_stop = self._should_hard_stop()

    def do_pdfa_transition(self, agent_observation: AgentObservation):
        """Do a PDFA transition."""
        if self.pdfa is None:
            self.current_state = None
            return
        self.pdfa = cast(PDFA, self.pdfa)
        s, a, r, sp, done = agent_observation
        symbol = self._rdp_generator.encoder((a, self._encode_reward(r), sp))
        self.current_state = self.pdfa.transition_dict.get(self.current_state, {}).get(
            symbol, [None]
        )[0]

    def _learn_pdfa(self):
        """Learn the PDFA."""
        if len(self.dataset) == 0:
            logging.error("Dataset length is 0.")
            return
        pdfa = learn_pdfa(
            dataset=self.dataset,
            n=self.current_l,
            alphabet_size=self._rdp_generator.alphabet_size(),
            delta=self.delta ** 2,
            with_infty_norm=False,
            with_smoothing=True,
        )
        self.pdfa = pdfa

        new_env = mdp_from_pdfa(
            cast(PDFA, self.pdfa),
            cast(RDPGenerator, self._rdp_generator),
            stop_probability=self.current_p,
        )
        logging.info("Computed MDP.")
        logging.info(f"Observation space: {new_env.observation_space}")
        logging.info(f"Action space: {new_env.action_space}")
        logging.info(f"Dynamics:\n{pprint.pformat(new_env.P)}")
        self.value_function, self.current_policy = value_iteration(
            new_env, max_iterations=50, discount=self.gamma
        )
        logging.info("Value iteration completed.")

        self._iteration_reset()
        logging.info(f"New l: {self.current_l}")
        logging.info(f"New M: {self.current_M}")
        logging.info(f"New p: {self.current_p}")

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
        episode = 0
        for self.current_l in range(1, self.max_l + 1):
            if episode > nb_episodes:
                break
            self._iteration_reset()
            while not self.hard_stop:
                if episode > nb_episodes:
                    break
                state = env.reset()
                done = False
                step = 0
                self._episode_reset()
                self.stop = self._should_stop()
                self.hard_stop = self._should_hard_stop()
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
                if not self.hard_stop:
                    self._add_trace()
                    self.current_i += len(self.current_episode)
                context.on_episode_end(episode)
                episode += 1
            self._learn_pdfa()
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
