# -*- coding: utf-8 -*-
"""Implementation of the Rotating Multi-Armed Bandit."""
from typing import Sequence

import gym
import numpy as np
from gym.spaces import Discrete


class RotatingMAB(gym.Env):
    """
    Rotating Multi-Armed Bandit.

    Let π be a vector that assigns the probability of winning
    the reward for each action. This probability shifts right (i.e.,+1 mod n)
    every time the agent receives a reward. Therefore, the probability to win
    for each arm depends on the entire history, but via a regular function.

    The observation space {0, ..., N-1}, one for each state of the probabilities.
    The action space is {0, ..., N-1}, where N is the number of arms.
    The reward range is [0.0, 1.0]

    First appeared in:

        Abadi, Eden, and Ronen I. Brafman. "Learning and Solving Regular
        Decision Processes." arXiv preprint arXiv:2003.01008 (2020).

    """

    reward_range = (0.0, 1.0)

    def __init__(
        self,
        winning_probs: Sequence[float],
        seed: int = 42,
        winning_reward: float = 1.0,
    ):
        """
        Initialize the environment.

        :param winning_probs: winning probabilities for each arm.
        :param seed: the seed for the RNG.
        :param winning_reward: the reward in case of success.
        """
        assert all(p <= 1.0 for p in winning_probs)
        self.winning_probs = np.asarray(winning_probs)
        self.current_winning_probs = np.copy(self.winning_probs)

        self.observation_space = Discrete(self.nb_arms)
        self.action_space = Discrete(self.nb_arms)

        self.current_state = 0
        self.seed = seed
        self.reward = winning_reward
        np.random.seed(seed)

    @property
    def nb_arms(self) -> int:
        """The number of arms."""
        return len(self.winning_probs)

    def _reward(self, action):
        chosen_arm_winning_prob = self.current_winning_probs[action]
        outcome = np.random.binomial(size=1, n=1, p=chosen_arm_winning_prob)[0]
        return self.reward * outcome

    def _rotate(self):
        """Shift probabilities."""
        self.current_state = (self.current_state + 1) % self.nb_arms
        self.current_winning_probs = np.roll(self.current_winning_probs, shift=1)

    def reset(self):
        """
        Reset the environment.

        :return: the initial state.
        """
        self.current_winning_probs = np.copy(self.winning_probs)
        self.current_state = 0
        return self.current_state

    def step(self, action):
        """
        Do an action.

        It implements the gym.Env.step method.

        :param action: the action.
        :return: (obs, reward, done, info)
        """
        reward = self._reward(action)
        if reward == 1.0:
            self._rotate()
        done = False
        return self.current_state, reward, done, {}

    def render(self, mode="human"):
        """
        Render the environment.

        :param mode: the modality. Only 'human' is supported so far.
        :return: None
        """
        print("*" * 50)
        print("Number of arms: ", self.nb_arms)
        print("Original winning probabilities: ", self.winning_probs)
        print("Current winning probabilities: ", self.current_winning_probs)
        print("*" * 50)


class NonMarkovianRotatingMAB(gym.Wrapper):
    """Non-Markovian Rotating MAB."""

    def __init__(self, *args, **kwargs):
        super().__init__(RotatingMAB(*args, **kwargs))
        self.observation_space = Discrete(2)
        self._last_reward = False

    def step(self, action):
        s, r, done, info = super().step(action)
        self._last_reward = r > 0.0
        return int(self._last_reward), r, done, info
