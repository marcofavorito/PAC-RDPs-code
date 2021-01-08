# -*- coding: utf-8 -*-
"""Implementation of the Malfunction Multi-Armed Bandit."""
from typing import Dict, Sequence

import gym
import numpy as np
from gym.spaces import Discrete

from src.helpers.gym import DiscreteEnv


class MalfunctionMAB(DiscreteEnv):
    """
    Malfunction Multi-Armed Bandit.

    One of the arms is broken, s.t. after the
    corresponding action is performed k times, its probability
    of winning drops to zero for one turn.

    The observation space {0, ..., k-1} (a counter up to k times).
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
        k: int,
        malfunctioning_arm: int,
        winning_reward: float = 1.0,
    ):
        """
        Initialize the environment.

        :param winning_probs: winning probabilities for each arm.
        :param k: the number of times after which the arm is broken.
        :param malfunctioning_arm: the arm that malfunctions.
        :param winning_reward: the reward in case of success.
        """
        assert all(p <= 1.0 for p in winning_probs)
        assert (
            0 <= malfunctioning_arm < len(winning_probs)
        ), f"Index of malfunctioning arm is not valid: '{malfunctioning_arm}'."
        self.winning_probs = np.asarray(winning_probs)
        self.k = k
        self.malfunctioning_arm = malfunctioning_arm
        self.reward = winning_reward

        P = self._compute_dynamics()
        nS = self.k + 1
        nA = self.nb_arms
        ids = [1.0] + [0.0] * (nS - 1)
        super().__init__(nS, nA, P, ids)

    @property
    def nb_arms(self) -> int:
        """Get the number of arms."""
        return len(self.winning_probs)

    def _compute_dynamics(self) -> Dict:
        """Compute the function of the dynamics."""
        P: Dict = {}
        last = self.k + 1
        for i in range(last):
            P[i] = {}
            for action in range(len(self.winning_probs)):
                P[i][action] = []
                if i == self.k:
                    # when arm is broken, next state is always 0
                    next_state = 0
                else:
                    # if action == malfunctioning arm, increase the counter, o/w don't change state
                    next_state = (
                        (i + 1) % last if action == self.malfunctioning_arm else i
                    )
                reward = self.reward
                probability = (
                    self.winning_probs[action]
                    if action != self.malfunctioning_arm or i != (last - 1)
                    else 0.0
                )
                transition_success = (probability, next_state, reward, False)
                transition_fail = (1 - probability, next_state, 0.0, False)
                P[i][action].append(transition_success)
                P[i][action].append(transition_fail)
        return P


class NonMarkovianMalfunctionMAB(gym.Wrapper):
    """
    Non-Markovian Malfunction MAB.

    The observation space has just one state.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the environment."""
        super().__init__(MalfunctionMAB(*args, **kwargs))
        self.observation_space = Discrete(1)

    def reset(self, **kwargs):
        """Reset."""
        super().reset(**kwargs)
        return 0

    def step(self, action):
        """Do a step."""
        s, r, done, info = super().step(action)
        return 0, r, done, info
