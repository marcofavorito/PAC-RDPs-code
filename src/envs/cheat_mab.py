# -*- coding: utf-8 -*-
"""Implementation of the Cheat Multi-Armed Bandit."""
from typing import Dict, Sequence

import gym
from gym.spaces import Discrete

from src.helpers.gym import DiscreteEnv


class CheatMAB(DiscreteEnv):
    """
    A variant of the classic MAB.

    There exists a sequence of actions s.t. after
    performing that sequence, all actions lead to
    a reward with probability 1 from that point on.

    E.g. pull arm 1 three times, then pull arm 3.

    The reward is deterministic.
    """

    def __init__(
        self,
        nb_arms: int,
        pattern: Sequence[int],
        reward: float = 1.0,
        terminate_on_win: bool = True,
    ):
        """
        Initialize a sequential MAB.

        :param nb_arms: the number of arms.
        :param pattern: the pattern to perform in order to give a reward.
        :param reward: the reward.
        :param terminate_on_win: whether the episode should terminate when the pattern is matched.
        """
        self.reward = reward
        self.nb_arms = nb_arms
        self.pattern = pattern
        self.terminate_on_win = terminate_on_win

        # validate pattern
        assert len(self.pattern) > 0, "Pattern cannot be empty."
        assert min(self.pattern) >= 0, "Negative values in patterns not allowed."
        assert (
            max(self.pattern) < self.nb_arms
        ), "Values greater than the number of arms not allowed."

        nS = len(self.pattern) + 1
        nA = nb_arms
        isd = [1.0] + [0.0] * (nS - 1)
        P = self._compute_dynamics()
        super().__init__(nS, nA, P, isd)

    def _compute_dynamics(self) -> Dict:
        """Compute dynamics."""
        P: Dict = {}
        nb_states = len(self.pattern) + 1
        last_state = nb_states - 1
        for i, step in enumerate(self.pattern):
            P[i] = {}
            for arm in range(self.nb_arms):
                P[i][arm] = []
                # go to next state if action is right,
                # else, go to second state if action is the right first action,
                # otherwise, go to start
                next_state = (
                    i + 1 if step == arm else (0 if arm != self.pattern[0] else 1)
                )
                next_is_final = next_state == last_state
                reward = self.reward if next_is_final else 0.0
                prob = 1.0
                tr = (prob, next_state, reward, self.terminate_on_win and next_is_final)
                P[i][arm].append(tr)

        # when on final state, every transition gives 1
        P[last_state] = {}
        for arm in range(self.nb_arms):
            tr = (1.0, last_state, self.reward, self.terminate_on_win)
            P[last_state][arm] = [tr]

        return P


class NonMarkovianSequentialMAB(gym.Wrapper):
    """Non-Markovian Sequential MAB."""

    def __init__(self, *args, **kwargs):
        """Initialize a non-Markovian sequential MAB."""
        super().__init__(CheatMAB(*args, **kwargs))
        self.observation_space = Discrete(2)
        self._last_reward = False

    def step(self, action):
        """Do a step."""
        s, r, done, info = super().step(action)
        self._last_reward = r > 0.0
        return int(self._last_reward), r, done, info
