# -*- coding: utf-8 -*-
"""Implementation of the Sequential Multi-Armed Bandit."""
from typing import Sequence

import gym
from gym.spaces import Discrete


class SequentialMAB(gym.Env):
    """
    A variant of MAB.

    In this variant of MAB, the agent has to
    take action in a certain pattern in order
    to collect a reward.

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

        self.observation_space = Discrete(len(pattern))
        self.action_space = Discrete(nb_arms)
        self.reward_range = (0.0, reward)

        self.current_state = 0

    def render(self, mode="human"):
        """Render environment state."""

    def reset(self):
        """Reset environment state."""
        self.current_state = 0
        return self.current_state

    def step(self, action):
        """Do a step."""
        done = False
        if action == self.pattern[self.current_state]:
            # if the action is the next of the pattern,
            # continue the pattern
            self.current_state += 1
        else:
            # otherwise, restart
            self.current_state = 0
            # handle the case when the action was wrong,
            # but it is also the first of the pattern
            if action == self.pattern[0]:
                self.current_state = 1

        # if we reached the end of the pattern, give 1.0
        reward = 0.0
        if self.current_state == len(self.pattern):
            self.current_state = 0
            reward = self.reward
            done = self.terminate_on_win

        return self.current_state, reward, done, {}


class NonMarkovianSequentialMAB(gym.Wrapper):
    """Non-Markovian Sequential MAB."""

    def __init__(self, *args, **kwargs):
        """Initialize a non-Markovian sequential MAB."""
        super().__init__(SequentialMAB(*args, **kwargs))
        self.observation_space = Discrete(1)
        self._last_reward = False

    def step(self, action):
        """Do a step."""
        s, r, done, info = super().step(action)
        self._last_reward = r > 0.0
        return 0, r, done, info
