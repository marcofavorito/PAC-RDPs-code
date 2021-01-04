# -*- coding: utf-8 -*-
"""Q-Learning implementation."""
import sys
from collections import defaultdict
from functools import partial
from typing import Any, Dict

import gym
import numpy as np

from src.core import Agent
from src.helpers.gym import space_size
from src.types import AgentObservation


class QLearning(Agent):
    """Q-Learning agent."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        alpha: float = 0.1,
        gamma: float = 0.99,
    ):
        """Initialize."""
        super().__init__(observation_space, action_space)

        self.alpha = alpha
        self.gamma = gamma

        self.q: Dict[Any, np.ndarray] = defaultdict(partial(self._random_action))

    def _random_action(self):
        """Random actions."""
        return (
            np.random.randn(
                space_size(self.action_space),
            )
            * sys.float_info.epsilon
        )

    def choose_action(self, state: Any):
        """Choose the best action."""
        return np.argmax(self.q[state])

    def on_step_end(self, step, agent_observation: AgentObservation, **kwargs) -> None:
        """On step end."""
        s, a, r, sp, _ = agent_observation
        alpha = self.alpha
        gamma = self.gamma
        q = self.q
        q[s][a] += alpha * (r + gamma * np.max(q[sp]) - q[s][a])
