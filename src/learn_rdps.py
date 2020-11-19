"""Module that includes algorithms to learn RDPs."""
from functools import partial
from typing import Callable, List, Sequence, cast

import gym
import numpy as np

from src.learn_pdfa.utils.generator import Generator
from src.pdfa.base import FINAL_SYMBOL
from src.pdfa.types import Word


class RDPGenerator(Generator):
    """Generate a trace against."""

    def __init__(
        self,
        env: gym.Env,
        nb_rewards: int,
        policy: Callable,
        stop_probability: float = 0.05,
    ):
        """Initialize the RDP generator."""
        self._env = env
        self._policy = policy
        self._stop_probability = stop_probability

        self.obs_space_dim = self._env.observation_space.n
        self.action_dim = self._env.action_space.n
        self.nb_rewards = nb_rewards
        self.encoder = partial(
            np.ravel_multi_index,
            dims=(self.action_dim, self.nb_rewards, self.obs_space_dim),
        )
        self.decoder = partial(
            np.unravel_index,
            dims=(self.action_dim, self.nb_rewards, self.obs_space_dim),
        )

    def alphabet_size(self) -> int:
        """Get the alphabet size."""
        return int(np.prod([self.action_dim, self.nb_rewards, self.obs_space_dim]))

    def sample(self, n: int = 1, with_final: bool = False) -> Sequence[Word]:
        """Sample a set of samples."""
        result = []
        for _ in range(n):
            word = self._sample_word()
            word = cast(List, word) + ([FINAL_SYMBOL] if with_final else [])
            result.append(word)
        return result

    def _should_stop(self) -> bool:
        """Return True if the current episode should stop, false otherwise."""
        return np.random.random() < self._stop_probability

    def _sample_word(self) -> Word:
        """Sample one word."""
        _initial_state = self._env.reset()  # noqa: ignore
        done = False
        trace = []  # [(0, 0, initial_state)]
        while not done:
            if self._should_stop():
                break
            action = self._policy()
            obs, reward, done, _ = self._env.step(action)
            trace += [(action, int(reward), obs)]

        encoded_trace = [self.encoder(x) for x in trace]
        return encoded_trace


def random_exploration_policy(env: gym.Env) -> int:
    """
    Random exploration policy.

    It is declared as a module function so to be pickable.
    """
    return env.action_space.sample()
