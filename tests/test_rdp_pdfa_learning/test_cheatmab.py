"""Test learning of CheatMAB's PDFA."""
from typing import List

import gym
import pytest

from tests.test_rdp_pdfa_learning.base import BaseTestPdfaRdp


class BaseTestCheatMAB(BaseTestPdfaRdp):
    """Base test class for malfunction MAB PDFA learning."""

    PATTERN: List[int] = [0, 0, 1]
    NB_ARMS = 2

    @classmethod
    def make_env(cls):
        """Make environment."""
        return gym.make(
            "NonMarkovianCheatMAB-v0",
            nb_arms=cls.NB_ARMS,
            pattern=cls.PATTERN,
            terminate_on_win=True,
        )

    def test_nb_states(self):
        """Test expected number of states."""
        assert self.pdfa.nb_states == len(self.PATTERN) + 1


class TestCheatMAB2Arms(BaseTestCheatMAB):
    """Test learning rotating MAB with 2 arms."""

    PATTERN = [0, 1]
    NB_ARMS = 2
    OVERWRITE_CONFIG = dict(nb_samples=50000)


class TestCheatMAB3Arms(BaseTestCheatMAB):
    """Test learning rotating MAB with 3 arms."""

    PATTERN = [0, 1, 2]
    NB_ARMS = 3
    OVERWRITE_CONFIG = dict(nb_samples=75000)


class TestCheatMAB4ArmsLen3(BaseTestCheatMAB):
    """Test learning rotating MAB with 4 arms, pattern of length 3."""

    MAX_EPISODE_STEPS = 200
    PATTERN = [0, 1, 2]
    NB_ARMS = 4
    OVERWRITE_CONFIG = dict(stop_probability=0.05, nb_samples=200000)


@pytest.mark.exclude_ci
class TestCheatMAB4ArmsLen4(BaseTestCheatMAB):
    """Test learning rotating MAB with 4 arms, pattern of length 4."""

    MAX_EPISODE_STEPS = 200
    PATTERN = [0, 1, 2, 3]
    NB_ARMS = 4
    OVERWRITE_CONFIG = dict(stop_probability=0.05, nb_samples=700000)
