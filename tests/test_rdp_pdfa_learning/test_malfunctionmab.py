"""Test RotMAB."""
from typing import Tuple

import gym
import pytest

from tests.test_rdp_pdfa_learning.base import BaseTestPdfaRdp


class BaseTestMalfunctionMAB(BaseTestPdfaRdp):
    """Base test class for malfunction MAB PDFA learning."""

    WINNING_PROBABILITIES: Tuple[float, ...] = (0.8, 0.2)
    K: int = 1
    MALFUNCTIONING_ARM = 0

    @classmethod
    def make_env(cls):
        """Make environment."""
        return gym.make(
            "NonMarkovianMalfunctionMAB-v0",
            winning_probs=cls.WINNING_PROBABILITIES,
            k=cls.K,
            malfunctioning_arm=cls.MALFUNCTIONING_ARM,
        )

    def test_nb_states(self):
        """Test expected number of states."""
        assert self.pdfa.nb_states == self.K + 1


class TestMalfunctionMAB2ArmsNondetK1(BaseTestMalfunctionMAB):
    """Test learning rotating MAB with 2 arms, nondeterministic, k=1."""

    K = 1
    OVERWRITE_CONFIG = dict(nb_samples=50000)


class TestMalfunctionMAB2ArmsNondetK2(BaseTestMalfunctionMAB):
    """Test learning rotating MAB with 2 arms, nondeterministic, k=1."""

    K = 2
    OVERWRITE_CONFIG = dict(nb_samples=50000)


@pytest.mark.exclude_ci
class TestMalfunctionMAB2ArmsNondetK3(BaseTestMalfunctionMAB):
    """Test learning rotating MAB with 2 arms, nondeterministic, k=1."""

    MAX_EPISODE_STEPS = 500
    WINNING_PROBABILITIES = (0.8, 0.2)
    K = 3
    OVERWRITE_CONFIG = dict(stop_probability=0.05, nb_samples=100000)
