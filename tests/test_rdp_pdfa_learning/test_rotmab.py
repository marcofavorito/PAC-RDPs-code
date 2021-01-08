"""Test RotMAB."""
from typing import Tuple

import gym

from tests.test_rdp_pdfa_learning.base import BaseTestPdfaRdp


class BaseTestRotatingMAB(BaseTestPdfaRdp):
    """Base test class for rotating MAB PDFA learning."""

    WINNING_PROBABILITIES: Tuple[float, ...] = ()

    @classmethod
    def make_env(cls):
        """Make environment."""
        return gym.make(
            "NonMarkovianRotatingMAB-v0", winning_probs=cls.WINNING_PROBABILITIES
        )

    def test_nb_states(self):
        """Test expected number of states."""
        assert self.pdfa.nb_states == len(self.WINNING_PROBABILITIES)


class TestRotatingMAB2ArmsNondet(BaseTestRotatingMAB):
    """Test learning rotating MAB with 2 arms, nondeterministic."""

    WINNING_PROBABILITIES = (0.7, 0.3)
    OVERWRITE_CONFIG = dict(nb_samples=10000)


class TestRotatingMAB3ArmsDet(BaseTestRotatingMAB):
    """Test learning rotating MAB with 3 arms, deterministic."""

    WINNING_PROBABILITIES = (1.0, 0.0, 0.0)
    OVERWRITE_CONFIG = dict(nb_samples=25000)


class TestRotatingMAB3ArmsNondet(BaseTestRotatingMAB):
    """Test learning rotating MAB with 3 arms, nondeterministic."""

    WINNING_PROBABILITIES = (0.1, 0.2, 0.9)
    OVERWRITE_CONFIG = dict(nb_samples=75000)


class TestRotatingMAB4ArmsDet(BaseTestRotatingMAB):
    """Test learning rotating MAB with 4 arms, deterministic."""

    WINNING_PROBABILITIES = (1.0, 0.0, 0.0, 0.0)
    OVERWRITE_CONFIG = dict(nb_samples=75000)


class TestRotatingMAB4ArmsNondet(BaseTestRotatingMAB):
    """Test learning rotating MAB with 4 arms, nondeterministic."""

    WINNING_PROBABILITIES = (0.9, 0.7, 0.5, 0.3)
    OVERWRITE_CONFIG = dict(nb_samples=200000, stop_probability=0.1)


class TestRotatingMAB5ArmsDet(BaseTestRotatingMAB):
    """Test learning rotating MAB with 5 arms, deterministic."""

    WINNING_PROBABILITIES = (1.0, 0.0, 0.0, 0.0, 0.0)
    OVERWRITE_CONFIG = dict(nb_samples=75000, stop_probability=0.1)


class TestRotatingMAB5ArmsNondet(BaseTestRotatingMAB):
    """Test learning rotating MAB with 5 arms, nondeterministic."""

    WINNING_PROBABILITIES = (0.9, 0.4, 0.3, 0.2, 0.1)
    OVERWRITE_CONFIG = dict(nb_samples=200000, stop_probability=0.1)
