"""Main test module."""
from copy import copy
from typing import Dict, Tuple

import gym
from gym.wrappers import TimeLimit

from src.experiment_utils.pac_rdp import RDPLearner

RDP_DEFAULT_CONFIG = dict(
    stop_probability=0.2,
    nb_samples=150000,
    delta=0.05,
    n_upperbound=5,
    nb_sampling_processes=8,
)


class BaseTestRotatingMAB:
    """Base test class for rotating MAB learning."""

    NB_PROCESSES = 8
    MAX_EPISODE_STEPS = 100
    CONFIG: Dict = RDP_DEFAULT_CONFIG
    WINNING_PROBABILITIES: Tuple[float, ...] = ()
    OVERWRITE_CONFIG: Dict = {}

    @classmethod
    def setup_class(cls):
        """Set up the test."""
        config = copy(cls.CONFIG)
        config.update(cls.OVERWRITE_CONFIG)
        env = gym.make(
            "NonMarkovianRotatingMAB-v0", winning_probs=cls.WINNING_PROBABILITIES
        )
        env = TimeLimit(env, max_episode_steps=cls.MAX_EPISODE_STEPS)
        cls.rdp_learner = RDPLearner(**config)
        cls.rdp_learner._learn_pdfa(env)
        cls.rdp_learner.to_graphviz().render("actual")

    def test_nb_states(self):
        """Test expected number of states."""
        assert self.rdp_learner.pdfa.nb_states == len(self.WINNING_PROBABILITIES)


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
