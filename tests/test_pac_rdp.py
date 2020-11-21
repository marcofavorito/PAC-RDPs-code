"""Main test module."""

import gym
from gym.wrappers import TimeLimit

from src.experiment_utils.pac_rdp import RDPLearner


def test_learning_rotating_mab_2_arms_nondeterministic(nb_processes):
    """Test learning rotating MAB with 2 arms, nondeterministic."""
    winning_probabilities = (0.7, 0.3)
    max_episode_steps = 15
    env = gym.make("NonMarkovianRotatingMAB-v0", winning_probs=winning_probabilities)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    rdp_learner = RDPLearner(
        stop_probability=0.2,
        nb_samples=150000,
        delta=0.05,
        n_upperbound=5,
        nb_processes=nb_processes,
    )
    rdp_learner._learn_pdfa(env)
    assert rdp_learner.pdfa.nb_states == 2


def test_learning_rotating_mab_3_arms_deterministic(nb_processes):
    """Test learning rotating MAB with 3 arms, deterministic."""
    winning_probabilities = (1.0, 0.0, 0.0)
    max_episode_steps = 15
    env = gym.make("NonMarkovianRotatingMAB-v0", winning_probs=winning_probabilities)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    rdp_learner = RDPLearner(
        stop_probability=0.2,
        nb_samples=150000,
        delta=0.1,
        n_upperbound=5,
        nb_processes=nb_processes,
    )
    rdp_learner._learn_pdfa(env)
    assert rdp_learner.pdfa.nb_states == 3


def test_learning_rotating_mab_3_arms_nondeterministic(nb_processes):
    """Test learning rotating MAB with 3 arms, nondeterministic."""
    winning_probabilities = (0.1, 0.2, 0.9)
    max_episode_steps = 15
    env = gym.make("NonMarkovianRotatingMAB-v0", winning_probs=winning_probabilities)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    rdp_learner = RDPLearner(
        stop_probability=0.2,
        max_episode_steps=1000,
        nb_samples=1000000,
        delta=0.05,
        n_upperbound=6,
        nb_processes=nb_processes,
    )
    rdp_learner._learn_pdfa(env)
    assert rdp_learner.pdfa.nb_states == 3


"""
def test_learning_rotating_mab_4_arms_deterministic(nb_processes):
    Test learning rotating MAB with 4 arms, deterministic.
    rdp_generator, (v, t) = learning_rotating_mab(
        stop_probability=0.2,
        winning_probabilities=(1.0, 0.0, 0.0, 0.0),
        max_episode_steps=50,
        nb_samples=300000,
        delta=0.1,
        n_upperbound=5,
        nb_processes=nb_processes,
    )
    assert len(v) == 4
"""
