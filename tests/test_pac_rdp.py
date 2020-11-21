"""Main test module."""
from functools import partial
from typing import Set, Tuple

import gym
import numpy as np
from gym.wrappers import TimeLimit

from src.learn_pdfa.base import Algorithm, learn_pdfa
from src.learn_pdfa.utils.generator import MultiprocessedGenerator
from src.learn_rdps import RDPGenerator, random_exploration_policy
from src.types import State, TransitionFunctionDict


def learning_rotating_mab(
    stop_probability: float,
    winning_probabilities: Tuple[float, ...],
    max_episode_steps: int,
    nb_samples: int,
    delta: float,
    n_upperbound: int,
    nb_processes: int = 8,
) -> Tuple[RDPGenerator, Tuple[Set[State], TransitionFunctionDict]]:
    """Test learning of Rotating MAB."""
    env = gym.make("NonMarkovianRotatingMAB-v0", winning_probs=winning_probabilities)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    policy = partial(random_exploration_policy, env)

    rdp_generator = RDPGenerator(
        env, policy=policy, nb_rewards=2, stop_probability=stop_probability
    )

    examples = rdp_generator.sample(n=nb_samples)

    print(
        f"Apriori expected length of traces: 1/stop_prob = {1 / (stop_probability + np.finfo(float).eps)}"
    )
    print(f"Average length of traces: {np.mean([len(e) for e in examples])}")

    mp_rdp_generator = MultiprocessedGenerator(rdp_generator, nb_processes=nb_processes)
    pdfa = learn_pdfa(
        algorithm=Algorithm.BALLE,
        nb_samples=nb_samples,
        sample_generator=mp_rdp_generator,
        alphabet_size=rdp_generator.alphabet_size(),
        delta=delta,
        n=n_upperbound,
    )
    return rdp_generator, (pdfa.states, pdfa.transition_dict)


def test_learning_rotating_mab_2_arms_nondeterministic(nb_processes):
    """Test learning rotating MAB with 2 arms, nondeterministic."""
    rdp_generator, (v, t) = learning_rotating_mab(
        stop_probability=0.2,
        winning_probabilities=(0.7, 0.3),
        max_episode_steps=15,
        nb_samples=100000,
        delta=0.1,
        n_upperbound=10,
        nb_processes=nb_processes,
    )
    assert len(v) == 2


def test_learning_rotating_mab_3_arms_deterministic(nb_processes):
    """Test learning rotating MAB with 3 arms, deterministic."""
    rdp_generator, (v, t) = learning_rotating_mab(
        stop_probability=0.2,
        winning_probabilities=(1.0, 0.0, 0.0),
        max_episode_steps=15,
        nb_samples=100000,
        delta=0.1,
        n_upperbound=5,
        nb_processes=nb_processes,
    )

    assert len(v) == 3


def test_learning_rotating_mab_3_arms_nondeterministic(nb_processes):
    """Test learning rotating MAB with 3 arms, nondeterministic."""
    rdp_generator, (v, t) = learning_rotating_mab(
        stop_probability=0.2,
        winning_probabilities=(0.1, 0.2, 0.9),
        max_episode_steps=1000,
        nb_samples=1000000,
        delta=0.05,
        n_upperbound=6,
        nb_processes=nb_processes,
    )

    assert len(v) == 3


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
