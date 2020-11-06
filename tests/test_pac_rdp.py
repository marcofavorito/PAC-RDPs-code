"""Main test module."""
from functools import partial
from typing import Dict, Set, Tuple

import gym
import numpy as np
from gym.wrappers import TimeLimit

from src.learn_pdfa.base import Algorithm, learn_pdfa
from src.learn_pdfa.common import MultiprocessedGenerator
from src.learn_rdps import RDPGenerator, random_exploration_policy
from src.pdfa.types import Character


def learning_rotating_mab(
    stop_probability: float,
    winning_probabilities: Tuple[float, ...],
    max_episode_steps: int,
    nb_samples: int,
    delta: float,
    n_upperbound: int,
    nb_processes: int = 8,
) -> Tuple[RDPGenerator, Tuple[Set[int], Dict[int, Dict[Character, int]]]]:
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
    v, t = learn_pdfa(
        algorithm=Algorithm.BALLE,
        nb_samples=nb_samples,
        sample_generator=mp_rdp_generator,
        alphabet_size=rdp_generator.alphabet_size(),
        delta=delta,
        n=n_upperbound,
    )
    return rdp_generator, (v, t)


def test_learning_rotating_mab():
    """Test learning rotating MAB."""
    rdp_generator, (v, t) = learning_rotating_mab(
        stop_probability=0.2,
        winning_probabilities=(0.7, 0.3),
        max_episode_steps=15,
        nb_samples=100000,
        delta=0.1,
        n_upperbound=10,
        nb_processes=8,
    )
    assert len(v) == 2
