"""Implement the Algorithm 2 of (Palmer and Goldberg 2007) to estimate probabilities."""
import math
import pprint
from collections import Counter
from math import ceil, log
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.learn_pdfa import logger
from src.learn_pdfa.common import _Params
from src.pdfa import PDFA
from src.pdfa.types import TransitionFunctionDict


def _sample_size(params: _Params) -> int:
    eps = params.epsilon
    s = params.alphabet_size
    n = params.n
    delta2 = params.delta_2
    n1 = 2 * n * s / delta2
    n2 = (64 * n * s / eps / delta2) ** 2
    n3 = 32 * n * s / eps
    n4 = log(2 * n * s / delta2)
    N = ceil(n1 * n2 * n3 * n4)
    return N


def _rename_final_state(
    vertices: Set[int], transition_dict: TransitionFunctionDict, final_node: int
) -> None:
    """
    Rename final state in set of vertices and transition dictionary.

    It does side-effects on the data structures.

    :param vertices: the set of vertices.
    :param transition_dict: the transition dictionary.
    :param final_node: the final node.
    :return: None
    """
    if final_node == len(vertices) - 1:
        vertices.remove(final_node)
    else:
        node_to_rename = len(vertices) - 1
        new_final_node = len(vertices) - 1
        new_node_name = final_node
        vertices.remove(node_to_rename)
        node_transitions = transition_dict.pop(node_to_rename)
        transition_dict[new_node_name] = node_transitions
        for _, out_transitions in transition_dict.items():
            for character, (next_state, prob) in out_transitions.items():
                if next_state == node_to_rename:
                    out_transitions[character] = (new_node_name, prob)
                elif next_state == final_node:
                    out_transitions[character] = (new_final_node, prob)


def learn_probabilities(
    graph: Tuple[Set[int], Dict[int, Dict[int, int]]], params: _Params
) -> PDFA:
    """
    Learn the probabilities of the PDFA.

    :param graph: the learned subgraph of the true PDFA.
    :param params: the parameters of the algorithms.
    :return: the PDFA.
    """
    logger.info("Start learning probabilities.")
    vertices, transitions = graph
    pprint.pprint(transitions)
    initial_state = 0
    N = _sample_size(params)
    logger.info(f"Sample size: {N}.")
    N = min(N, params.n2_max_debug if params.n2_max_debug else N)
    logger.info(f"Using N = {N}.")
    generator = params.sample_generator
    sample = generator.sample(N)
    n_observations: Dict[Tuple[int, int], List[int]] = {}
    for word in sample:
        q_visits: Counter = Counter()
        current_state = initial_state
        for character in word:
            # update statistics
            q_visits.update([current_state])
            n_observations.setdefault((current_state, character), []).append(
                q_visits[current_state]
            )

            # compute next state
            next_state: Optional[int] = transitions.get(current_state, {}).get(
                character
            )

            if next_state is None:
                break
            current_state = next_state

    gammas: Dict[int, Dict[int, float]] = {}
    # compute mean
    for (q, sigma), counts in n_observations.items():
        gammas.setdefault(q, {})[sigma] = 1 / np.mean(counts)
    # rescale probabilities
    for _, out_probabilities in gammas.items():
        characters, probabilities = zip(*list(out_probabilities.items()))
        probability_sum = math.fsum(probabilities)
        new_probabilities = [p / probability_sum for p in probabilities]
        out_probabilities.update(dict(zip(characters, new_probabilities)))

    # compute transition function for the PDFA
    transition_dict: TransitionFunctionDict = {}
    for q, out_transitions in transitions.items():
        transition_dict.setdefault(q, {})
        for sigma, q_prime in out_transitions.items():
            prob = gammas.get(q, {}).get(sigma, 0.0)
            transition_dict[q][sigma] = (q_prime, prob)

    # the final node is the one without outgoing transitions.
    # renumber the vertices and the transition dictionary accordingly.
    no_out_transitions = vertices.difference(set(transition_dict.keys()))
    logger.info(f"Computed vertices: {pprint.pformat(vertices)}")
    logger.info(f"Computed transition dictionary: {pprint.pformat(transition_dict)}")

    assert (
        len(no_out_transitions) == 1
    ), f"Cannot determine which is the final state. The set of candidates is: {no_out_transitions}."
    final_node = list(no_out_transitions)[0]
    logger.info(f"Computed final node: {final_node} (no outgoing transitions)")
    _rename_final_state(vertices, transition_dict, final_node)
    logger.info(f"Renamed vertices: {pprint.pformat(vertices)}")
    logger.info(f"Renamed transition dictionary: {pprint.pformat(transition_dict)}")

    return PDFA(len(vertices), params.alphabet_size, transition_dict)
