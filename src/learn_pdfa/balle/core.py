"""Entrypoint for the algorithm."""

import pprint
from collections import Counter
from math import log, sqrt
from typing import Dict, Optional, Set, Tuple

from src.learn_pdfa import logger
from src.learn_pdfa.balle.params import BalleParams
from src.learn_pdfa.common import l_infty_norm, prefix_distance_infty_norm
from src.pdfa import PDFA
from src.pdfa.types import Character, State, Word


def learn_pdfa(**kwargs) -> PDFA:
    """
    PAC-learn a PDFA.

    :param kwargs: the keyword arguments of the algorithm (see the BalleParams class).
    :return: the learnt PDFA.
    """
    params = BalleParams(**kwargs)
    logger.info(f"Parameters: {pprint.pformat(str(params))}")
    vertices, transitions = learn_subgraph(params)
    return vertices, transitions  # type: ignore


def extended_transition_fun(
    transitions: Dict[State, Dict[Character, State]], word: Word
) -> Optional[int]:
    """
    Compute the successor state from a word.

    :param transitions: the transitions indexed by state and character.
    :param word: the word.
    :return:
    """
    current_state: Optional[int] = 0
    for c in word:
        if current_state is None:
            return None
        current_state = transitions.get(current_state, {}).get(c)
    return current_state


def _compute_threshold(m_u, m_v, s_u, s_v, delta):
    n1 = 2 / min(m_u, m_v)
    n2 = log(8 * (s_u + s_v) / delta)
    return sqrt(n1 * n2)


def test_distinct(
    multiset_candidate: Counter, multiset_safe: Counter, params: BalleParams
) -> bool:
    """Test whether a candidate state is distinct from a safe node."""
    cardinality_candidate = sum(multiset_candidate.values())
    cardinality_safe = sum(multiset_safe.values())
    # +1 for the empty trace
    prefixes_candidate = sum(
        [(len(trace) + 1) * count for trace, count in multiset_candidate.items()]
    )
    prefixes_safe = sum(
        [(len(trace) + 1) * count for trace, count in multiset_safe.items()]
    )
    threshold = _compute_threshold(
        cardinality_candidate,
        cardinality_safe,
        prefixes_candidate,
        prefixes_safe,
        params.delta_0,
    )
    distance = max(
        l_infty_norm(multiset_candidate, multiset_safe),
        prefix_distance_infty_norm(multiset_candidate, multiset_safe),
    )
    return distance > threshold


def learn_subgraph(  # noqa: ignore
    params: BalleParams,
) -> Tuple[Set[int], Dict[int, Dict[Character, int]]]:
    """
    Learn a subgraph of the true PDFA.

    :param params: the parameters of the algorithms.
    :return: the graph
    """
    generator = params.sample_generator
    N = 100000
    # initialize variables
    initial_state = 0
    vertices = {initial_state}
    transitions: Dict[int, Dict[Character, int]] = {}
    alphabet = set(range(params.alphabet_size))
    vertex2multiset: Dict[int, Counter] = {}
    done = False
    iteration = 0
    iteration_upper_bound = params.n * params.alphabet_size

    # generate sample
    samples = generator.sample(n=N)
    samples = list(map(lambda x: tuple(x), samples))

    # attach the entire sample as a multiset ot the initial state.
    vertex2multiset[initial_state] = Counter()
    vertex2multiset[initial_state].update(samples)

    while not done:
        logger.info(f"Iteration {iteration}")

        candidate_nodes_by_transitions: Dict[Tuple[State, Character], int] = {}
        candidate_nodes_to_transitions: Dict[int, Tuple[State, Character]] = {}
        multisets: Dict[int, Counter] = {}

        # recompute candidate nodes
        for v in vertices:
            for c in alphabet:
                if transitions.get(v, {}).get(c) is None:  # if transition undefined
                    transition = (v, c)
                    new_candidate = len(vertices) + len(candidate_nodes_to_transitions)
                    candidate_nodes_to_transitions[new_candidate] = transition
                    candidate_nodes_by_transitions[transition] = new_candidate
                    multisets[new_candidate] = Counter()

        no_candidate_nodes = len(candidate_nodes_to_transitions) == 0
        if no_candidate_nodes:
            break

        # compute multisets
        for word in samples:
            # s is always non-empty
            for i in range(len(word)):
                r, sigma, t = word[:i], word[i], word[i + 1 :]
                q = extended_transition_fun(transitions, r)
                if q is None:
                    continue
                transition = (q, sigma)
                if transition in candidate_nodes_by_transitions:
                    candidate_node = candidate_nodes_by_transitions[transition]
                    multisets[candidate_node].update([tuple(t)])

        chosen_candidate_node, biggest_multiset = max(
            multisets.items(), key=lambda x: sum(x[1].values())
        )
        if sum(biggest_multiset.values()) == 0:
            break

        start_state, character = candidate_nodes_to_transitions[chosen_candidate_node]
        non_distinct_vertices = set()
        for v in vertices:
            is_distinct = test_distinct(biggest_multiset, vertex2multiset[v], params)
            if not is_distinct:
                non_distinct_vertices.add(v)

        if len(non_distinct_vertices) == 0:
            # we've got a new node
            new_vertex = len(vertices)
            vertices.add(new_vertex)
            vertex2multiset[new_vertex] = biggest_multiset
            transitions.setdefault(start_state, {})[character] = new_vertex
        else:
            # pick a safe node that has not distinguished from best candidate
            # for deterministic behaviour, pick the smallest
            old_vertex = sorted(non_distinct_vertices)[0]
            transitions.setdefault(start_state, {})[character] = old_vertex

        iteration += 1
        done = iteration >= iteration_upper_bound

    return vertices, transitions
