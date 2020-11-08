"""Entrypoint for the algorithm."""

import pprint
from abc import ABC
from math import log, sqrt
from typing import Dict, Optional, Set, Tuple

from src.learn_pdfa import logger
from src.learn_pdfa.balle.params import BalleParams
from src.learn_pdfa.utils.base import l_infty_norm, prefix_distance_infty_norm
from src.learn_pdfa.utils.multiset import Counter as ConcreteMultiset  # noqa: ignore
from src.learn_pdfa.utils.multiset import Multiset
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
    # vertices, transitions = learn_subgraph(params)  # noqa
    vertices, transitions = SubgraphLearner(params).learn()
    # TODO do the part of learning probabilities
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
    multiset_candidate: Multiset, multiset_safe: Multiset, params: BalleParams
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


class SubgraphLearner(ABC):
    """Abstract learner of subgraphs."""

    def __init__(self, params: BalleParams):
        """Initialize the learner."""
        self._params = params

    @property
    def params(self) -> BalleParams:
        """Get the parameters."""
        return self._params

    def _initialize_variables(self):
        # global variables
        self.initial_state = 0
        self.vertices = {self.initial_state}
        self.transitions: Dict[int, Dict[Character, int]] = {}
        self.alphabet = set(range(self.params.alphabet_size))
        self.vertex2multiset: Dict[int, Multiset] = {}
        self.done = False
        self.iteration = 0
        self.iteration_upper_bound = self.params.n * self.params.alphabet_size
        self.samples = []

    def _sample(self):
        """Do the sampling."""
        generator = self.params.sample_generator
        self.samples = generator.sample(n=self.params.nb_samples)
        self.samples = list(map(lambda x: tuple(x), self.samples))
        # attach the entire sample as a multiset ot the initial state.
        self.vertex2multiset[self.initial_state] = ConcreteMultiset()
        self.vertex2multiset[self.initial_state].update(self.samples)

    def learn(self) -> Tuple[Set[int], Dict[int, Dict[Character, int]]]:
        """
        Do the learning.

        This is the main entry-point of the class.
        """
        self._initialize_variables()
        self._sample()
        while not self.done:
            self._do_iteration()

        return self.vertices, self.transitions

    def _do_iteration(self):
        """Do one iteration."""
        logger.info(f"Iteration {self.iteration}")

        # compute candidate nodes. If none, terminate.
        self._compute_candidate_nodes()
        no_candidate_nodes = len(self.candidate_nodes_to_transitions) == 0
        if no_candidate_nodes:
            self.done = True
            return

        # compute multisets
        self._compute_multisets()
        self.chosen_candidate_node, self.biggest_multiset = max(
            self.multisets.items(), key=lambda x: sum(x[1].values())
        )
        if sum(self.biggest_multiset.values()) == 0:
            self.done = True
            return

        # compute non-distinct vertices, and add a new state or just a new edge
        self._compute_non_distinct_vertices()
        self._add_new_state_or_edge()

        # end of iteration
        self.iteration += 1
        self.done = self.iteration >= self.iteration_upper_bound

    def _compute_candidate_nodes(self):
        # per-iteration variables
        self.candidate_nodes_by_transitions: Dict[Tuple[State, Character], int] = {}
        self.candidate_nodes_to_transitions: Dict[int, Tuple[State, Character]] = {}
        self.multisets: Dict[int, ConcreteMultiset] = {}

        # recompute candidate nodes
        for v in self.vertices:
            for c in self.alphabet:
                if (
                    self.transitions.get(v, {}).get(c) is None
                ):  # if transition undefined
                    transition = (v, c)
                    new_candidate = len(self.vertices) + len(
                        self.candidate_nodes_to_transitions
                    )
                    self.candidate_nodes_to_transitions[new_candidate] = transition
                    self.candidate_nodes_by_transitions[transition] = new_candidate
                    self.multisets[new_candidate] = ConcreteMultiset()

    def _compute_multisets(self):
        for word in self.samples:
            # s is always non-empty
            for i in range(len(word)):
                r, sigma, t = word[:i], word[i], word[i + 1 :]
                q = extended_transition_fun(self.transitions, r)
                if q is None:
                    continue
                transition = (q, sigma)
                if transition in self.candidate_nodes_by_transitions:
                    candidate_node = self.candidate_nodes_by_transitions[transition]
                    self.multisets[candidate_node].update([tuple(t)])

    def _compute_non_distinct_vertices(self):
        self.start_state, self.character = self.candidate_nodes_to_transitions[
            self.chosen_candidate_node
        ]
        self.non_distinct_vertices = set()
        for v in self.vertices:
            is_distinct = test_distinct(
                self.biggest_multiset, self.vertex2multiset[v], self.params
            )
            if not is_distinct:
                self.non_distinct_vertices.add(v)

    def _add_new_state_or_edge(self):
        all_nodes_are_distinct = len(self.non_distinct_vertices) == 0
        maximum_nb_states_reached = len(self.vertices) == self.params.n
        if all_nodes_are_distinct and not maximum_nb_states_reached:
            # we've got a new node
            new_vertex = len(self.vertices)
            self.vertices.add(new_vertex)
            self.vertex2multiset[new_vertex] = self.biggest_multiset
            self.transitions.setdefault(self.start_state, {})[
                self.character
            ] = new_vertex
        else:
            # pick a safe node that has not distinguished from best candidate.
            # For deterministic behaviour, pick the smallest
            old_vertex = sorted(self.non_distinct_vertices)[0]
            self.transitions.setdefault(self.start_state, {})[
                self.character
            ] = old_vertex
