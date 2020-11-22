"""Entrypoint for the algorithm."""

import pprint
from abc import ABC
from math import log, sqrt
from typing import Dict, Optional, Sequence, Tuple, Type, cast

from src.helpers.base import normalize
from src.learn_pdfa import logger
from src.learn_pdfa.balle.params import BalleParams
from src.learn_pdfa.utils.base import (
    MultisetLike,
    get_prefix_probability,
    l_infty_norm,
    prefix_distance_infty_norm,
)
from src.learn_pdfa.utils.multiset import Counter as ConcreteMultiset  # noqa: ignore
from src.learn_pdfa.utils.multiset import (  # noqa: ignore
    Multiset,
    NaiveMultiset,
    Node,
    PrefixTreeMultiset,
    ReadOnlyPrefixTreeMultiset,
    node_to_graphviz,
)
from src.pdfa import PDFA
from src.pdfa.base import FINAL_STATE, FINAL_SYMBOL
from src.types import Character, State, TransitionFunctionDict, Word


def learn_pdfa(**kwargs) -> PDFA:
    """
    PAC-learn a PDFA.

    This is a wrapper function to the 'Learner' class, defined below.

    :param kwargs: the keyword arguments of the algorithm (see the BalleParams class).
    :return: the learnt PDFA.
    """
    params = BalleParams(**kwargs)
    logger.info(f"Parameters: {pprint.pformat(str(params))}")
    automaton = Learner(params).learn()
    return automaton


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
    multiset_candidate: MultisetLike, multiset_safe: MultisetLike, params: BalleParams
) -> Tuple[float, float]:
    """
    Test whether a candidate state is distinct from a safe node.

    :param multiset_candidate: the multiset of the candidate node.
    :param multiset_safe: the multiset of the safe node.
    :param params: the algorithm parameters.
    :return: a pair of floats: the distance and the distinctness threshold
    """
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
    return distance, threshold


class Learner(ABC):
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
        self.done = False
        self.iteration = 0
        self.iteration_upper_bound = self.params.n * self.params.alphabet_size
        self.samples = []
        self.multiset_manager: Optional[SampleMultisetManager] = None

    def _sample(self):
        """Do the sampling."""
        logger.info("Generating the sample.")
        generator = self.params.sample_generator
        self.samples = generator.sample(n=self.params.nb_samples, with_final=True)
        self.samples = list(map(lambda x: tuple(x), self.samples))
        self.expected_trace_length = sum(map(len, self.samples)) / len(self.samples)
        logger.info("Populate root multiset.")

        self.multiset_manager = SampleMultisetManager(self.params, self.samples)

    def learn(self) -> PDFA:
        """
        Do the learning.

        This is the main entry-point of the class.
        """
        self._initialize_variables()
        self._sample()
        while not self.done:
            self._do_iteration()

        self._complete_graph()
        self._compute_probabilities()
        return PDFA(len(self.vertices), len(self.alphabet), self.pdfa_transitions)

    def _do_iteration(self):
        """Do one iteration."""
        logger.info(f"Iteration {self.iteration}")

        # compute candidate nodes. If none, terminate.
        self.multiset_manager.reset_candidate_nodes(
            self.vertices, self.alphabet, self.transitions
        )

        no_candidate_nodes = (
            len(self.multiset_manager.candidate_nodes_to_transitions) == 0
        )
        if no_candidate_nodes:
            self.done = True
            return

        # compute multisets
        logger.debug("Computing multisets...")
        self.multiset_manager.compute_multisets(self.transitions)

        (
            self.chosen_candidate_node,
            self.biggest_multiset,
        ) = self.multiset_manager.biggest_multiset()

        if sum(self.biggest_multiset.values()) == 0:
            logger.info("Biggest multiset has cardinality 0, done")
            self.done = True
            return

        # compute non-distinct vertices, and add a new state or just a new edge
        logger.info("Compute non-distinct vertices...")
        self._compute_non_distinct_vertices()
        logger.info("Add new state/new node")
        self._add_new_state_or_edge()

        # end of iteration
        self.iteration += 1
        self.done = self.iteration >= self.iteration_upper_bound

    def _compute_non_distinct_vertices(self):
        (
            self.start_state,
            self.character,
        ) = self.multiset_manager.candidate_nodes_to_transitions[
            self.chosen_candidate_node
        ]
        self.non_distinct_vertices: Dict[int, Tuple[float, float]] = {}
        for v in self.vertices:
            distance, threshold = self.multiset_manager.test_distinct(
                self.chosen_candidate_node, v
            )
            is_distinct = distance > threshold
            if not is_distinct:
                self.non_distinct_vertices[v] = (distance, threshold)

    def _add_new_state_or_edge(self):
        all_nodes_are_distinct = len(self.non_distinct_vertices) == 0
        maximum_nb_states_reached = len(self.vertices) == self.params.n
        if all_nodes_are_distinct and not maximum_nb_states_reached:
            # we've got a new node
            new_vertex = len(self.vertices)
            self.vertices.add(new_vertex)
            self.multiset_manager.add_vertex(new_vertex, self.biggest_multiset)
            self.transitions.setdefault(self.start_state, {})[
                self.character
            ] = new_vertex
        else:
            # pick a safe node that has not distinguished from best candidate.
            # For deterministic behaviour, pick the smallest
            sorted_non_distinct_vertices = sorted(self.non_distinct_vertices.keys())
            if len(sorted_non_distinct_vertices) > 1:
                logger.warning(
                    f"More than one non-distinct vertex: {sorted_non_distinct_vertices}"
                )
                logger.warning(
                    f"Distances and thresholds: {pprint.pformat(self.non_distinct_vertices)}"
                )
            old_vertex = sorted_non_distinct_vertices[0]
            self.transitions.setdefault(self.start_state, {})[
                self.character
            ] = old_vertex

    def _complete_graph(self):
        """
        Complete graph.

        Add a ground node (only if needed, and if allowed by params), and a final node.
        """
        if self.params.with_ground:
            self._add_ground_node()

        final_node = FINAL_STATE
        for vertex in self.vertices:
            self.transitions.setdefault(vertex, {})[FINAL_SYMBOL] = final_node

    def _compute_probabilities(self):
        """Given vertices, transitions and its multisets, estimate edge probabilities."""
        self.pdfa_transitions: TransitionFunctionDict = {}

        # compute gammas
        for start, out_transitions in self.transitions.items():
            self.pdfa_transitions[start] = {}
            for character, next_state in out_transitions.items():
                probability = self._compute_edge_probability(start, character)
                self.pdfa_transitions[start][character] = (next_state, probability)

        # normalize
        self.pdfa_transitions = normalize(self.pdfa_transitions)

    def _compute_edge_probability(self, state: int, character: int):
        """Given state and character, compute probability."""
        multiset = self.multiset_manager.vertex2multiset.get(state, ConcreteMultiset())  # type: ignore
        size = sum(multiset.values())
        smoothing_probability = (
            self.params.get_gamma_min(self.expected_trace_length)
            if self.params.with_smoothing
            else 0.0
        )
        if size == 0:
            return self._params.get_gamma_min(self.expected_trace_length)
        char_prob = get_prefix_probability(multiset, (character,))
        factor = 1 - (self.params.alphabet_size + 1) * smoothing_probability
        return char_prob * factor + smoothing_probability

    def _add_ground_node(self):
        """Add a ground node."""
        ground_node = len(self.vertices)
        ground_node_used = False

        for vertex in self.vertices:
            transitions_from_vertex = self.transitions.get(vertex, {})
            for character in self.alphabet:
                if character not in transitions_from_vertex:
                    ground_node_used = True
                    transitions_from_vertex[character] = ground_node
            self.transitions[vertex] = transitions_from_vertex

        if ground_node_used:
            self.vertices.add(ground_node)
            self.transitions[ground_node] = {}
            for character in self.alphabet:
                self.transitions[ground_node][character] = ground_node


class SampleMultisetManager:
    """Sample multiset manager."""

    MULTISET_CLASS: Type[MultisetLike] = NaiveMultiset

    def __init__(self, params: BalleParams, sample: Sequence[Word]):
        """Initialize."""
        self.params = params
        self.main_multiset = self.MULTISET_CLASS()
        self.main_multiset.update(sample)

        self.vertex2multiset: Dict[int, MultisetLike] = {}
        self.vertex2multiset[0] = self.main_multiset

        # temporary
        self.candidate_nodes_by_transitions: Dict[Tuple[State, Character], int] = {}
        self.candidate_nodes_to_transitions: Dict[int, Tuple[State, Character]] = {}
        self.multisets: Dict[int, ConcreteMultiset] = {}

    def reset_candidate_nodes(self, vertices, alphabet, transitions: Dict):
        """Reset multisets."""
        self.candidate_nodes_by_transitions = {}
        self.candidate_nodes_to_transitions = {}
        self.multisets = {}

        # recompute candidate nodes
        for v in vertices:
            for c in alphabet:
                if transitions.get(v, {}).get(c) is None:  # if transition undefined
                    transition = (v, c)
                    new_candidate = len(vertices) + len(
                        self.candidate_nodes_to_transitions
                    )
                    self.candidate_nodes_to_transitions[new_candidate] = transition
                    self.candidate_nodes_by_transitions[transition] = new_candidate
                    self.multisets[new_candidate] = self.MULTISET_CLASS()  # type: ignore

    def compute_multisets(self, transitions):
        """Compute multisets for the current iteration."""
        for word in self.main_multiset.elements():
            # s is always non-empty
            for i in range(len(word)):
                r, sigma, t = word[:i], word[i], word[i + 1 :]
                q = extended_transition_fun(transitions, r)
                if q is None:
                    continue
                transition = (q, sigma)
                if transition in self.candidate_nodes_by_transitions:
                    candidate_node = self.candidate_nodes_by_transitions[transition]
                    self.multisets[candidate_node].update([tuple(t)])

    def biggest_multiset(self) -> Tuple[int, ConcreteMultiset]:
        """Compute the biggest multiset."""
        return max(self.multisets.items(), key=lambda x: sum(x[1].values()))

    def test_distinct(self, chosen_candidate_node: int, v: int):
        """Test distinctness of two vertices."""
        distance, threshold = test_distinct(
            self.multisets[chosen_candidate_node], self.vertex2multiset[v], self.params
        )
        return distance, threshold

    def add_vertex(self, new_vertex, biggest_multiset):
        """Add a vertex to the multiset manager."""
        self.vertex2multiset[new_vertex] = biggest_multiset


class SampleMultisetManagerWithPrefix(SampleMultisetManager):
    """Multiset manager, but with Prefix trees."""

    MULTISET_CLASS = PrefixTreeMultiset

    def reset_candidate_nodes(self, *args):
        """Reset multisets."""
        super().reset_candidate_nodes(*args)
        self.multisets: Dict[int, Multiset] = {}

    def compute_multisets(self, transitions):
        """DFS, non-recursive."""
        self.main_multiset = cast(PrefixTreeMultiset, self.main_multiset)
        root = self.main_multiset._node
        pdfa_initial_state = 0

        def _visit(node: Node, state: State):
            outgoing_from_last: Dict[Character, int] = transitions.get(state, {})
            next_transitions = node.next_transitions()
            for c, n in next_transitions:
                transition = (state, c)
                if transition in self.candidate_nodes_by_transitions:
                    candidate_node = self.candidate_nodes_by_transitions[transition]
                    multiset = self.multisets.get(candidate_node)
                    if multiset is None:
                        self.multisets[candidate_node] = ReadOnlyPrefixTreeMultiset({n})
                    else:
                        cast(ReadOnlyPrefixTreeMultiset, multiset)._nodes.add(n)
                elif c != FINAL_SYMBOL:
                    _visit(n, outgoing_from_last[c])

        _visit(root, pdfa_initial_state)
