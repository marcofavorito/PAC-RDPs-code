"""Main test module."""
from abc import abstractmethod
from copy import copy
from typing import Dict

import numpy as np
from hypothesis import assume, given, strategies

from tests.pdfas import (
    make_pdfa_one_state,
    make_pdfa_sequence_three_states,
    make_pdfa_two_state,
)

from src.learn_pdfa.base import Algorithm, learn_pdfa
from src.learn_pdfa.utils.generator import MultiprocessedGenerator, SimpleGenerator
from src.pdfa import PDFA
from src.pdfa.helpers import FINAL_SYMBOL

BALLE_CONFIG = dict(
    algorithm=Algorithm.BALLE,
    nb_samples=5000,
    delta=0.1,
    n=10,
)

PALMER_CONFIG = dict(
    algorithm=Algorithm.PALMER,
    epsilon=0.4,
    delta_1=0.2,
    delta_2=0.2,
    mu=0.4,
    n=5,
    n1_max_debug=100000,
    n2_max_debug=100000,
    m0_max_debug=100000 / 10,
)

RTOL = 0.4


class BaseTestLearnPDFA:
    """Base test class for PDFA learning."""

    NB_PROCESSES = 8
    CONFIG: Dict = BALLE_CONFIG
    ALPHABET_LEN = 3
    OVERWRITE_CONFIG: Dict = {}

    @classmethod
    @abstractmethod
    def _make_automaton(cls) -> PDFA:
        """Make automaton."""

    @classmethod
    def setup_class(cls):
        """Set up the test."""
        cls.expected = cls._make_automaton()
        generator = MultiprocessedGenerator(
            SimpleGenerator(cls.expected), nb_processes=cls.NB_PROCESSES
        )

        config = copy(cls.CONFIG)
        config.update(cls.OVERWRITE_CONFIG)
        cls.actual = learn_pdfa(
            sample_generator=generator,
            alphabet_size=cls.expected.alphabet_size,
            **config
        )

    def test_same_nb_states(self):
        """Test same number of states."""
        assert len(self.expected.states) == len(self.actual.states)

    @given(
        trace=strategies.lists(
            strategies.integers(min_value=0, max_value=ALPHABET_LEN - 1),
            min_size=0,
            max_size=100,
        )
    )
    def test_equivalence(self, trace):
        """Test equivalence between expected and learned PDFAs."""
        max_value = self.expected.alphabet_size - 1
        assume(all(x <= max_value for x in trace))
        actual_trace = tuple(trace) + (FINAL_SYMBOL,)
        actual_prob = self.actual.get_probability(actual_trace)
        expected_prob = self.expected.get_probability(actual_trace)
        assert np.isclose(expected_prob, actual_prob, rtol=RTOL)


class TestOneState(BaseTestLearnPDFA):
    """Test PDFA learning of one state PDFA."""

    ALPHABET_LEN = 2

    @classmethod
    def _make_automaton(cls) -> PDFA:
        """Make automaton."""
        return make_pdfa_one_state()


class TestTwoState(BaseTestLearnPDFA):
    """Test PDFA learning of two state PDFA."""

    ALPHABET_LEN = 2

    @classmethod
    def _make_automaton(cls) -> PDFA:
        """Make automaton."""
        return make_pdfa_two_state()


class TestSequenceThreeStates(BaseTestLearnPDFA):
    """Test PDFA learning of two state PDFA."""

    PROBABILITIES = (0.4, 0.3, 0.2, 0.1)
    ALPHABET_LEN = 3
    OVERWRITE_CONFIG = dict(nb_samples=75000)

    @classmethod
    def _make_automaton(cls) -> PDFA:
        """Make automaton."""
        return make_pdfa_sequence_three_states(*cls.PROBABILITIES)
