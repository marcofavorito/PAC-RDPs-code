"""Main test module."""
from abc import abstractmethod

import numpy as np
import pytest

from src.learn_pdfa.base import learn_pdfa
from src.learn_pdfa.utils.generator import (
    Generator,
    MultiprocessedGenerator,
    SimpleGenerator,
)
from src.pdfa import PDFA
from src.pdfa.base import FINAL_STATE


class BaseTestGenerator:
    """Base test class for generators."""

    def make_automaton(self) -> PDFA:
        """Make a PDFA to generate samples from."""
        automaton = PDFA(
            1,
            2,
            {
                0: {
                    0: (0, 0.5),
                    1: (FINAL_STATE, 1 - 0.5),
                }
            },
        )
        return automaton

    @abstractmethod
    def make_generator(self) -> Generator:
        """Make a sample generator. To be implemented."""

    def setup(self):
        """Set up the test."""
        self.generator = self.make_generator()

    @pytest.mark.parametrize("nb_samples", [5, 10, 20, 100])
    def test_generation(self, nb_samples):
        """Test generator 'sample' method."""
        sample = self.generator.sample(n=nb_samples)
        assert len(sample) == nb_samples
        assert all(character in {0, 1} for s in sample for character in s)


class TestSimpleGenerator(BaseTestGenerator):
    """Test simple generator."""

    def make_generator(self) -> Generator:
        """Make a generator for testing."""
        return SimpleGenerator(self.make_automaton())


class TestMultiprocessedGenerator(BaseTestGenerator):
    """Test multiprocessed generator."""

    def make_generator(self) -> Generator:
        """Make a generator for testing."""
        return MultiprocessedGenerator(SimpleGenerator(self.make_automaton()))


def test_multiprocess_generator_helper_function():
    """Test multiprocess generator helper function."""
    automaton = PDFA(
        1,
        2,
        {
            0: {
                0: (0, 0.5),
                1: (FINAL_STATE, 1 - 0.5),
            }
        },
    )
    sample = MultiprocessedGenerator._job(10, SimpleGenerator(automaton).sample)
    assert len(sample) == 10
    assert all(character in {0, 1} for s in sample for character in s)


def test_learn_pdfa_1_state(pdfa_one_state):
    """Test the PDFA learning, 1 state."""
    automaton = pdfa_one_state
    expected_p = 0.3
    generator = MultiprocessedGenerator(SimpleGenerator(automaton), nb_processes=8)

    pdfa = learn_pdfa(
        sample_generator=generator,
        alphabet_size=automaton.alphabet_size,
        epsilon=0.4,
        delta_1=0.2,
        delta_2=0.2,
        mu=0.4,
        n=2,
        n1_max_debug=100000,
        n2_max_debug=100000,
    )

    assert len(pdfa.states) == 1

    transitions_from_initial_state = pdfa.transition_dict[pdfa.initial_state]
    # reading symbol 0
    dest_state, prob = transitions_from_initial_state[0]
    assert dest_state == 0
    assert np.isclose(prob, expected_p, rtol=0.1)

    # reading symbol 1
    dest_state, prob = transitions_from_initial_state[1]
    assert dest_state == pdfa.final_state
    assert np.isclose(prob, 1 - expected_p, rtol=0.1)


def test_learn_pdfa_2_states(pdfa_two_states):
    """Test the PDFA learn example with 2 states."""
    automaton = pdfa_two_states
    expected_p1 = 0.4
    expected_p2 = 0.7
    generator = MultiprocessedGenerator(SimpleGenerator(automaton), nb_processes=8)

    pdfa = learn_pdfa(
        sample_generator=generator,
        alphabet_size=automaton.alphabet_size,
        epsilon=0.4,
        delta_1=0.2,
        delta_2=0.2,
        mu=0.1,
        n=3,
        n1_max_debug=100000,
        n2_max_debug=100000,
        m0_max_debug=100000 / 10,
    )

    assert len(pdfa.states) == 2

    # test transitions from initial state
    transitions_from_initial_state = pdfa.transition_dict[pdfa.initial_state]
    # reading symbol 0
    dest_state, prob = transitions_from_initial_state[0]
    assert dest_state == 1
    assert np.isclose(prob, expected_p1, rtol=0.1)
    # reading symbol 1
    dest_state, prob = transitions_from_initial_state[1]
    assert dest_state == pdfa.final_state
    assert np.isclose(prob, 1 - expected_p1, rtol=0.1)

    # test transitions from second state
    transitions_from_second_state = pdfa.transition_dict[1]
    # reading symbol 0
    dest_state, prob = transitions_from_second_state[0]
    assert dest_state == pdfa.final_state
    assert np.isclose(prob, 1 - expected_p2, rtol=0.1)
    # reading symbol 1
    dest_state, prob = transitions_from_second_state[1]
    assert dest_state == 1
    assert np.isclose(prob, expected_p2, rtol=0.1)
