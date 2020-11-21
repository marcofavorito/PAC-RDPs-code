"""Main test module."""

import numpy as np

from src.learn_pdfa.base import Algorithm, learn_pdfa
from src.learn_pdfa.utils.generator import MultiprocessedGenerator, SimpleGenerator


def test_learn_pdfa_1_state(pdfa_one_state, nb_processes):
    """Test the PDFA learning, 1 state."""
    automaton = pdfa_one_state
    expected_p = 0.3
    generator = MultiprocessedGenerator(
        SimpleGenerator(automaton), nb_processes=nb_processes
    )

    pdfa = learn_pdfa(
        algorithm=Algorithm.PALMER,
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


def test_learn_pdfa_2_states(pdfa_two_states, nb_processes):
    """Test the PDFA learn example with 2 states."""
    automaton = pdfa_two_states
    expected_p1 = 0.4
    expected_p2 = 0.7
    generator = MultiprocessedGenerator(
        SimpleGenerator(automaton), nb_processes=nb_processes
    )

    pdfa = learn_pdfa(
        algorithm=Algorithm.PALMER,
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
