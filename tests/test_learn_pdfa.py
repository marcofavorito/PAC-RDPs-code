"""Main test module."""
import numpy as np

from src.learn_pdfa.base import learn_pdfa
from src.learn_pdfa.common import MultiprocessedGenerator
from src.pdfa import PDFA


def test_learn_pdfa_1_state():
    """Test the PDFA learning, 1 state."""
    p = 0.3
    automaton = PDFA(
        1,
        2,
        {
            0: {
                0: (0, p),
                1: (1, 1 - p),
            }
        },
    )
    generator = MultiprocessedGenerator(automaton, nb_processes=8)

    pdfa = learn_pdfa(
        sample_generator=generator,
        alphabet_size=2,
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
    assert np.isclose(prob, p, rtol=0.1)

    # reading symbol 1
    dest_state, prob = transitions_from_initial_state[1]
    assert dest_state == 1
    assert np.isclose(prob, 1 - p, rtol=0.1)


def test_learn_pdfa_2_states():
    """Test the PDFA learn example with 2 states."""
    p1 = 0.4
    p2 = 0.7
    automaton = PDFA(
        2,
        2,
        {
            0: {
                0: (1, p1),
                1: (2, 1 - p1),
            },
            1: {
                0: (2, 1 - p2),
                1: (1, p2),
            },
        },
    )
    generator = MultiprocessedGenerator(automaton, nb_processes=8)

    pdfa = learn_pdfa(
        sample_generator=generator,
        alphabet_size=2,
        epsilon=0.4,
        delta_1=0.2,
        delta_2=0.2,
        mu=0.1,
        n=3,
        n1_max_debug=3000000,
        n2_max_debug=1000000,
        m0_max_debug=3000000 / 10,
    )

    assert len(pdfa.states) == 2

    # test transitions from initial state
    transitions_from_initial_state = pdfa.transition_dict[pdfa.initial_state]
    # reading symbol 0
    dest_state, prob = transitions_from_initial_state[0]
    assert dest_state == 1
    assert np.isclose(prob, p1, rtol=0.01)
    # reading symbol 1
    dest_state, prob = transitions_from_initial_state[1]
    assert dest_state == 2
    assert np.isclose(prob, 1 - p1, rtol=0.01)

    # test transitions from second state
    transitions_from_second_state = pdfa.transition_dict[1]
    # reading symbol 0
    dest_state, prob = transitions_from_second_state[0]
    assert dest_state == 2
    assert np.isclose(prob, 1 - p2, rtol=0.01)
    # reading symbol 1
    dest_state, prob = transitions_from_second_state[1]
    assert dest_state == 1
    assert np.isclose(prob, p2, rtol=0.01)
