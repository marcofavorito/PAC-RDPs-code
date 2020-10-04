"""Main test module."""
import numpy as np

from src.learn_pdfa.base import learn_pdfa
from src.learn_pdfa.common import MultiprocessedGenerator
from src.pdfa import PDFA


def test_learn_pdfa_example():
    """Test the PDFA learn example."""
    p = 0.5
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
    )

    assert len(pdfa.states) == 1

    transitions_from_initial_state = pdfa.transition_dict[pdfa.initial_state]
    # reading symbol 0
    dest_state, prob = transitions_from_initial_state[0]
    assert dest_state == 0
    assert np.isclose(prob, p, rtol=0.01)

    # reading symbol 1
    dest_state, prob = transitions_from_initial_state[1]
    assert dest_state == 1
    assert np.isclose(prob, 1 - p, rtol=0.01)
