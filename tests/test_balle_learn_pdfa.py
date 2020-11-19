"""Main test module."""
import numpy as np
import pytest

from src.learn_pdfa.base import Algorithm, learn_pdfa
from src.learn_pdfa.utils.generator import MultiprocessedGenerator, SimpleGenerator
from src.pdfa.render import to_graphviz


def test_learn_pdfa_one_state(pdfa_one_state):
    """Test the PDFA learning, 1 state."""
    automaton = pdfa_one_state
    generator = MultiprocessedGenerator(SimpleGenerator(automaton), nb_processes=8)

    pdfa = learn_pdfa(
        algorithm=Algorithm.BALLE,
        nb_samples=100000,
        sample_generator=generator,
        alphabet_size=automaton.alphabet_size,
        delta=0.01,
        n=10,
    )

    assert len(pdfa.states) == 3


def test_learn_pdfa_two_state(pdfa_two_states):
    """Test the PDFA learning, 2 states."""
    automaton = pdfa_two_states
    generator = MultiprocessedGenerator(SimpleGenerator(automaton), nb_processes=8)

    pdfa = learn_pdfa(
        algorithm=Algorithm.BALLE,
        nb_samples=10000,
        sample_generator=generator,
        alphabet_size=automaton.alphabet_size,
        delta=0.01,
        epsilon=0.01,
        n=10,
    )

    p1 = 0.4
    p2 = 0.7
    to_graphviz(pdfa_two_states, lower_bound=0.0).render("expected")
    to_graphviz(
        pdfa,
    ).render("actual")
    to_graphviz(pdfa, lower_bound=0).render("actual_no_filter")
    assert len(pdfa.states) == 4
    assert np.isclose(pdfa.get_probability([1]), p1, atol=0.4)
    assert np.isclose(pdfa.get_probability([0]), 0.0, atol=0.01)
    assert np.isclose(pdfa.get_probability([0, 1]), p1 * (1 - p2), atol=0.2)
    assert np.isclose(pdfa.get_probability([0, 0, 1]), p1 * p2 * (1 - p2), atol=0.2)


@pytest.mark.parametrize(
    "pdfa_sequence_three_states",
    [
        (0.4, 0.3, 0.2, 0.1),
    ],
    indirect=True,
)
def test_learn_pdfa_sequence_three_states(pdfa_sequence_three_states):
    """
    Test the PDFA learning, with the "ring" PDFA, 3 states.

    Due to same probabilities of two symbols, the three states collapse into one.
    """
    automaton = pdfa_sequence_three_states
    generator = MultiprocessedGenerator(SimpleGenerator(automaton), nb_processes=8)

    pdfa = learn_pdfa(
        algorithm=Algorithm.BALLE,
        nb_samples=50000,
        sample_generator=generator,
        alphabet_size=automaton.alphabet_size,
        delta=0.1,
        n=10,
    )

    assert len(pdfa.states) == 5
