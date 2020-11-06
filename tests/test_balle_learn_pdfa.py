"""Main test module."""
import pytest

from src.learn_pdfa.base import Algorithm, learn_pdfa
from src.learn_pdfa.common import MultiprocessedGenerator, SimpleGenerator


def test_learn_pdfa_one_state(pdfa_one_state):
    """Test the PDFA learning, 1 state."""
    automaton = pdfa_one_state
    generator = MultiprocessedGenerator(SimpleGenerator(automaton), nb_processes=8)

    v, t = learn_pdfa(
        algorithm=Algorithm.BALLE,
        nb_samples=10000,
        sample_generator=generator,
        alphabet_size=automaton.alphabet_size,
        delta=0.01,
        n=10,
    )

    assert len(v) == 2
    assert t == {
        0: {0: 0, 1: 1},
    }


def test_learn_pdfa_two_state(pdfa_two_states):
    """Test the PDFA learning, 2 states."""
    automaton = pdfa_two_states
    generator = MultiprocessedGenerator(SimpleGenerator(automaton), nb_processes=8)

    v, t = learn_pdfa(
        algorithm=Algorithm.BALLE,
        nb_samples=10000,
        sample_generator=generator,
        alphabet_size=automaton.alphabet_size,
        delta=0.01,
        n=10,
    )

    assert len(v) == 3
    assert t == {
        0: {0: 2, 1: 1},
        2: {0: 1, 1: 2},
    }


@pytest.mark.parametrize(
    "pdfa_sequence_three_states",
    [
        (0.4, 0.3, 0.2, 0.1),
    ],
    indirect=True,
)
def test_learn_pdfa_three_states(pdfa_sequence_three_states):
    """
    Test the PDFA learning, with the "ring" PDFA, 3 states.

    Due to same probabilities of two symbols, the three states collapse into one.
    """
    automaton = pdfa_sequence_three_states
    generator = MultiprocessedGenerator(SimpleGenerator(automaton), nb_processes=8)

    v, t = learn_pdfa(
        algorithm=Algorithm.BALLE,
        nb_samples=50000,
        sample_generator=generator,
        alphabet_size=automaton.alphabet_size,
        delta=0.1,
        n=10,
    )

    assert len(v) == 4
    assert t == {
        0: {0: 1, 1: 0, 2: 0, 3: 3},
        1: {0: 0, 1: 2, 2: 0, 3: 3},
        2: {0: 0, 1: 0, 2: 3, 3: 3},
    }
