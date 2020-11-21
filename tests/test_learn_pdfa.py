"""Main test module."""
import numpy as np
import pytest

from src.learn_pdfa.base import Algorithm, learn_pdfa
from src.learn_pdfa.utils.generator import MultiprocessedGenerator, SimpleGenerator

BALLE_CONFIG = dict(
    algorithm=Algorithm.BALLE,
    nb_samples=100000,
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


@pytest.mark.parametrize(
    "config,expected_nb_states", [(PALMER_CONFIG, 1), (BALLE_CONFIG, 2)]
)
def test_learn_pdfa_one_state(pdfa_one_state, nb_processes, config, expected_nb_states):
    """Test the PDFA learning, 1 state."""
    automaton = pdfa_one_state
    expected_p = 0.3
    generator = MultiprocessedGenerator(
        SimpleGenerator(automaton), nb_processes=nb_processes
    )

    pdfa = learn_pdfa(
        sample_generator=generator, alphabet_size=automaton.alphabet_size, **config
    )

    assert len(pdfa.states) == expected_nb_states
    assert np.isclose(pdfa.get_probability([0]), 0.0, atol=0.1)
    assert np.isclose(
        pdfa.get_probability([0, 1]), (expected_p * (1 - expected_p)), atol=0.1
    )
    assert np.isclose(pdfa.get_probability([1]), 1 - expected_p, atol=0.1)


@pytest.mark.parametrize(
    "config,expected_nb_states", [(PALMER_CONFIG, 2), (BALLE_CONFIG, 3)]
)
def test_learn_pdfa_two_state(pdfa_two_states, config, expected_nb_states):
    """Test the PDFA learning, 2 states."""
    automaton = pdfa_two_states
    generator = MultiprocessedGenerator(SimpleGenerator(automaton), nb_processes=8)

    pdfa = learn_pdfa(
        sample_generator=generator, alphabet_size=automaton.alphabet_size, **config
    )

    p1 = 0.4
    p2 = 0.7
    assert len(pdfa.states) == expected_nb_states
    assert np.isclose(pdfa.get_probability([1]), p1, atol=0.4)
    assert np.isclose(pdfa.get_probability([0]), 0.0, atol=0.01)
    assert np.isclose(pdfa.get_probability([0, 1]), p1 * (1 - p2), atol=0.2)
    assert np.isclose(pdfa.get_probability([0, 0, 1]), p1 * p2 * (1 - p2), atol=0.2)


# TODO add (PALMER_CONFIG, 3),
@pytest.mark.parametrize(
    "config,expected_nb_states",
    [(BALLE_CONFIG, 4)],
)
@pytest.mark.parametrize(
    "pdfa_sequence_three_states",
    [
        (0.4, 0.3, 0.2, 0.1),
    ],
    indirect=True,
)
def test_learn_pdfa_sequence_three_states(
    pdfa_sequence_three_states, config, expected_nb_states
):
    """
    Test the PDFA learning, with the "ring" PDFA, 3 states.

    Due to same probabilities of two symbols, the three states collapse into one.
    """
    automaton = pdfa_sequence_three_states
    generator = MultiprocessedGenerator(SimpleGenerator(automaton), nb_processes=8)

    pdfa = learn_pdfa(
        sample_generator=generator, alphabet_size=automaton.alphabet_size, **config
    )

    assert 4 == len(pdfa.states)
