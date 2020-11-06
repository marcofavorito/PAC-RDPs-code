"""Main test module."""

from src.learn_pdfa.base import Algorithm, learn_pdfa
from src.learn_pdfa.common import MultiprocessedGenerator, SimpleGenerator


def test_learn_pdfa_state(pdfa_two_states):
    """Test the PDFA learning, 1 state."""
    automaton = pdfa_two_states
    generator = MultiprocessedGenerator(SimpleGenerator(automaton), nb_processes=8)

    v, t = learn_pdfa(
        algorithm=Algorithm.BALLE,
        nb_samples=10000,
        sample_generator=generator,
        alphabet_size=2,
        delta=0.01,
        n=10,
    )

    assert len(v) == 3
