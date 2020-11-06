"""Main test module."""

from src.learn_pdfa.base import Algorithm, learn_pdfa
from src.learn_pdfa.common import MultiprocessedGenerator, SimpleGenerator
from src.pdfa import PDFA


def test_learn_pdfa_1_state():
    """Test the PDFA learning, 1 state."""
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
    generator = MultiprocessedGenerator(SimpleGenerator(automaton), nb_processes=8)

    v, t = learn_pdfa(
        algorithm=Algorithm.BALLE,
        nb_samples=10000,
        sample_generator=generator,
        alphabet_size=2,
        delta=0.1,
        n=3,
    )
    # TODO
    assert len(v) == 3
