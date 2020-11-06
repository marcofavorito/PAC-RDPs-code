"""Definition of PDFAs."""
import pytest

from src.pdfa import PDFA


@pytest.fixture
def pdfa_one_state():
    """Get a PDFA with one state."""
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
    return automaton


@pytest.fixture
def pdfa_two_states():
    """Get a PDFA with two states."""
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
    return automaton
