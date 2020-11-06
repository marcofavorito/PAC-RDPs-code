"""Definition of PDFAs."""
import pytest

from src.pdfa import PDFA
from src.pdfa.base import FINAL_STATE


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
                1: (FINAL_STATE, 1 - p),
            }
        },
    )
    return automaton


@pytest.fixture
def pdfa_two_states():
    """Get a PDFA with two states."""
    p1 = 0.4
    p2 = 0.7
    automaton = PDFA(
        2,
        2,
        {
            0: {
                0: (1, p1),
                1: (FINAL_STATE, 1 - p1),
            },
            1: {
                0: (FINAL_STATE, 1 - p2),
                1: (1, p2),
            },
        },
    )
    return automaton


@pytest.fixture
def pdfa_sequence_three_states(request):
    """Get a PDFA with two states."""
    p1, p2, p3, stop_probability = request.param
    automaton = PDFA(
        3,
        4,
        {
            0: {
                0: (1, p1),
                1: (0, p2),
                2: (0, p3),
                3: (FINAL_STATE, stop_probability),
            },
            1: {
                0: (0, p1),
                1: (2, p2),
                2: (0, p3),
                3: (FINAL_STATE, stop_probability),
            },
            2: {
                0: (0, p1),
                1: (0, p2),
                2: (FINAL_STATE, p3),
                3: (FINAL_STATE, stop_probability),
            },
        },
    )
    return automaton
