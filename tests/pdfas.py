"""Definition of PDFAs."""
import pytest

from src.pdfa import PDFA
from src.pdfa.base import FINAL_STATE
from src.pdfa.helpers import FINAL_SYMBOL


def make_pdfa_one_state(p: float = 0.3):
    """Make a PDFA with one state, for testing purposes."""
    automaton = PDFA(
        2,
        2,
        {
            0: {
                0: (0, p),
                1: (1, 1 - p),
            },
            1: {FINAL_SYMBOL: (FINAL_STATE, 1.0)},
        },
    )
    return automaton


@pytest.fixture
def pdfa_one_state():
    """Get a PDFA with one state."""
    return make_pdfa_one_state()


def make_pdfa_two_state(p1: float = 0.4, p2: float = 0.7):
    """Make a PDFA with two states, for testing purposes."""
    automaton = PDFA(
        3,
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
            2: {FINAL_SYMBOL: (FINAL_STATE, 1.0)},
        },
    )
    return automaton


@pytest.fixture
def pdfa_two_states():
    """Get a PDFA with two states."""
    return make_pdfa_two_state()


def make_pdfa_sequence_three_states(
    p1: float, p2: float, p3: float, stop_probability: float
):
    """Make a PDFA with three states, for testing purposes."""
    automaton = PDFA(
        4,
        3,
        {
            0: {
                0: (1, p1),
                1: (0, p2),
                2: (0, p3),
                FINAL_SYMBOL: (FINAL_STATE, stop_probability),
            },
            1: {
                0: (0, p1),
                1: (2, p2),
                2: (0, p3),
                FINAL_SYMBOL: (FINAL_STATE, stop_probability),
            },
            2: {
                0: (0, p1),
                1: (0, p2),
                2: (3, p3),
                FINAL_SYMBOL: (FINAL_STATE, stop_probability),
            },
            3: {
                FINAL_SYMBOL: (FINAL_STATE, 1.0),
            },
        },
    )
    return automaton


@pytest.fixture
def pdfa_sequence_three_states(request):
    """Get a PDFA with two states."""
    p1, p2, p3, stop_probability = request.param
    return make_pdfa_sequence_three_states(p1, p2, p3, stop_probability)
