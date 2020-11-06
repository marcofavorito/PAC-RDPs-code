"""Base module for the learn pdfa implementation."""
from enum import Enum
from typing import Callable, Dict

from src.learn_pdfa.balle.core import learn_pdfa as balle_learn_pdfa
from src.learn_pdfa.palmer.core import learn_pdfa as palmer_learn_pdfa


class Algorithm(Enum):
    """Enumeration of supported PAC learning algorithms for PDFAs."""

    PALMER = "palmer"
    BALLE = "balle"


_algorithm_to_function: Dict[Algorithm, Callable] = {
    Algorithm.PALMER: palmer_learn_pdfa,
    Algorithm.BALLE: balle_learn_pdfa,
}


def learn_pdfa(algorithm: Algorithm = Algorithm.PALMER, **kwargs):
    """
    PAC-learn a PDFA.

    :param kwargs: the keyword arguments of the algorithm.
    :return: the learnt PDFA.
    """
    return _algorithm_to_function[algorithm](**kwargs)
