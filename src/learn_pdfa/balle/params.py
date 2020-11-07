"""Params class for Balle's algorithm."""

from dataclasses import dataclass

from src.helpers.base import assert_
from src.learn_pdfa.utils.generator import Generator


@dataclass(frozen=True)
class BalleParams:
    """
    Parameters for the (Balel et al., 2013) learning algorithm.

    sample_generator: the sample generator from the true PDFA.
    alphabet_size: the alphabet size.
    epsilon: the tolerance error.
    delta: the failure probability for the subgraph construction.
    delta: the failure probability for the probability estimation.
    mu: the distinguishability factor.
    n: the upper bound of the number of states.
    """

    sample_generator: Generator
    nb_samples: int
    n: int
    alphabet_size: int
    delta: float = 0.1

    def __post_init__(self):
        """Validate inputs."""
        assert_(
            0 < self.delta < 1.0,
            "Delta must be a non-zero probaility.",
        )

    @property
    def delta_0(self) -> float:
        """Get the error probability of test distinct."""
        d = self.delta
        s = self.alphabet_size
        n = self.n
        return d / (n * (n * s + s + 1))
