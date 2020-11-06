"""Common utilities for the learning PDFA algorithm."""
from abc import ABC, abstractmethod
from collections import Counter
from math import ceil
from multiprocessing import Pool
from typing import Callable, Sequence

from src.pdfa import PDFA
from src.pdfa.types import Word


class Generator(ABC):
    """Wrapper to a PDFA to make sampling as a function call."""

    @abstractmethod
    def sample(self, n: int = 1) -> Sequence[Word]:
        """
        Generate a sample of size n.

        :param n: the size of the sample.
        :return: the list of sampled traces.
        """


class SimpleGenerator(Generator):
    """A simple sample generator."""

    def __init__(self, pdfa: PDFA):
        """Initialize an abstract generator."""
        self._pdfa = pdfa

    def __call__(self):
        """Sample a trace."""
        return self._pdfa.sample()

    def sample(self, n: int = 1) -> Sequence[Word]:
        """Generate a sample of size n."""
        return [self() for _ in range(n)]


class MultiprocessedGenerator(Generator):
    """Generate a sample, multiprocessed."""

    def __init__(self, generator: Generator, nb_processes: int = 4):
        """
        Generate a sample.

        :param nb_processes: the number of processes.
        """
        self._generator = generator
        self._nb_processes = nb_processes
        self._pool = Pool(nb_processes)

    def __call__(self):
        """Sample a trace."""
        return self._generator.sample()

    @staticmethod
    def _job(n: int, sample_func: Callable[[int], Sequence[Word]]):
        return [sample_func(1)[0] for _ in range(n)]

    def sample(self, n: int = 1) -> Sequence[Word]:
        """Generate a sample, multiprocessed."""
        n_per_process = ceil(n / self._nb_processes)
        sample = []

        results = [
            self._pool.apply_async(
                self._job, args=[n_per_process, self._generator.sample]
            )
            for _ in range(self._nb_processes)
        ]
        for r in results:
            n_samples = r.get()
            sample.extend(n_samples)

        nb_samples_to_drop = len(sample) - n
        return sample[: len(sample) - nb_samples_to_drop]


def l_infty_norm(multiset1: Counter, multiset2: Counter) -> float:
    """Compute the supremum distance between two probability distributions."""
    current_max = 0.0
    card1 = sum(multiset1.values())
    card2 = sum(multiset2.values())
    assert card1 > 0, "Cardinality of multiset shouldn't be zero."
    assert card2 > 0, "Cardinality of multiset shouldn't be zero."
    all_strings = set(multiset1).union(multiset2)
    for string in all_strings:
        norm = abs(multiset1[string] / card1 - multiset2[string] / card2)
        current_max = max([norm, current_max])
    return current_max


def prefix_distance_infty_norm(multiset1: Counter, multiset2: Counter) -> float:
    """Compute the supremum distance of prefixes of two probability distributions."""
    current_max = 0.0
    card1 = sum(multiset1.values())
    card2 = sum(multiset2.values())
    assert card1 > 0, "Cardinality of multiset shouldn't be zero."
    assert card2 > 0, "Cardinality of multiset shouldn't be zero."
    all_strings = set(multiset1).union(multiset2)
    for string in all_strings:
        for i in range(len(string)):
            prefix, suffix = string[:i], string[i:]
            d1, d2 = 0.0, 0.0
            for j in range(len(suffix)):
                current_suffix = suffix[:j]
                d1 += multiset1[prefix + current_suffix]
                d2 += multiset2[prefix + current_suffix]

            norm = abs(d1 / card1 - d2 / card2)
            current_max = max([norm, current_max])
    return current_max
