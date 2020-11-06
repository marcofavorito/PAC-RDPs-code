"""Test generator."""
from abc import abstractmethod

import pytest

from src.learn_pdfa.common import Generator, MultiprocessedGenerator, SimpleGenerator
from src.pdfa import PDFA
from src.pdfa.base import FINAL_STATE


class BaseTestGenerator:
    """Base test class for generators."""

    def make_automaton(self) -> PDFA:
        """Make a PDFA to generate samples from."""
        automaton = PDFA(
            1,
            2,
            {
                0: {
                    0: (0, 0.5),
                    1: (FINAL_STATE, 1 - 0.5),
                }
            },
        )
        return automaton

    @abstractmethod
    def make_generator(self) -> Generator:
        """Make a sample generator. To be implemented."""

    def setup(self):
        """Set up the test."""
        self.generator = self.make_generator()

    @pytest.mark.parametrize("nb_samples", [5, 10, 20, 100])
    def test_generation(self, nb_samples):
        """Test generator 'sample' method."""
        sample = self.generator.sample(n=nb_samples)
        assert len(sample) == nb_samples
        assert all(character in {0, 1} for s in sample for character in s)


class TestSimpleGenerator(BaseTestGenerator):
    """Test simple generator."""

    def make_generator(self) -> Generator:
        """Make a generator for testing."""
        return SimpleGenerator(self.make_automaton())


class TestMultiprocessedGenerator(BaseTestGenerator):
    """Test multiprocessed generator."""

    def make_generator(self) -> Generator:
        """Make a generator for testing."""
        return MultiprocessedGenerator(SimpleGenerator(self.make_automaton()))


def test_multiprocess_generator_helper_function():
    """Test multiprocess generator helper function."""
    automaton = PDFA(
        1,
        2,
        {
            0: {
                0: (0, 0.5),
                1: (FINAL_STATE, 1 - 0.5),
            }
        },
    )
    sample = MultiprocessedGenerator._job(10, SimpleGenerator(automaton).sample)
    assert len(sample) == 10
    assert all(character in {0, 1} for s in sample for character in s)
