"""Test generator."""
from abc import abstractmethod

import pytest

from tests.pdfas import make_pdfa_one_state

from src.learn_pdfa.utils.generator import (
    Generator,
    MultiprocessedGenerator,
    SimpleGenerator,
)
from src.pdfa import PDFA
from src.pdfa.helpers import FINAL_SYMBOL


class BaseTestGenerator:
    """Base test class for generators."""

    def make_automaton(self) -> PDFA:
        """Make a PDFA to generate samples from."""
        automaton = make_pdfa_one_state()
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
        assert all(character in {0, 1, FINAL_SYMBOL} for s in sample for character in s)


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
    automaton = make_pdfa_one_state()
    sample = MultiprocessedGenerator._job(10, False, SimpleGenerator(automaton).sample)
    assert len(sample) == 10
    assert all(character in {0, 1, FINAL_SYMBOL} for s in sample for character in s)
