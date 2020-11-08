"""Tests for the prefix-tree based multiset implementation."""
import pytest
from hypothesis import given, settings, strategies

from src.learn_pdfa.utils.multiset import NaiveMultiset, PrefixTreeMultiset


@pytest.mark.parametrize("multiset_class", [NaiveMultiset, PrefixTreeMultiset])
def test_multiset(multiset_class):
    """Test multiset."""
    multiset = multiset_class()
    assert multiset.size == 0
    assert multiset.traces == set()
    assert multiset.items() == set()
    assert multiset.get_probability((0, 1)) == 0.0
    assert multiset.get_prefix_probability((0,)) == 0.0

    multiset.add((0,))
    assert multiset.size == 1
    assert multiset.traces == {(0,)}
    assert multiset.items() == {((0,), 1)}
    assert multiset.get_probability((0,)) == 1.0
    assert multiset.get_prefix_probability((0,)) == 1.0

    multiset.add((0, 1))
    assert multiset.size == 2
    assert multiset.traces == {(0,), (0, 1)}
    assert multiset.items() == {((0,), 1), ((0, 1), 1)}
    assert multiset.get_probability((0,)) == 0.5
    assert multiset.get_probability((0, 1)) == 0.5
    assert multiset.get_prefix_probability((0,)) == 1.0
    assert multiset.get_prefix_probability((0, 1)) == 1 / 2

    multiset.add((0,))
    assert multiset.size == 3
    assert multiset.traces == {(0,), (0, 1)}
    assert multiset.items() == {((0,), 2), ((0, 1), 1)}
    assert multiset.get_probability((0,)) == 2 / 3
    assert multiset.get_probability((0, 1)) == 1 / 3
    assert multiset.get_prefix_probability((0,)) == 1.0
    assert multiset.get_prefix_probability((0, 1)) == 1 / 3

    multiset.add((1,))
    assert multiset.size == 4
    assert multiset.traces == {(0,), (1,), (0, 1)}
    assert multiset.items() == {((0,), 2), ((1,), 1), ((0, 1), 1)}
    assert multiset.get_probability((0,)) == 1 / 2
    assert multiset.get_probability((1,)) == 1 / 4
    assert multiset.get_probability((0, 1)) == 1 / 4
    assert multiset.get_prefix_probability((0,)) == 3 / 4
    assert multiset.get_prefix_probability((0, 1)) == 1 / 4
    assert multiset.get_prefix_probability((1,)) == 1 / 4


@given(
    samples=strategies.lists(
        strategies.lists(
            strategies.integers(min_value=0, max_value=4), min_size=0, max_size=100
        ),
        min_size=0,
        max_size=1000,
    )
)
@settings(max_examples=1000)
def test_naive_multiset_and_prefix_based_multiset_equivalent(samples):
    """Test equivalence between two multiset implementations."""
    multiset_1 = NaiveMultiset()
    multiset_2 = PrefixTreeMultiset()

    for s in samples:
        s = tuple(s)
        multiset_1.add(s)
        multiset_2.add(s)

    assert multiset_1.size == multiset_2.size
    assert multiset_1.traces == multiset_2.traces
    assert multiset_1.items() == multiset_2.items()

    for s in samples:
        s = tuple(s)
        assert multiset_1.get_counts(s) == multiset_2.get_counts(s)
        assert multiset_1.get_probability(s) == multiset_2.get_probability(s)
        assert multiset_1.get_prefix_probability(
            s
        ) == multiset_1.get_prefix_probability(s)
