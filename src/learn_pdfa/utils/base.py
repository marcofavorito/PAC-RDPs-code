"""Base module for miscellaneous utilities."""
from collections import Counter
from functools import singledispatch
from typing import Union

from src.learn_pdfa.utils.multiset import Multiset, NaiveMultiset, PrefixTreeMultiset
from src.types import Word


def l_infty_norm(
    multiset1: Union[Multiset, Counter], multiset2: Union[Multiset, Counter]
) -> float:
    """Compute the supremum distance between two probability distributions."""
    current_max = 0.0
    card1 = sum(multiset1.values())
    card2 = sum(multiset2.values())
    assert card1 > 0, "Cardinality of multiset shouldn't be zero."
    assert card2 > 0, "Cardinality of multiset shouldn't be zero."
    all_strings = set(multiset1).union(multiset2)
    for string in all_strings:
        norm = abs(
            get_probability(multiset1, string) - get_probability(multiset2, string)
        )
        current_max = max([norm, current_max])
    return current_max


def prefix_distance_infty_norm(
    multiset1: Union[Multiset, Counter], multiset2: Union[Multiset, Counter]
) -> float:
    """Compute the supremum distance of prefixes of two probability distributions."""
    current_max = 0.0
    card1 = sum(multiset1.values())
    card2 = sum(multiset2.values())
    assert card1 > 0, "Cardinality of multiset shouldn't be zero."
    assert card2 > 0, "Cardinality of multiset shouldn't be zero."
    all_strings = set(multiset1).union(multiset2)
    for string in all_strings:
        string = tuple(string)
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


@singledispatch
def get_probability(multiset, trace) -> float:
    """Get the probability of a trace against a multiset."""


@get_probability.register(NaiveMultiset)  # type: ignore
def _(multiset, trace) -> float:
    return multiset.get_probability(trace)


@get_probability.register(PrefixTreeMultiset)  # type: ignore
def _(multiset, trace) -> float:
    return multiset.get_probability(trace)


@get_probability.register(Counter)  # type: ignore
def _(multiset, trace) -> float:
    return multiset[trace] / sum(multiset.values())


@singledispatch
def get_prefix_probability(multiset: Multiset, trace: Word) -> float:
    """Get the prefix probability of a trace against a multiset."""


@get_prefix_probability.register(NaiveMultiset)  # type: ignore
def _(multiset, trace) -> float:
    return multiset.get_prefix_probability(trace)


@get_prefix_probability.register(PrefixTreeMultiset)  # type: ignore
def _(multiset, trace) -> float:
    return multiset.get_prefix_probability(trace)


@get_prefix_probability.register(Counter)  # type: ignore
def _(multiset, trace) -> float:
    card1 = sum(multiset.values())
    d1 = 0.0
    for string in multiset.keys():
        for i in range(len(string) + 1):
            prefix, suffix = string[:i], string[i:]
            if prefix != trace:
                continue
            d1 += multiset[prefix + suffix]
    return d1 / card1


"""
for string in all_strings:
    string = tuple(string)
    for i in range(len(string)):
        prefix, suffix = string[:i], string[i:]
        d1 = get_prefix_probability(multiset1, prefix)
        d2 = get_prefix_probability(multiset1, prefix)
        norm = abs(d1 / card1 - d2 / card2)
        current_max = max([norm, current_max])
"""
