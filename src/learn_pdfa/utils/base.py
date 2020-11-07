"""Base module for miscellaneous utilities."""
from collections import Counter


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
