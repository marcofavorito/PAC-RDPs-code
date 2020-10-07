# -*- coding: utf-8 -*-

"""
This package contains an implementation of the Sequential Multi-Armed Bandit environment.

It implements a behaviour like: "give a reward for using
arm 1 four consecutive steps, followed by arm 3."
"""
from .sequential_mab import NonMarkovianSequentialMAB, SequentialMAB  # noqa: ignore
