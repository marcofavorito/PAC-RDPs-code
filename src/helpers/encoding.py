# -*- coding: utf-8 -*-

"""Encode/decode a multi-discrete space into a discrete one."""

from typing import Sequence

import numpy as np
from gym.spaces import Discrete, MultiDiscrete

from src.helpers.base import assert_


class Encoder:
    """Encode/Decode a multi-discrete space into a discrete one."""

    def __init__(self, input_space: MultiDiscrete):
        """
        Initialize the encoder.

        :param input_space: the space to encode.
        """
        self.input_space = input_space
        self.output_space = Discrete(np.prod(input_space.nvec))

    def encode(self, s: Sequence[int]) -> int:
        """
        Do the encoding.

        :param s: a point in the multi-discrete space.
        :return: the encoded version of the point.
        """
        if type(s) in {list, tuple}:
            s = np.array(s)
        assert_(
            self.input_space.contains(s), f"{s} is not contained in the input space."
        )
        dims = self.input_space.nvec
        result = 0
        for i, dim in zip(s, dims[+1:]):
            result += i
            result *= dim
        result += s[-1]
        assert_(
            self.output_space.contains(result),
            f"{result} is not contained in the output space.",
        )
        return result

    def decode(self, s: int) -> Sequence[int]:
        """
        Do the decoding.

        :param s: a point in a discrete space.
        :return: the decoded version of the point into a multi-discrete space.
        """
        assert_(
            self.output_space.contains(s), f"{s} is not contained in the output space."
        )
        dims = self.input_space.nvec
        result = []
        tmp = s
        for dim in reversed(dims[+1:]):
            result.append(tmp % dim)
            tmp = tmp // dim
        result.append(tmp)
        result = np.array(list(reversed(result)))
        assert_(
            self.input_space.contains(result),
            f"{result} is not contained in the input space.",
        )
        return result
