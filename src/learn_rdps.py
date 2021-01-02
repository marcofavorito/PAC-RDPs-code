"""Module that includes algorithms to learn RDPs."""
from collections import deque
from functools import partial
from typing import Callable, Deque, Dict, List, Sequence, Set, Tuple, cast

import gym
import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv

from src.learn_pdfa.utils.generator import Generator
from src.pdfa.base import FINAL_SYMBOL, PDFA
from src.types import Character, State, Word


class AbstractRDPGenerator:
    """Abstract RDP generator from gym environment."""

    def __init__(
        self,
        env: gym.Env,
        nb_rewards: int,
        stop_probability: float = 0.05,
    ):
        """Initialize the RDP generator."""
        self._env = env
        self._stop_probability = stop_probability

        self.obs_space_dim = self._env.observation_space.n
        self.action_dim = self._env.action_space.n
        self.nb_rewards = nb_rewards
        self.encoder = partial(
            np.ravel_multi_index,
            dims=(self.action_dim, self.nb_rewards, self.obs_space_dim),
        )
        self.decoder = partial(
            np.unravel_index,
            dims=(self.action_dim, self.nb_rewards, self.obs_space_dim),
        )

    def alphabet_size(self) -> int:
        """Get the alphabet size."""
        return int(np.prod([self.action_dim, self.nb_rewards, self.obs_space_dim]))

    def _should_stop(self) -> bool:
        """Return True if the current episode should stop, false otherwise."""
        return np.random.random() < self._stop_probability


class RDPGenerator(AbstractRDPGenerator, Generator):
    """Implementation of trace sampler with Generator interface.."""

    def __init__(
        self,
        env: gym.Env,
        nb_rewards: int,
        policy: Callable,
        stop_probability: float = 0.05,
    ):
        """Initialize the RDP generator."""
        super().__init__(env, nb_rewards, stop_probability)
        self._policy = policy

    def sample(self, n: int = 1) -> Sequence[Word]:
        """Sample a set of samples."""
        result = []
        for _ in range(n):
            word = self._sample_word()
            word = cast(List, word)
            result.append(word)
        return result

    def _sample_word(self) -> Word:
        """Sample one word."""
        _initial_state = self._env.reset()  # noqa: ignore
        done = False
        trace = []  # [(0, 0, initial_state)]
        while not done:
            if self._should_stop():
                break
            action = self._policy()
            obs, reward, done, _ = self._env.step(action)
            trace += [(action, int(reward), obs)]

        encoded_trace = [self.encoder(x) for x in trace] + [-1]
        return encoded_trace


class RDPGeneratorWrapper(AbstractRDPGenerator, gym.Wrapper):
    """Trace generator."""

    def __init__(
        self,
        env: gym.Env,
        nb_rewards: int,
        stop_probability: float = 0.05,
    ):
        """Initialize the RDP wrapper for a Gym Env."""
        AbstractRDPGenerator.__init__(self, env, nb_rewards, stop_probability)
        gym.Wrapper.__init__(self, env)

    def reset(self):
        """Reset state."""
        result = super().reset()
        self.current_trace: List = []
        return result

    def step(self, action):
        """Do a step."""
        o, r, done, info = super().step(action)
        next_symbol = self.encoder((action, int(r), o))
        self.current_trace.append(next_symbol)
        done = done or self._should_stop()
        return o, r, done, info


def random_exploration_policy(env: gym.Env) -> int:
    """
    Random exploration policy.

    It is declared as a module function so to be pickable.
    """
    return env.action_space.sample()


def mdp_from_pdfa(
    pdfa: PDFA, rdp_generator: RDPGenerator, stop_probability: float
) -> DiscreteEnv:
    """Infer the MDP from a PDFA."""
    P: Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]] = {}
    initial_state_distribution = [1.0] + [0.0] * (pdfa.nb_states - 1)
    nb_actions = rdp_generator.action_dim
    factor = nb_actions / (1 - stop_probability)

    for s in range(pdfa.nb_states):
        for a in range(nb_actions):
            P.setdefault(s, {})[a] = []

    queue: Deque = deque()
    queue.append(pdfa.initial_state)
    visited: Set[State] = {pdfa.initial_state}
    while len(queue) > 0:
        current = queue.pop()
        P.setdefault(current, {})

        # process transitions
        outgoing: Dict[Character, Tuple[State, float]] = pdfa.transition_dict[current]
        transitions_by_arqf: Dict[Tuple[int, float, State], float] = {}
        for character, (qf, probability) in outgoing.items():
            if character == FINAL_SYMBOL:
                continue
            a, r, s = rdp_generator.decoder(character)
            transitions_by_arqf.setdefault((a, r, qf), 0.0)
            transitions_by_arqf[(a, r, qf)] += probability

            if qf not in visited:
                visited.add(qf)
                queue.appendleft(qf)

        for (a, r, qf), p in transitions_by_arqf.items():
            P[current].setdefault(a, []).append((p * factor, qf, r, False))

    return DiscreteEnv(pdfa.nb_states, nb_actions, P, initial_state_distribution)
