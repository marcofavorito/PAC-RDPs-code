"""Agent PAC-RDP."""
import logging
import pprint
from typing import List, Optional, cast

import gym
from graphviz import Digraph
from yarllib.core import Model
from yarllib.planning.gpi import ValueIterationAgent

from src.learn_pdfa.base import Algorithm, learn_pdfa
from src.learn_rdps import (
    AbstractRDPGenerator,
    RDPGenerator,
    RDPGeneratorWrapper,
    mdp_from_pdfa,
)
from src.pdfa import PDFA
from src.pdfa.render import to_graphviz
from src.types import Word


class RDPLearner(Model):
    """RDP learner class."""

    def __init__(
        self,
        env: gym.Env,
        upperbound: int = 10,
        epsilon: float = 0.1,
        delta: float = 0.1,
        nb_samples: int = 10000,
        stop_probability: float = 0.2,
        gamma: float = 0.9,
        update_frequency: int = 100,
        **_kwargs,
    ):
        """Initialize the PAC-RDP learner."""
        self.env = env
        self.upperbound = upperbound
        self.nb_samples = nb_samples
        self.epsilon = epsilon
        self.delta = delta
        self.stop_probability = stop_probability
        self.gamma = gamma
        self.update_frequency = update_frequency

        self._rdp_generator: Optional[AbstractRDPGenerator] = None
        self.pdfa: Optional[PDFA] = None
        self.dataset: List[Word] = []
        self.value_iteration_agent: Optional[ValueIterationAgent] = None

    @property
    def rdp_generator(self) -> AbstractRDPGenerator:
        """Get the RDP generator wrapper."""
        assert self._rdp_generator is not None, "RDP generator not yet set."
        return self._rdp_generator

    def on_session_begin(self, *args, **kwargs) -> None:
        """On session begin."""
        self._rdp_generator = cast(RDPGeneratorWrapper, self.context.environment)

    def on_episode_end(self, episode, **kwargs) -> None:
        """On episode end."""
        if episode != 0 and episode % self.update_frequency == 0:
            logging.info(f"Updating policy at episode {episode}")
            self._update()
        self._add_trace()

    def get_best_action(self, state):
        """Get best action."""
        if self.value_iteration_agent is None:
            return self.context.environment.action_space.sample()
        return self.value_iteration_agent.get_best_action(state)

    def _learn_pdfa(self) -> PDFA:
        """Learn a PDFA from the environment."""
        pdfa = learn_pdfa(
            algorithm=Algorithm.BALLE,
            dataset=self.dataset,
            alphabet_size=self.rdp_generator.alphabet_size(),
            delta=self.delta,
            n=self.upperbound,
            with_infty_norm=False,
        )
        return pdfa

    def to_graphviz(self) -> Digraph:
        """Get the PDFA automaton."""
        if self.pdfa is None:
            raise ValueError("PDFA not learned yet.")
        return to_graphviz(
            self.pdfa,
            char2str=lambda c: str(self.rdp_generator.decoder(c))
            if c != -1
            else str(-1),
        )

    def _add_trace(self):
        env = cast(RDPGeneratorWrapper, self.context.environment)
        self.dataset.append(env.current_trace + [-1])

    def _update(self):
        try:
            self.pdfa = self._learn_pdfa()
        except ValueError:
            logging.info("PDFA Construction failed.")
            return

        logging.debug(f"PDFA learned. Number of states: {self.pdfa.nb_states}")
        new_env = mdp_from_pdfa(
            cast(PDFA, self.pdfa),
            cast(RDPGenerator, self.rdp_generator),
            stop_probability=self.stop_probability,
        )
        logging.info("Computed MDP.")
        logging.info(f"Observation space: {new_env.observation_space}")
        logging.info(f"Action space: {new_env.action_space}")
        logging.info(f"Dynamics:\n{pprint.pformat(new_env.P)}")
        self.value_iteration_agent = ValueIterationAgent(
            new_env.observation_space, new_env.action_space, gamma=self.gamma
        )
        self.value_iteration_agent.train(new_env, max_nb_iterations=50)
        logging.info("Value iteration completed.")


class RDPWrapper(gym.Wrapper):
    """Wrapper to gym environment."""

    def __init__(self, env: gym.Env, pdfa: PDFA, rdp_generator: AbstractRDPGenerator):
        """Initialize an RDP-wrapped environment."""
        super().__init__(env)
        self.pdfa = pdfa
        self.rdp_generator = rdp_generator
        self.current_state = self.pdfa.initial_state
        self.last_observation = None
        self.last_action = None
        self.last_reward = None

    def reset(self, **kwargs):
        """Reset the wrapped env."""
        initial_state = super().reset(**kwargs)
        self.last_observation = initial_state
        self.current_state = self.pdfa.initial_state
        return self.current_state

    def step(self, action):
        """Do a step in the wrapped env."""
        next_state, reward, done, info = super().step(action)
        character = self.rdp_generator.encoder((action, int(reward), next_state))
        self.current_state = self.pdfa.get_successor(self.current_state, character)
        return self.current_state, reward, done, info
