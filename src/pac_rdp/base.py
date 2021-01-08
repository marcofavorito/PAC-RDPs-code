"""Abstract RDP agent."""
import logging
import pprint
from abc import ABC, abstractmethod
from typing import Any, List, Optional, cast

from gym import Space
from pdfa_learning.learn_pdfa.balle.core import learn_pdfa
from pdfa_learning.pdfa import PDFA
from pdfa_learning.types import Character, Word

from src.algorithms.value_iteration import value_iteration
from src.core import Agent, random_policy
from src.helpers.gym import DiscreteEnv
from src.pac_rdp.helpers import AbstractRDPGenerator, RDPGenerator, mdp_from_pdfa
from src.types import AgentObservation


class BasePacRdpAgent(Agent, ABC):
    """An abstract PAC-RDP agent."""

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        env: DiscreteEnv,
        epsilon: float = 0.1,
        delta: float = 0.1,
        gamma: float = 0.9,
    ):
        """Initialize."""
        super().__init__(observation_space, action_space, random_policy)
        self.env = env
        self.epsilon = epsilon
        self.delta = delta
        self.gamma = gamma
        self.nb_rewards = env.nb_rewards  # type: ignore

        self.dataset: List[Word] = []
        self.current_episode: List[Character] = []
        self.current_state: Optional[int] = 0
        self.value_function: Optional[List] = None
        self.current_policy: Optional[List] = None
        self.pdfa: Optional[PDFA] = None
        # TODO fix
        self._rdp_generator: AbstractRDPGenerator = RDPGenerator(env, self.nb_rewards, None)  # type: ignore

    @abstractmethod
    def get_upperbound(self) -> int:
        """Get current upperbound on the number of states."""

    @abstractmethod
    def get_stop_probability(self) -> float:
        """Get current stop probability."""

    def _encode_reward(self, reward: float) -> int:
        """Encode the reward."""
        return self.env.rewards.index(reward)  # type: ignore

    def _add_trace(self):
        """Add current trace to dataset."""
        new_trace = [
            self._rdp_generator.encoder((a, self._encode_reward(r), sp))
            for _, a, r, sp, _ in self.current_episode
        ]
        self.dataset.append(new_trace + [-1])

    def choose_best_action(self, state: Any):
        """Choose best action with the currently learned policy."""
        if self.current_state is not None and self.value_function is not None:
            self.current_policy = cast(List, self.current_policy)
            return self.current_policy[self.current_state]
        return self.action_space.sample()

    def do_pdfa_transition(self, agent_observation: AgentObservation):
        """Do a PDFA transition."""
        if self.pdfa is None:
            self.current_state = None
            return
        self.pdfa = cast(PDFA, self.pdfa)
        s, a, r, sp, done = agent_observation
        symbol = self._rdp_generator.encoder((a, self._encode_reward(r), sp))
        self.current_state = self.pdfa.transition_dict.get(self.current_state, {}).get(
            symbol, [None, None]
        )[0]

    def _learn_pdfa(self):
        """Learn the PDFA."""
        if len(self.dataset) == 0:
            logging.error("Dataset length is 0.")
            return
        try:
            pdfa = learn_pdfa(
                dataset=self.dataset,
                n=self.get_upperbound(),
                alphabet_size=self._rdp_generator.alphabet_size(),
                delta=self.delta ** 2,
                with_infty_norm=False,
                with_smoothing=True,
            )
        except Exception as e:
            logging.exception(e)
            return
        self.pdfa = pdfa

        new_env = mdp_from_pdfa(
            cast(PDFA, self.pdfa),
            cast(RDPGenerator, self._rdp_generator),
            stop_probability=self.get_stop_probability(),
        )
        logging.info("Computed MDP.")
        logging.info(f"Observation space: {new_env.observation_space}")
        logging.info(f"Action space: {new_env.action_space}")
        logging.info(f"Dynamics:\n{pprint.pformat(new_env.P)}")
        self.value_function, self.current_policy = value_iteration(
            new_env, max_iterations=50, discount=self.gamma
        )
        logging.info("Value iteration completed.")
