"""Abstract RDP agent."""
from typing import Any, List, Optional, cast

from gym import Space
from pdfa_learning.pdfa import PDFA
from pdfa_learning.types import Character, Word

from src.core import Agent, random_policy
from src.helpers.gym import DiscreteEnv
from src.pac_rdp.helpers import AbstractRDPGenerator, RDPGenerator
from src.types import AgentObservation


class BasePacRdpAgent(Agent):
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
        if (
            self.current_state is not None
            and self.value_function is not None
            and state < len(self.value_function)
        ):
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
            symbol, [None]
        )[0]
