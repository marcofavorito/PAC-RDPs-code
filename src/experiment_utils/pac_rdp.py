"""Agent PAC-RDP."""
from functools import partial
from typing import Optional, cast

import gym

from src.learn_pdfa.base import Algorithm, learn_pdfa
from src.learn_pdfa.utils.generator import Generator, MultiprocessedGenerator
from src.learn_rdps import RDPGenerator, mdp_from_pdfa, random_exploration_policy
from src.pdfa import PDFA
from yarllib.base import AbstractAgent
from yarllib.helpers.history import History
from yarllib.planning.gpi import ValueIterationAgent


class RDPLearner(AbstractAgent):
    """RDP learner class."""

    def __init__(
        self,
        upperbound: int = 10,
        epsilon: float = 0.1,
        delta: float = 0.1,
        nb_samples: int = 10000,
        stop_probability: float = 0.2,
        gamma: float = 0.9,
        nb_sampling_processes: int = 1,
        **_kwargs
    ):
        """Initialize the PAC-RDP learner."""
        self.upperbound = upperbound
        self.nb_samples = nb_samples
        self.epsilon = epsilon
        self.delta = delta
        self.stop_probability = stop_probability
        self.gamma = gamma
        self.nb_sampling_processes = nb_sampling_processes

        self.rdp_generator: Optional[RDPGenerator] = None
        self.pdfa: Optional[PDFA] = None
        self.value_iteration_agent: Optional[ValueIterationAgent] = None

    def train(self, env: gym.Env, *args, max_nb_iterations: int = 50, **kwargs):
        """Train the agent."""
        policy = partial(random_exploration_policy, env)

        self.rdp_generator = RDPGenerator(
            env,
            policy=policy,
            nb_rewards=2,
            stop_probability=self.stop_probability,
        )
        if self.nb_sampling_processes == 1:
            generator: Generator = self.rdp_generator
        else:
            generator = MultiprocessedGenerator(
                self.rdp_generator, nb_processes=self.nb_sampling_processes
            )

        self.pdfa = learn_pdfa(
            algorithm=Algorithm.BALLE,
            nb_samples=self.nb_samples,
            sample_generator=generator,
            alphabet_size=self.rdp_generator.alphabet_size(),
            delta=self.delta,
            n=self.upperbound,
        )
        new_env = mdp_from_pdfa(
            self.pdfa, self.rdp_generator, stop_probability=self.stop_probability
        )
        self.value_iteration_agent = ValueIterationAgent(
            new_env.observation_space, new_env.action_space, gamma=self.gamma
        )
        self.value_iteration_agent.train(
            new_env, *args, max_nb_iterations=max_nb_iterations, **kwargs
        )
        return self.test(env, **kwargs)

    def test(self, env: gym.Env, *args, **kwargs) -> History:
        """Test the agent."""
        wrapper = RDPWrapper(
            env, cast(PDFA, self.pdfa), cast(RDPGenerator, self.rdp_generator)
        )
        return cast(ValueIterationAgent, self.value_iteration_agent).test(
            wrapper, **kwargs
        )

    def get_best_action(self, state):
        """Get best action."""
        return self.value_iteration_agent.get_best_action(state)


class RDPWrapper(gym.Wrapper):
    """Wrapper to gym environment."""

    def __init__(self, env: gym.Env, pdfa: PDFA, rdp_generator: RDPGenerator):
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
