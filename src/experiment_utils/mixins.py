"""Mixins to configure experiments."""
from abc import ABC
from pathlib import Path
from typing import List, Type, cast

import gym
from gym.wrappers import TimeLimit
from yarllib.base import AbstractAgent
from yarllib.core import Agent, Policy
from yarllib.helpers.history import History
from yarllib.learning.tabular import TabularQLearning
from yarllib.policies import EpsGreedyPolicy, GreedyPolicy, RandomPolicy

from src import NonMarkovianRotatingMAB
from src.experiment_utils.base import Experiment
from src.experiment_utils.pac_rdp import RDPLearner
from src.learn_rdps import RDPGeneratorWrapper
from src.pdfa.base import FINAL_SYMBOL
from src.pdfa.render import to_graphviz


class RotMABExperiment(ABC):
    """Experiments using Rotating MAB."""

    def __init__(self, winning_probs: List[float], max_steps: int = 50, **_kwargs):
        """Initialize a RotMAB experiment."""
        self.winning_probs = winning_probs
        self.max_steps = max_steps

    def make_env(self) -> gym.Env:
        """Make env."""
        return TimeLimit(
            NonMarkovianRotatingMAB(winning_probs=self.winning_probs),
            max_episode_steps=self.max_steps,
        )


class RotMABRDPWrapper(RotMABExperiment, ABC):
    """Experiments for RDPs."""

    def __init__(self, stop_probability: float = 0.1, **_kwargs):
        """Initialize a RotMAB experiment."""
        super().__init__(**_kwargs)
        self.stop_probability = stop_probability

    def make_env(self) -> gym.Env:
        """Make env."""
        env = super().make_env()
        return RDPGeneratorWrapper(env, 2, self.stop_probability)


class QLearningExperiment(ABC):
    """Experiments using Q-Learning."""

    def __init__(self, **kwargs):
        """
        Initialize Q-Learning experiment.

        :param kwargs: arguments to Q-Learning agent.
        """
        self.kwargs = kwargs

    def make_agent(self, env: gym.Env) -> Agent:
        """Make an RL agent."""
        keys = ["alpha", "gamma"]
        kwargs = dict(
            [(key, self.kwargs.get(key)) for key in keys if self.kwargs.get(key)]
        )
        return TabularQLearning(
            env.observation_space, env.action_space, sparse=True, **kwargs
        ).agent()

    def make_policy(self, _env: gym.Env) -> Policy:
        """Make policy."""
        return EpsGreedyPolicy(self.kwargs.get("epsilon", 0.1))


class PACRDPExperiment(ABC):
    """Experiments using PAC-RDP."""

    def __init__(self, **kwargs):
        """
        Initialize PAC-RDP experiment.

        :param kwargs: arguments to Q-Learning agent.
        """
        self.kwargs = kwargs

    def make_agent(self, env: gym.Env) -> AbstractAgent:
        """Make an RL agent."""
        keys = [
            "gamma",
            "upperbound",
            "nb_samples",
            "epsilon",
            "delta",
            "stop_probability",
            "update_frequency",
        ]
        kwargs = dict(
            [(key, self.kwargs.get(key)) for key in keys if self.kwargs.get(key)]
        )
        return RDPLearner(env, **kwargs).agent()

    def make_policy(self, env: gym.Env) -> Policy:
        """Make policy."""
        return RandomPolicy(env.action_space)

    def _process_experiment(
        self, agent: AbstractAgent, history: History, experiment_dir: Path
    ):
        super()._process_experiment(agent, history, experiment_dir)  # type: ignore
        agent = cast(Agent, agent)
        model = cast(RDPLearner, agent.model)
        char2str = (
            lambda c: str(model.rdp_generator.decoder(c)) if c != FINAL_SYMBOL else "-1"
        )
        to_graphviz(model.pdfa, char2str=char2str).render(str(experiment_dir / "pdfa"))


def mixin_experiment(*_cls) -> Type[Experiment]:
    """Build the mixin experiment."""

    class _Experiment(*_cls, Experiment):  # type: ignore
        def __init__(self, **kwargs):
            for cls in _cls:
                cls.__init__(self, **kwargs)
            Experiment.__init__(self, **kwargs)

    return _Experiment
