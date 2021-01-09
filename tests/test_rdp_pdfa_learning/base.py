"""Main test module."""
from copy import copy
from functools import partial
from typing import Dict, Sequence

from gym.wrappers import TimeLimit
from pdfa_learning.learn_pdfa.balle.core import learn_pdfa
from pdfa_learning.learn_pdfa.utils.generator import MultiprocessedGenerator
from pdfa_learning.pdfa.helpers import FINAL_SYMBOL
from pdfa_learning.pdfa.render import to_graphviz
from pdfa_learning.types import Word

from tests.conftest import get_nb_processes

from src.pac_rdp.agent import PacRdpAgent
from src.pac_rdp.helpers import RDPGenerator, random_exploration_policy

RDP_DEFAULT_CONFIG = dict(
    stop_probability=0.2,
    nb_samples=20000,
    delta=0.05,
    epsilon=0.05,
    n_upperbound=10,
    nb_processes=get_nb_processes(),
)


class BaseTestPdfaRdp:
    """Base test class for rotating MAB PDFA learning."""

    NB_PROCESSES = get_nb_processes()
    MAX_EPISODE_STEPS = 100
    CONFIG: Dict = RDP_DEFAULT_CONFIG
    OVERWRITE_CONFIG: Dict = {}

    @classmethod
    def setup_class(cls):
        """Set up the test."""
        config = copy(cls.CONFIG)
        config.update(cls.OVERWRITE_CONFIG)
        nb_samples = config["nb_samples"]
        stop_probability = config["stop_probability"]
        nb_processes = config["nb_processes"]
        delta = config["delta"]
        epsilon = config["epsilon"]
        n_upperbound = config["n_upperbound"]

        env = cls.make_env()
        env = TimeLimit(env, max_episode_steps=cls.MAX_EPISODE_STEPS)
        cls.agent = PacRdpAgent(env.observation_space, env.action_space, env)
        dataset = cls.sample(
            env,
            nb_samples=nb_samples,
            stop_probability=stop_probability,
            nb_processes=nb_processes,
        )
        pdfa = learn_pdfa(
            dataset=dataset,
            n=n_upperbound,
            alphabet_size=cls.agent._rdp_generator.alphabet_size(),
            delta=delta,
            epsilon=epsilon,
            with_infty_norm=False,
            with_smoothing=True,
        )
        cls.pdfa = pdfa
        decoder = cls.agent._rdp_generator.decoder
        to_graphviz(
            cls.pdfa, char2str=lambda c: decoder(c) if c != FINAL_SYMBOL else "-1"
        ).render(cls.__name__)

    @classmethod
    def make_env(cls):
        """Make environment."""
        raise NotImplementedError

    @classmethod
    def sample(
        cls, env, nb_samples: int, stop_probability: float, nb_processes: int
    ) -> Sequence[Word]:
        """Sample a dataset."""
        policy = partial(random_exploration_policy, env)
        _rdp_generator = RDPGenerator(
            env,
            policy=policy,
            nb_rewards=env.nb_rewards,
            stop_probability=stop_probability,
        )
        generator = MultiprocessedGenerator(_rdp_generator, nb_processes=nb_processes)
        return generator.sample(n=nb_samples)
