# -*- coding: utf-8 -*-
"""Experiment helpers."""

import multiprocessing
from typing import Collection, Dict, List, Sequence, Type

import gym

from src.callbacks.base import Callback
from src.callbacks.stats import StatsCallback
from src.core import Agent
from src.helpers.misc import set_seed
from src.helpers.stats import Stats


class Experiment:
    """Class to coordinate a RL experiment."""

    def __init__(
        self,
        name: str,
        env: gym.Env,
        agent_cls: Type[Agent],
        agent_params: Dict,
        nb_episodes: int = 1000,
        nb_runs: int = 50,
        nb_processes: int = 8,
        callbacks: Collection[Callback] = (),
        seeds: Sequence[int] = (),
    ):
        """
        Set up the experiment.

        :param env: the environment.
        :param nb_runs: the number of runs to run in parallel.
        :param nb_processes: the number of processes available.
        :param seeds: the random seeds to set up a run.
        """
        self.name = name
        self.env = env
        self.agent_cls = agent_cls
        self.agent_params = agent_params
        self.nb_episodes = nb_episodes
        self.nb_runs = nb_runs
        self.nb_processes = nb_processes
        self.callbacks = callbacks
        self.seeds = (
            list(seeds) if seeds and len(seeds) == nb_runs else list(range(nb_runs))
        )

    def experiment_id(self, id_: int) -> str:
        """Get the experiment id."""
        nb_digits = len(str(self.nb_runs))
        return f"{self.name}-{id_:0{nb_digits}d}"

    @staticmethod
    def _do_job(
        experiment_id: str,
        env: gym.Env,
        agent_cls: Type[Agent],
        seed: int,
        agent_params: Dict,
        nb_episodes: int = 10000,
        callbacks: Collection[Callback] = (),
    ):
        """Do a single run."""
        set_seed(env, seed)
        agent = agent_cls(env.observation_space, env.action_space, **agent_params)
        history_callback = StatsCallback()
        callbacks = list(callbacks) + [history_callback]
        agent.train(
            env,
            experiment_name=experiment_id,
            nb_episodes=nb_episodes,
            callbacks=callbacks,
        )
        stats = history_callback
        stats.seed = seed
        return agent, stats

    def run(self) -> List[Stats]:
        """
        Run the experiments.

        :return: a list of statistics, one for each run.
        """
        pool = multiprocessing.Pool(processes=self.nb_processes)

        experiment_names = list(map(self.experiment_id, range(self.nb_runs)))
        results = [
            pool.apply_async(
                self._do_job,
                args=(
                    experiment_id,
                    self.env,
                    self.agent_cls,
                    seed,
                    self.agent_params,
                    self.nb_episodes,
                    self.callbacks,
                ),
            )
            for experiment_id, seed in zip(experiment_names, self.seeds)
        ]

        try:
            for p in results:
                p.wait()
        except KeyboardInterrupt:
            pass

        stats = []
        for p in filter(lambda x: x.ready(), results):
            _, stat = p.get()
            stats.append(stat)
        return stats
