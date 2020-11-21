"""Base utility for experiments."""

import json
import logging
import pickle
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import gym

from yarllib.base import AbstractAgent
from yarllib.callbacks import FrameCapture, RenderEnv
from yarllib.core import Agent, Policy
from yarllib.experiment_utils import run_experiments
from yarllib.helpers.history import History, history_to_json
from yarllib.helpers.plots import plot_summaries
from yarllib.policies import GreedyPolicy

logging.basicConfig(
    format="[%(asctime)s][%(name)s][%(funcName)s][%(levelname)s]: %(message)s",
    level=logging.INFO,
)


def setup_and_teardown(func):
    """Set up and teardown decorator."""

    def _run(self):
        self.setup()
        result = func(self)
        self.teardown()
        return result

    return _run


class Experiment(ABC):
    """Base experiment."""

    def __init__(
        self,
        nb_runs: int = 25,
        nb_episodes: int = 500,
        nb_processes: int = 4,
        rendering: bool = False,
        output_dir: Union[str, Path] = "./output",
        experiment_name: str = "experiment",
        seeds: Optional[List[int]] = None,
        **_kwargs,
    ):
        """Initialize the experiment."""
        self.nb_runs = nb_runs
        self.nb_processes = nb_processes
        self.nb_episodes = nb_episodes
        self.rendering: bool = rendering
        self.output_dir: Path = Path(output_dir)
        self.experiment_name: str = experiment_name
        self.seeds = seeds

        self.callbacks: List = []

    @abstractmethod
    def make_policy(self) -> Policy:
        """Make the training policy."""

    @abstractmethod
    def make_env(self) -> gym.Env:
        """Make a OpenAI Gym env.."""

    @abstractmethod
    def make_agent(self, env: gym.Env) -> Agent:
        """Make a RL agent."""

    def setup(self):
        """Set up the experiment."""
        output_dir = Path(self.output_dir)
        if output_dir.exists():
            logging.info(f"Removing directory {output_dir}...")
            shutil.rmtree(output_dir)
        logging.info(f"Creating directory {output_dir}...")
        output_dir.mkdir(parents=True, exist_ok=False)

        callbacks = []
        if self.rendering:
            logging.info("Rendering enabled.")
            callbacks.append(RenderEnv())

    @setup_and_teardown
    def run(self):
        """Run the experiment."""
        logging.info(
            f"Start experiment with {self.nb_runs} runs, parallelized on {self.nb_processes} processes."
        )
        policy = self.make_policy()
        self.agents, self.histories = run_experiments(
            self.make_agent,
            self.make_env,
            policy,
            nb_runs=self.nb_runs,
            nb_episodes=self.nb_episodes,
            seeds=self.seeds,
            nb_workers=self.nb_processes,
        )
        return self.agents, self.histories

    def teardown(self):
        """Tear down the experiment."""
        logging.info("Runs completed.")
        logging.info("Plotting summaries...")
        plot_summaries(
            [self.histories], labels=[self.experiment_name], prefix=str(self.output_dir)
        )

        env = self.make_env()
        for ag, h in zip(self.agents, self.histories):
            experiment_dir = self.output_dir / h.name
            self._process_experiment(ag, h, experiment_dir)

            ag.test(
                env,
                nb_episodes=1,
                policy=GreedyPolicy(),
                callbacks=[FrameCapture(experiment_dir / "frames")]
                if self.rendering
                else [],
            )

        logging.info("Saving environment object...")
        with Path(self.output_dir / "env.obj").open("wb") as fp:
            pickle.dump(env, fp)

        print("Experiment done.")
        print(f"Find output in {self.output_dir}.")

    def _process_experiment(
        self, agent: AbstractAgent, history: History, experiment_dir: Path
    ):
        logging.info(f"Processing experiment {history.name} outputs...")
        experiment_dir.mkdir()
        agent_file = experiment_dir / "agent.obj"
        history_file = experiment_dir / "history.json"
        with agent_file.open("wb") as fpb:
            pickle.dump(agent, fpb)
        with history_file.open("w") as fp:
            json.dump(history_to_json(history), fp)
