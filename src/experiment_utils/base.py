"""Base utility for experiments."""

import json
import logging
import pickle
import shutil
from abc import ABC, abstractmethod
from copy import copy
from pathlib import Path
from typing import List, Optional, Union

import gym
from yarllib.base import AbstractAgent
from yarllib.callbacks import FrameCapture, RenderEnv
from yarllib.core import Agent, LearningEventListener, Policy
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
        self.experiment_dir: Path = self.output_dir / self.experiment_name
        self.seeds = seeds or list(range(0, nb_runs))

        self.update_frequency = _kwargs["update_frequency"]

        self.callbacks: List = []

    @abstractmethod
    def make_policy(self, env: gym.Env) -> Policy:
        """Make the training policy."""

    @abstractmethod
    def make_env(self) -> gym.Env:
        """Make a OpenAI Gym env.."""

    @abstractmethod
    def make_agent(self, env: gym.Env) -> Agent:
        """Make a RL agent."""

    def setup(self):
        """Set up the experiment."""
        checkpoint_callback = Checkpoint(
            self.update_frequency, self.experiment_dir, self.rendering
        )
        self.callbacks = [checkpoint_callback]
        if self.rendering:
            logging.info("Rendering enabled.")
            self.callbacks.append(RenderEnv())

    @setup_and_teardown
    def run(self):
        """Run the experiment."""
        logging.info(
            f"Start experiment with {self.nb_runs} runs, parallelized on {self.nb_processes} processes."
        )
        self.agents, self.histories = run_experiments(
            self.make_agent,
            self.make_env,
            self.make_policy,
            nb_runs=self.nb_runs,
            nb_episodes=self.nb_episodes,
            callbacks=self.callbacks,
            seeds=self.seeds,
            nb_workers=self.nb_processes,
            name_prefix=self.experiment_name,
        )
        return self.agents, self.histories

    def teardown(self):
        """Tear down the experiment."""
        logging.info("Runs completed.")
        logging.info("Plotting summaries...")
        plot_summaries(
            [self.histories],
            labels=[self.experiment_name],
            prefix=str(self.experiment_dir),
        )

        env = self.make_env()
        for ag, h in zip(self.agents, self.histories):
            run_dir = self.experiment_dir / h.name
            self._process_experiment(ag, h, run_dir)

            ag.test(
                env,
                nb_episodes=1,
                policy=GreedyPolicy(),
                callbacks=[FrameCapture(run_dir / "frames")] if self.rendering else [],
            )

        logging.info("Saving environment object...")
        with Path(self.experiment_dir / "env.obj").open("wb") as fp:
            pickle.dump(env, fp)

        print("Experiment done.")
        print(f"Find output in {self.experiment_dir}.")

    def _process_experiment(
        self, agent: AbstractAgent, history: History, experiment_dir: Path
    ):
        logging.info(f"Processing experiment {history.name} outputs...")
        experiment_dir.mkdir(exist_ok=True)
        agent_file = experiment_dir / "agent.obj"
        history_file = experiment_dir / "history.json"
        with agent_file.open("wb") as fpb:
            pickle.dump(agent, fpb)
        with history_file.open("w") as fp:
            json.dump(history_to_json(history), fp)


class Checkpoint(LearningEventListener):
    """Save a checkpoint."""

    def __init__(
        self,
        frequency: int,
        directory: Path,
        rendering: bool = False,
        nb_trials: int = 50,
    ):
        """
        Initialize the checkpoint callback.

        :param frequency: the number episodes to wait
          for every greedy run.
        """
        self.output_dir = directory
        self.experiment_dir = directory
        self.frequency = frequency
        self.rendering = rendering
        self.nb_trials = nb_trials

    def on_session_begin(self, *args, **kwargs) -> None:
        """On session begin."""
        self.experiment_dir = self.output_dir / self.context.experiment_name
        if self.experiment_dir.exists():
            logging.info(f"Removing directory {self.experiment_dir}...")
            shutil.rmtree(self.experiment_dir)
        logging.info(f"Creating directory {self.experiment_dir}...")
        self.experiment_dir.mkdir(parents=True, exist_ok=False)

        logging.info("Saving environment object...")
        with Path(self.experiment_dir / "env.obj").open("wb") as fp:
            pickle.dump(self.context.environment, fp)

    def on_episode_end(self, episode, **kwargs) -> None:
        """Handle on episode end."""
        if episode % self.frequency == 0:
            logging.info(f"Checkpoint at episode {episode}")
            self._run_test(episode)

    def _run_test(self, episode):
        """Run the test."""
        agent = copy(self.context.agent)
        history = rollout(agent, self.context.environment)
        run_dir = self.experiment_dir / history.name
        ep_string = f"{episode:010d}"
        agent_file = run_dir / f"agent-{ep_string}.obj"
        history_file = run_dir / f"history-{ep_string}.json"
        with agent_file.open("wb") as fpb:
            pickle.dump(agent, fpb)
        with history_file.open("w") as fp:
            json.dump(history_to_json(history), fp)

    def on_session_end(self, exception: Optional[Exception], *args, **kwargs) -> None:
        """On session end."""
        episode = self.context.current_episode
        logging.info(f"Training ended at episode {episode}.")
        self._run_test(episode)
        logging.info("Experiment done.")
        logging.info(f"Find output in {self.experiment_dir}.")


def rollout(
    agent: AbstractAgent,
    env: gym.Env,
    nb_episodes: int = 1,
):
    """
    Do a test.

    :param agent: an agent.
    :param env: the OpenAI Gym environment.
    :param nb_episodes: the number of test episodes.
    :return: None
    """
    histories = []
    for _ in range(nb_episodes):
        steps = []
        state = env.reset()
        done = False
        while not done:
            action = agent.get_best_action(state)
            next_state, reward, done, info = env.step(action)
            steps.append((state, action, reward, next_state))
            state = next_state
        histories.append(steps)
    return History(histories, is_training=False)
