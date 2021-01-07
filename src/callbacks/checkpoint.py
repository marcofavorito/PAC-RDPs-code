"""This module contains the implementation of the checkpoint callback."""
import json
import logging
import pickle
import shutil
from copy import copy
from pathlib import Path
from typing import cast

from src.callbacks.base import Callback
from src.helpers.stats import stats_from_env, stats_to_json
from src.wrappers.utils import StatsRecorder


class Checkpoint(Callback):
    """Save a checkpoint."""

    def __init__(
        self,
        frequency: int,
        directory: Path,
        rendering: bool = False,
        nb_trials: int = 50,
        save_env: bool = True,
        save_history: bool = True,
        save_agent: bool = False,
    ):
        """
        Initialize the checkpoint callback.

        :param frequency: the number episodes to wait
          for every greedy run.
        """
        super().__init__()
        self.output_dir = directory
        self.experiment_dir = directory
        self.frequency = frequency
        self.rendering = rendering
        self.nb_trials = nb_trials
        self.episode = 0

        self.save_env = save_env
        self.save_history = save_history
        self.save_agent = save_agent

    def on_training_begin(self, **kwargs) -> None:
        """On session begin."""
        self.experiment_dir = self.output_dir / cast(str, self.experiment_name)
        if self.experiment_dir.exists():
            logging.info(f"Removing directory {self.experiment_dir}...")
            shutil.rmtree(self.experiment_dir)
        logging.info(f"Creating directory {self.experiment_dir}...")
        self.experiment_dir.mkdir(parents=True, exist_ok=False)

        if self.save_env:
            logging.info("Saving environment object...")
            with Path(self.experiment_dir / "env.obj").open("wb") as fp:
                pickle.dump(self.env, fp)

    def on_episode_end(self, episode, **kwargs) -> None:
        """Handle on episode end."""
        self.episode = episode
        if episode % self.frequency == 0:
            logging.info(f"Checkpoint at episode {episode}")
            self._run_test(episode)

    def _run_test(self, episode):
        """Run the test."""
        agent = copy(self.agent)
        env = StatsRecorder(self.env, "test_")
        agent.test(env, nb_episodes=self.nb_trials)
        history = stats_from_env(env, prefix="test_")
        run_dir = self.experiment_dir
        run_dir.mkdir(exist_ok=True)
        ep_string = f"{episode:010d}"
        if self.save_history:
            history_file = run_dir / f"history-{ep_string}.json"
            with history_file.open("w") as fp:
                json.dump(stats_to_json(history), fp)
        if self.save_agent:
            agent_file = run_dir / f"agent-{ep_string}.obj"
            with agent_file.open("wb") as fpb:
                pickle.dump(agent, fpb)

    def on_training_end(self, **kwargs) -> None:
        """On session end."""
        logging.info("Training ended.")
        self._run_test(self.episode)
        logging.info("Experiment done.")
        logging.info(f"Find output in {self.experiment_dir}.")
