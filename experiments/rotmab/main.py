"""This script runs RotMAB experiment."""
import logging
import shutil
from pathlib import Path
from typing import Dict

import gym
from gym.wrappers import TimeLimit

from src.algorithms.base import make_eps_greedy_policy
from src.algorithms.q_learning import QLearning
from src.callbacks.checkpoint import Checkpoint
from src.experiment import Experiment
from src.helpers.stats import plot_average_stats

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("yarllib").setLevel(logging.DEBUG)

if __name__ == "__main__":
    output = Path("outputs")
    shutil.rmtree(output, ignore_errors=True)

    # parameters
    nb_episodes = nb_samples = 1000
    nb_runs = 32
    winning_probs = [0.9, 0.2]
    max_steps = 30
    epsilon = 0.8
    nb_processes = 8
    update_frequency = 25
    seeds = None
    """
    env = TimeLimit(
        NonMarkovianRotatingMAB(winning_probs=winning_probs),
        max_episode_steps=max_steps,
    )
    """
    env = gym.make("FrozenLake-v0", is_slippery=False)
    env = TimeLimit(env, max_episode_steps=max_steps)

    agent_params: Dict = dict()
    callbacks = [Checkpoint(update_frequency, output / "q-learning")]

    experiment = Experiment(
        "q-learning",
        env,
        QLearning,
        agent_params,
        policy=make_eps_greedy_policy(epsilon=epsilon),
        nb_episodes=nb_episodes,
        callbacks=callbacks,
        nb_runs=nb_runs,
        nb_processes=nb_processes,
        seeds=seeds or (),
    )

    stats = experiment.run()

    plot_average_stats([stats], ["q-learning"])
