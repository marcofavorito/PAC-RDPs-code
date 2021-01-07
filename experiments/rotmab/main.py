"""This script runs RotMAB experiment."""
import logging
import shutil
from pathlib import Path
from typing import Dict

from gym.wrappers import TimeLimit

from src import NonMarkovianRotatingMAB
from src.algorithms.q_learning import QLearning
from src.callbacks.checkpoint import Checkpoint
from src.core import make_eps_greedy_policy
from src.experiment import Experiment
from src.helpers.stats import plot_average_stats
from src.pac_rdp.agent import PacRdpAgent
from src.pac_rdp.checkpoint import RDPCheckpoint

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("yarllib").setLevel(logging.DEBUG)

if __name__ == "__main__":
    output = Path("outputs")
    shutil.rmtree(output, ignore_errors=True)

    # parameters
    nb_episodes = nb_samples = 8000
    nb_runs = 32
    winning_probs = [0.9, 0.2]
    max_steps = 30
    epsilon = 0.1
    nb_processes = 8
    update_frequency = 500
    seeds = None
    env = TimeLimit(
        NonMarkovianRotatingMAB(winning_probs=winning_probs),
        max_episode_steps=max_steps,
    )

    label = "q-learning"
    callbacks = [Checkpoint(update_frequency, output / label)]
    agent_params: Dict = dict(policy=make_eps_greedy_policy(epsilon))
    experiment = Experiment(
        label,
        env,
        QLearning,
        agent_params,
        nb_episodes=nb_episodes,
        callbacks=callbacks,
        nb_runs=nb_runs,
        nb_processes=nb_processes,
        seeds=seeds or (),
    )
    stats = experiment.run()
    plot_average_stats([stats], [label])

    label = "pac-rdp"
    agent_params = dict(env=env)
    callbacks = [RDPCheckpoint(update_frequency, output / label)]
    experiment = Experiment(
        label,
        env,
        PacRdpAgent,
        agent_params,
        nb_episodes=nb_episodes,
        callbacks=callbacks,
        nb_runs=nb_runs,
        nb_processes=nb_processes,
        seeds=seeds or (),
    )
    stats = experiment.run()
    plot_average_stats([stats], [label])
