"""This script runs RotMAB experiment."""
import logging
import shutil
from pathlib import Path
from typing import Dict

import gym
from gym.wrappers import TimeLimit

from src import NonMarkovianRotatingMAB
from src.algorithms.q_learning import QLearning
from src.callbacks.checkpoint import Checkpoint
from src.core import make_eps_greedy_policy
from src.experiment import Experiment
from src.pac_rdp.agent import PacRdpAgent
from src.pac_rdp.agent_simple import PacRdpAgentSimple
from src.pac_rdp.checkpoint import RDPCheckpoint

logging.basicConfig(level=logging.DEBUG)


def make_env(winning_probs, max_episode_steps) -> gym.Env:
    """Make environment."""
    return TimeLimit(
        NonMarkovianRotatingMAB(winning_probs=winning_probs),
        max_episode_steps=max_episode_steps,
    )


def run_pac_rdp(env: gym.Env, checkpoint_frequency: int = 500, **experiment_configs):
    """Run PAC-RDP experiments (v1)."""
    label = "pac-rdp"
    agent_params = dict(env=env)
    callbacks = [RDPCheckpoint(checkpoint_frequency, output / label)]
    experiment = Experiment(
        label, env, PacRdpAgent, agent_params, callbacks=callbacks, **experiment_configs
    )
    stats = experiment.run()
    return stats


def run_q_learning(
    env: gym.Env,
    epsilon: float = 0.1,
    checkpoint_frequency: int = 500,
    **experiment_configs
):
    """Run Q-Learning experiments."""
    label = "q-learning"
    agent_params: Dict = dict(policy=make_eps_greedy_policy(epsilon))
    callbacks = [Checkpoint(checkpoint_frequency, output / label)]
    experiment = Experiment(
        label, env, QLearning, agent_params, callbacks=callbacks, **experiment_configs
    )
    stats = experiment.run()
    return stats


def run_pac_rdp_simple(
    env: gym.Env,
    update_frequency: int = 500,
    checkpoint_frequency: int = 500,
    **experiment_configs
):
    """Run PAC-RDL experiments (v2)."""
    label = "pac-rdp-simple"
    agent_params = dict(env=env, update_frequency=update_frequency)
    callbacks = [RDPCheckpoint(checkpoint_frequency, output / label)]
    experiment = Experiment(
        label,
        env,
        PacRdpAgentSimple,
        agent_params,
        callbacks=callbacks,
        **experiment_configs
    )
    stats = experiment.run()
    return stats


if __name__ == "__main__":
    output = Path("outputs")
    shutil.rmtree(output, ignore_errors=True)

    # configure env
    env_config = dict(winning_probs=[0.9, 0.2], max_steps=100)
    env = make_env(**env_config)

    # configure experiments
    checkpoint_frequency = 500
    experiment_configs = dict(
        nb_episodes=8000,
        nb_runs=32,
        nb_processes=8,
        checkpoint_frequency=checkpoint_frequency,
        seeds=None,
    )

    # run experiments
    run_q_learning(env, epsilon=0.1)
    run_pac_rdp(env)
    run_pac_rdp_simple(env, update_frequency=500)
