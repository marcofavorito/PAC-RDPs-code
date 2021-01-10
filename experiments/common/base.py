"""Common functions."""
import pprint
from pathlib import Path
from typing import Dict, Optional

import gym

from experiments.common.utils import locate

from src.algorithms.q_learning import QLearning
from src.callbacks.checkpoint import Checkpoint
from src.core import make_eps_greedy_policy
from src.experiment import Experiment
from src.pac_rdp.agent import PacRdpAgent
from src.pac_rdp.agent_simple import PacRdpAgentSimple
from src.pac_rdp.checkpoint import RDPCheckpoint


def run_q_learning(
    experiment_id: str,
    experiment_dir: Path,
    env: gym.Env,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    gamma: float = 0.99,
    checkpoint_frequency: int = 500,
    **experiment_configs
):
    """Run Q-Learning experiments."""
    agent_params: Dict = dict(
        policy=make_eps_greedy_policy(epsilon), alpha=alpha, gamma=gamma
    )
    callbacks = [Checkpoint(checkpoint_frequency, experiment_dir / experiment_id)]
    experiment = Experiment(
        experiment_id,
        env,
        QLearning,
        agent_params,
        callbacks=callbacks,
        **experiment_configs
    )
    stats = experiment.run()
    return stats


def run_pac_rdp(
    experiment_id: str,
    experiment_dir: Path,
    env: gym.Env,
    epsilon: float = 0.05,
    delta: float = 0.05,
    gamma: float = 0.99,
    max_l: int = 10,
    checkpoint_frequency: int = 500,
    **experiment_configs
):
    """Run PAC-RDP experiments (v1)."""
    agent_params = dict(env=env, epsilon=epsilon, delta=delta, gamma=gamma, max_l=max_l)
    callbacks = [RDPCheckpoint(checkpoint_frequency, experiment_dir / experiment_id)]
    experiment = Experiment(
        experiment_id,
        env,
        PacRdpAgent,
        agent_params,
        callbacks=callbacks,
        **experiment_configs
    )
    stats = experiment.run()
    return stats


def run_pac_rdp_simple(
    experiment_id: str,
    experiment_dir: Path,
    env: gym.Env,
    epsilon: float = 0.05,
    delta: float = 0.05,
    gamma: float = 0.99,
    max_depth: int = 10,
    update_frequency: int = 500,
    checkpoint_frequency: int = 500,
    **experiment_configs
):
    """Run PAC-RDL experiments (v2)."""
    agent_params = dict(
        env=env,
        epsilon=epsilon,
        delta=delta,
        gamma=gamma,
        max_depth=max_depth,
        update_frequency=update_frequency,
    )
    callbacks = [RDPCheckpoint(checkpoint_frequency, experiment_dir / experiment_id)]
    experiment = Experiment(
        experiment_id,
        env,
        PacRdpAgentSimple,
        agent_params,
        callbacks=callbacks,
        **experiment_configs
    )
    stats = experiment.run()
    return stats


def run_experiment_from_config(
    experiment_dir: Path,
    config: Dict,
    overwrite_experiment_config: Optional[Dict] = None,
):
    """Run experiments from configuration."""
    # configure experiments
    pprint.pprint("Configurations: ")
    pprint.pprint(config)
    pprint.pprint("Overwrite configurations: ")
    pprint.pprint(overwrite_experiment_config)
    experiment_configs = config["experiment"]
    overwrite_experiment_config = overwrite_experiment_config or {}
    experiment_configs.update(overwrite_experiment_config)

    # configure env
    env_config = config["environment"]
    env_entrypoint = locate(env_config["entrypoint"])
    env = env_entrypoint(**env_config["kwargs"])

    # run experiments
    algorithm_configs = config["algorithms"]
    for algorithm_config in algorithm_configs:
        experiment_id = algorithm_config["id"]
        entrypoint = locate(algorithm_config["entrypoint"])
        kwargs = algorithm_config["kwargs"]
        entrypoint(experiment_id, experiment_dir, env, **kwargs, **experiment_configs)
