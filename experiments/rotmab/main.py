"""This script runs RotMAB experiment."""
import logging
import shutil
from pathlib import Path

from src.experiment_utils.mixins import (
    PACRDPExperiment,
    QLearningExperiment,
    RotMABExperiment,
    RotMABRDPWrapper,
    mixin_experiment,
)

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("yarllib").setLevel(logging.DEBUG)

if __name__ == "__main__":
    output = Path("outputs")
    shutil.rmtree(output, ignore_errors=True)
    nb_episodes = nb_samples = 10000

    common_configurations = dict(
        nb_runs=1,
        winning_probs=[0.9, 0.2],
        max_steps=30,
        gamma=0.9,
        epsilon=0.1,  # eps-greedy
        nb_episodes=nb_episodes,
        nb_processes=8,
        update_frequency=1000,
    )

    # Experiment1 = mixin_experiment(QLearningExperiment, RotMABExperiment)  # type: ignore
    # e1 = Experiment1(  # type: ignore
    #     **common_configurations,  # type: ignore
    #     output_dir=output,
    #     experiment_name="q-learning",
    # )
    # e1.run()

    Experiment2 = mixin_experiment(PACRDPExperiment, RotMABRDPWrapper)

    e2 = Experiment2(  # type: ignore
        **common_configurations,  # type: ignore
        output_dir=output,
        experiment_name="pac-rdp",
        stop_probability=0.1,
        nb_samples=nb_samples,
        delta=0.05,
        n_upperbound=10,
    )
    e2.run()
