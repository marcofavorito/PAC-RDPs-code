"""This script runs RotMAB experiment."""
import shutil
from pathlib import Path

from src.experiment_utils.mixins import (
    PACRDPExperiment,
    RotMABExperiment,
    mixin_experiment,
)

if __name__ == "__main__":
    output = Path("outputs")
    shutil.rmtree(output, ignore_errors=True)

    common_configurations = dict(
        nb_runs=4,
        winning_probs=[0.7, 0.3],
        max_steps=15,
        gamma=0.9,
        nb_episodes=1000,
        nb_processes=1,
    )

    # Experiment1 = mixin_experiment(QLearningExperiment, RotMABExperiment)
    # e1 = Experiment1(
    #     **common_configurations,
    #     output_dir=output / "q-learning",
    #     experiment_name="q-learning",
    # )
    # e1.run()

    Experiment2 = mixin_experiment(PACRDPExperiment, RotMABExperiment)

    e2 = Experiment2(
        **common_configurations,
        output_dir=output / "pac-rdp",
        experiment_name="pac-rdp",
        nb_samples=100000,
        upperbound=10,
        stop_probability=0.2,
        delta=0.1,
        nb_sampling_processes=6
    )
    e2.run()
