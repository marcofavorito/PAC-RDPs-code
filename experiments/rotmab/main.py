"""This script runs RotMAB experiment."""
import shutil
from pathlib import Path

from src.experiment_utils.mixins import (
    PACRDPExperiment,
    QLearningExperiment,
    RotMABExperiment,
    mixin_experiment,
)

if __name__ == "__main__":
    output = Path("outputs")
    shutil.rmtree(output, ignore_errors=True)

    if __name__ == "__main__":
        output = Path("outputs")
        shutil.rmtree(output, ignore_errors=True)

        common_configurations = dict(
            nb_runs=48,
            winning_probs=[0.9, 0.2],
            max_steps=15,
            gamma=0.9,
            epsilon=0.1,  # eps-greedy
            nb_episodes=1000,
            nb_processes=8,
        )

        Experiment1 = mixin_experiment(QLearningExperiment, RotMABExperiment)  # type: ignore
        e1 = Experiment1(  # type: ignore
            **common_configurations,  # type: ignore
            output_dir=output,
            experiment_name="q-learning",
        )
        e1.run()

        Experiment2 = mixin_experiment(PACRDPExperiment, RotMABExperiment)

        e2 = Experiment2(  # type: ignore
            **common_configurations,  # type: ignore
            output_dir=output,
            experiment_name="pac-rdp",
            stop_probability=0.2,
            nb_samples=300000,
            delta=0.05,
            n_upperbound=5,
            nb_sampling_processes=1,
        )
        e2.run()
