#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot results."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import json
from pathlib import Path
from typing import List
import numpy as np

from src.helpers.stats import stats_from_json


def _is_dir(p: Path):
    return p.is_dir()


def _dir_filter(p):
    return p.is_dir() and p.name.startswith(p.name)


def _find_frequency(output_dir: Path):
    """Find checkpoint frequency."""
    # get first directory
    experiment_dir = list(output_dir.iterdir())[0]
    run_dir = list(experiment_dir.iterdir())[0]
    checkpoint_history_paths = list(run_dir.glob("history-*.json"))
    assert len(checkpoint_history_paths) >= 3
    p1, p2 = checkpoint_history_paths[0], checkpoint_history_paths[1]
    step = int(p1.stem[7:]) - int(p2.stem[7:])
    return step


parser = argparse.ArgumentParser("plotter")
parser.add_argument("--datadir", default="outputs", help="Path to data directory.")

if __name__ == '__main__':
    arguments = parser.parse_args()
    output_dir = Path(arguments.datadir)
    assert output_dir.exists(), f"Path {output_dir} does not exists."

    histories = []
    steps = []
    names = []

    experiment_directories = list(filter(_is_dir, output_dir.iterdir()))
    nb_experiments = len(experiment_directories)
    for experiment_dir in experiment_directories:
        names.append(experiment_dir.name)
        experiment_history = []

        run_directories = list(filter(_dir_filter, experiment_dir.iterdir()))
        for run_dir in sorted(filter(_dir_filter, experiment_dir.iterdir())):
            episode_histories = []
            steps = []
            for checkpoint_history_path in sorted(run_dir.glob("history-*.json")):
                checkpoint_history_fp = json.load(checkpoint_history_path.open())
                checkpoint_history = stats_from_json(checkpoint_history_fp)
                step = int(checkpoint_history_path.stem[8:])
                steps.append(step)
                episode_histories.append(checkpoint_history.episode_rewards)
            experiment_history.append((steps, episode_histories))
        histories.append(experiment_history)


    datas = []
    for experiment in histories:
        experiment_data = []
        for run in experiment:
            _, episodes = run
            rewards = np.asarray(episodes, dtype=np.float64)
            # run_data = np.concatenate([steps, rewards], axis=1)
            run_data = rewards
            experiment_data.append(run_data)
        datas.append(np.stack(experiment_data))

    for experiment_id, history in enumerate(histories):
        steps = max(histories[experiment_id], key=lambda run: len(run[0]))[0]
        experiment_data = datas[experiment_id]
        label = names[experiment_id]
        average_rewards = experiment_data.mean(axis=2)
        average_rewards_mean = average_rewards.mean(axis=0)
        average_rewards_std = average_rewards.std(axis=0)
        sns_ax = sns.lineplot(steps, average_rewards_mean, label=label)
        sns_ax.fill_between(steps, average_rewards_mean - average_rewards_std, average_rewards_mean + average_rewards_std, alpha=0.3)
    plt.savefig(Path(output_dir) / "plot.svg")
    plt.show()
