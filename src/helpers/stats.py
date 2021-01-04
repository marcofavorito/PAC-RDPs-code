# -*- coding: utf-8 -*-
"""Utils for collecting statistics."""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import gym
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame


@dataclass()
class Stats:
    """Record RL experiment statistics."""

    episode_lengths: List[int] = field(default_factory=list)
    episode_rewards: List[float] = field(default_factory=list)
    total_steps: int = 0
    timestamps: List[int] = field(default_factory=list)
    seed: Optional[int] = None


def stats_from_env(env: gym.Wrapper, prefix: str = "") -> Stats:
    """
    Get statistics from environment.

    It works only if some of the wrappers is StatsRecorder.
    """
    return Stats(
        episode_lengths=getattr(env, prefix + "episode_lengths"),
        episode_rewards=getattr(env, prefix + "episode_rewards"),
        total_steps=getattr(env, prefix + "total_steps"),
        timestamps=getattr(env, prefix + "timestamps"),
        seed=getattr(env, prefix + "random_seed"),
    )


def stats_to_json(stats: Stats) -> Dict:
    """Return a JSON object."""
    return dict(
        episode_rewards=stats.episode_rewards,
        episode_lengths=stats.episode_lengths,
        total_steps=stats.total_steps,
        timestamps=stats.timestamps,
        seed=stats.seed,
    )


def stats_from_json(obj: Dict) -> Stats:
    """Return a stats object from JSON."""
    return Stats(**obj)


def plot_average_stats(
    stats_list: Sequence[Sequence[Stats]],
    labels: Sequence[str],
    show: bool = True,
    prefix: str = "",
):
    """Plot average stats."""
    assert len(stats_list) == len(
        labels
    ), "Please provide the correct number of labels."
    attributes = ["episode_lengths", "episode_rewards"]
    figures = []
    for attribute in attributes:
        f = plt.figure()
        ax = f.add_subplot()
        ax.set_title(attribute)
        for label, history_list in zip(labels, stats_list):
            _plot_stats_attribute(history_list, attribute, label, ax=ax)
        figures.append(f)
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(prefix, attribute + ".svg"))

    return figures


def _plot_stats_attribute(stats_list: Sequence[Stats], attribute: str, label, ax=None):
    """Plot a certain attribute of a collection of histories."""
    data = np.asarray([getattr(h, attribute) for h in stats_list])
    df = DataFrame(data.T)

    df_mean = df.mean(axis=1)
    df_std = df.std(axis=1)
    sns_ax = sns.lineplot(df_mean.index, df_mean, label=label, ax=ax)
    sns_ax.fill_between(df_mean.index, df_mean - df_std, df_mean + df_std, alpha=0.3)
