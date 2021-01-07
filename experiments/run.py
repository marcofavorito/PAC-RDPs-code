#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Command-line tool to start the experiments."""

import json
from pathlib import Path

import click

from experiments.common.base import run_experiment_from_config


@click.command("run")
@click.argument("config", type=click.Path(exists=True, dir_okay=False, readable=True))
def run(config):
    """Run experiments."""
    config_path = Path(config)
    experiment_dir = config_path.parent / "outputs"
    config_json = json.load(config_path.open())
    run_experiment_from_config(experiment_dir, config_json)


if __name__ == "__main__":
    run()
