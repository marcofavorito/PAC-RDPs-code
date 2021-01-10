#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Command-line tool to start the experiments."""

import json
from pathlib import Path

import click

from experiments.common.base import run_experiment_from_config


@click.command("run")
@click.argument("config", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option("--output-dir", type=click.Path(exists=False, file_okay=False, dir_okay=True, writable=True), default=None)
@click.option("--overwrite", type=click.Path(exists=True, dir_okay=False, readable=True), default=None)
def run(config, output_dir, overwrite):
    """Run experiments."""
    config_path = Path(config)

    experiment_dir = Path(output_dir or config_path.parent / "outputs")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    config_json = json.load(config_path.open())
    overwrite_json = json.load(Path(overwrite).open()) if overwrite else {}
    run_experiment_from_config(experiment_dir, config_json, overwrite_json)


if __name__ == "__main__":
    run()
