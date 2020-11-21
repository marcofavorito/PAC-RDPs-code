"""Plot results."""
import json
import argparse
from pathlib import Path

from yarllib.helpers.history import history_from_json
from yarllib.helpers.plots import plot_summaries


def _is_dir(p: Path):
    return p.is_dir()


parser = argparse.ArgumentParser("plotter")
parser.add_argument("--datadir", default="outputs", help="Path to data directory.")

if __name__ == '__main__':
    arguments = parser.parse_args()
    output_dir = Path(arguments.datadir)
    assert output_dir.exists(), f"Path {output_dir} does not exists."

    histories = []
    names = []

    for experiment_dir in filter(_is_dir, output_dir.iterdir()):
        names.append(experiment_dir.name)
        experiment_histories = []
        for run_dir in filter(lambda p: p.is_dir() and p.name.startswith("experiment"), experiment_dir.iterdir()):
            history_json = json.load((run_dir / "history.json").open())
            experiment_histories.append(history_from_json(history_json))

        histories.append(experiment_histories)

    plot_summaries(histories, labels=names, prefix=str(output_dir))
