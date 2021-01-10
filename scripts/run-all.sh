#!/usr/bin/env bash

set -e

export PYTHONPATH="."
DEFAULT_OUTPUT_DIR="."
DEFAULT_ENVS="driving_agent cheatmab-02-001 malfunctionmab-02-80-20 rotmab-03-10-20-90 rotmab-02-90-20"
OUTPUT_DIR="${1:-${DEFAULT_OUTPUT_DIR}}"
ENVS="${2:-${DEFAULT_ENVS}}"

for env in ${ENVS}; do
  python3 scripts/run.py experiments/${env}/config.json --output-dir ${OUTPUT_DIR}/${env}
done

set +e
