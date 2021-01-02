#!/usr/bin/env bash

set -e

pipenv run pip install ./yarllib
pipenv run pip install ./pdfa_learning

set +e
