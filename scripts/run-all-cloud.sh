#!/usr/bin/env bash

# clone repository 
TOKEN="$(cat .github_access_token)"
git clone https://marcofavorito:${TOKEN}@github.com/whitemech/PAC-RDPs-code.git code
cd code
git checkout master

pipenv --python python3.8
pipenv install --dev --skip-lock

pipenv run scripts/run-all.sh "$@"
