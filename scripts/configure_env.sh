#!/usr/bin/env bash

set -e

function install_poetry_dependency(){
    cd "$1"
    poetry build
    pip install `ls dist/*.whl`
    cd ..
}

install_poetry_dependency yarllib


set +e
