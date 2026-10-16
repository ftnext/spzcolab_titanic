#!/usr/bin/env bash
SCRIPT_DIR=$(cd "$(dirname $0)" && pwd)
mkdir -p "${SCRIPT_DIR}/data"

TMPDIR=$(mktemp -d)
cd "${TMPDIR}"

kaggle competitions download titanic -w
unzip titanic.zip -d data

mv data/{train,test}.csv "${SCRIPT_DIR}/data/"
