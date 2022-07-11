#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
OTHER=$3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/train.py $CONFIG --gpus $GPUS $OTHER
