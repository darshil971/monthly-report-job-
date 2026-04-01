#!/bin/bash
set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate app

# If no arguments or first arg starts with -, run the default command with args
if [ $# -eq 0 ] || [ "${1:0:1}" = "-" ]; then
    exec python -m src "$@"
else
    # Otherwise, run whatever command was passed
    exec "$@"
fi
