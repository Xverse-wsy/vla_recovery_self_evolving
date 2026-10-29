#!/usr/bin/env bash
# all: A training -> B training -> original/A/B full evaluation -> comparison.
set -euo pipefail
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${PI05_EXPERIMENT_CONFIG:-$PACKAGE/experiment.env}"
if [[ ! -f "$CONFIG" ]]; then
    echo "Copy $PACKAGE/experiment.env.example to $PACKAGE/experiment.env and edit paths/GPUs/batch first." >&2
    exit 2
fi
set -a
source "$CONFIG"
set +a
: "${PI05_ROOT:?Set PI05_ROOT in the config}"
: "${PI05_TRAIN_PYTHON:?Set the training Python in the config}"
: "${PI05_EVAL_PYTHON:?Set the evaluation Python in the config}"
: "${PI05_EXPERIMENT_OUTPUT:?Set the experiment output directory in the config}"
export PYTHONPATH="$PACKAGE:$PI05_ROOT/RLinf${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
if [[ "${1:-all}" != plan ]]; then
    mkdir -p "$PI05_EXPERIMENT_OUTPUT"
    exec > >(tee -a "$PI05_EXPERIMENT_OUTPUT/pipeline.log") 2>&1
fi
if [[ "$#" == 0 ]]; then
    set -- all
fi
exec "$PI05_TRAIN_PYTHON" -Bu "$PACKAGE/experiment.py" "$@"
