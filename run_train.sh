#!/usr/bin/env bash
set -euo pipefail
: "${PI05_ROOT:?Set the existing project root}"
: "${PI05_GPUS:?Set allocated physical GPUs, e.g. 1,3}"
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PI05_TRAIN_PYTHON:-$PI05_ROOT/environment-setup/.venv-pi05-sft/bin/python}"
export PYTHONPATH="$PACKAGE:$PI05_ROOT/RLinf${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset CUDA_VISIBLE_DEVICES RAY_ADDRESS
exec "$PYTHON" -Bu "$PACKAGE/train.py" --gpus "$PI05_GPUS" "$@"
