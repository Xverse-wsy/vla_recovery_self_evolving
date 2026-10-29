#!/usr/bin/env bash
set -euo pipefail
: "${PI05_ROOT:?Set the existing project root}"
: "${PI05_EVAL_PYTHON:?Set a Python with RLinf/OpenPI and standard LIBERO}"
: "${PI05_EVAL_GPU:?Set an allocated physical GPU}"
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="$PI05_EVAL_GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$PACKAGE:$PI05_ROOT/RLinf:$PI05_ROOT/openpi/src:$PI05_ROOT/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LIBERO_CONFIG_PATH="$PI05_ROOT/libero-resources/config"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_TYPE=standard ROBOT_PLATFORM=LIBERO
exec "$PI05_EVAL_PYTHON" -Bu "$PACKAGE/evaluate.py" "$@"
