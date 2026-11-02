#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=${PI05_ROOT:-/home/wenchaoxu/wsy/vla-recovery-posttrain-loop}
TASK_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TASK_GPU=${CONVERSION_GPU:-3}
export CUDA_VISIBLE_DEVICES="$TASK_GPU"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID="$TASK_GPU"
export LIBERO_CONFIG_PATH="$TASK_ROOT/libero-resources/config"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${CONVERSION_PYTHON:-/home/wenchaoxu/miniconda3/envs/corrector-eval/bin/python}" -B \
  "$TASK_CODE/convert_selected_success.py" \
  --root "$TASK_ROOT" \
  --raw "${CONVERSION_RAW:-$TASK_ROOT/dataset_subsets/pi05_recovery_raw_scheme1_20261102_v1}" \
  --output "${CONVERSION_OUTPUT:-$TASK_ROOT/trajectory_conversion_tests/pi05_scheme1_success20_20261102}" \
  --workers "${CONVERSION_WORKERS:-2}" "$@"
