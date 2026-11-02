#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=${PI05_ROOT:-/home/wenchaoxu/wsy/vla-recovery-posttrain-loop}
TASK_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TASK_PY=${SCHEME1_PYTHON:-$TASK_ROOT/environment-setup/.venv-pi05-sft/bin/python}
TASK_PAIR=${SCHEME1_PAIR_CODE:-$TASK_ROOT/training_configs/pi05_ablation_direct_20261029/pi05_recovery_pair}
TASK_OUT="$TASK_ROOT/dataset_subsets/pi05_recovery_scheme1_stage1_20261102"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=${SCHEME1_VISION_GPU:-1}
mkdir -p "$TASK_OUT"
# Success export is independent of failure conversion/replay.
"$TASK_PY" -B "$TASK_CODE/build_scheme1.py" --root "$TASK_ROOT" --pair-code "$TASK_PAIR" --stage export --group success
while ! "$TASK_PY" -c 'import json,sys;from pathlib import Path;p=Path(sys.argv[1]);sys.exit(0 if p.exists() and json.loads(p.read_text()).get("state")=="complete" else 1)' \
    "$TASK_ROOT/trajectory_conversion_tests/pi05_scheme1_failure20_20261102/status.json"; do
  sleep 30
done
"$TASK_PY" -B "$TASK_CODE/build_scheme1.py" --root "$TASK_ROOT" --pair-code "$TASK_PAIR" --stage export --group failure
"$TASK_PY" -B "$TASK_CODE/build_scheme1.py" --root "$TASK_ROOT" --pair-code "$TASK_PAIR" --stage banks
"$TASK_PY" -B "$TASK_CODE/build_scheme1.py" --root "$TASK_ROOT" --pair-code "$TASK_PAIR" --stage match --device cuda:0 --memory-gib 6
"$TASK_PY" -B "$TASK_CODE/build_scheme1.py" --root "$TASK_ROOT" --pair-code "$TASK_PAIR" --stage prepare
"$TASK_PY" -B "$TASK_CODE/build_scheme1.py" --root "$TASK_ROOT" --pair-code "$TASK_PAIR" --stage align-training
