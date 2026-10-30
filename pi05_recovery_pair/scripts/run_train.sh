#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PI05_CHECKPOINT:?Set original base checkpoint path}"
: "${PI05_NORM_STATS:?Set original norm_stats.json path}"
: "${PI05_PREPARED:?Set prepared dataset artifact path}"
: "${PI05_CONFIG:?Set calibrated config.json path}"
: "${PI05_OUTPUT:?Set new output path}"
: "${CUDA_VISIBLE_DEVICES:?Select free GPU IDs explicitly after checking nvidia-smi}"
TASK_NPROC="${NPROC_PER_NODE:-1}"
if [[ "$TASK_NPROC" == 1 ]]; then
  exec python -m recovery_pair train --checkpoint "$PI05_CHECKPOINT" \
    --norm-stats "$PI05_NORM_STATS" --prepared "$PI05_PREPARED" \
    --config "$PI05_CONFIG" --output "$PI05_OUTPUT" "$@"
else
  exec torchrun --standalone --nproc_per_node="$TASK_NPROC" -m recovery_pair train \
    --checkpoint "$PI05_CHECKPOINT" --norm-stats "$PI05_NORM_STATS" \
    --prepared "$PI05_PREPARED" --config "$PI05_CONFIG" --output "$PI05_OUTPUT" "$@"
fi
