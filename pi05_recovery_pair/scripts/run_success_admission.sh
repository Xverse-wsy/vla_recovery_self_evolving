#!/usr/bin/env bash
# New immutable output directory. No training, replay, API, or negative pairing.
set -euo pipefail
if [[ $# -lt 4 || $# -gt 7 ]]; then
  echo "Usage: bash $0 WINDOWS CHECKPOINT CONFIG OUTPUT [DEVICE=cuda:0] [MEMORY_GIB=8] [IMAGE_BATCH=2]" >&2
  exit 2
fi
PAIR_WINDOWS=$1
PAIR_CHECKPOINT=$2
PAIR_CONFIG=$3
PAIR_OUTPUT=$4
PAIR_DEVICE=${5:-cuda:0}
PAIR_MEMORY=${6:-8}
PAIR_IMAGE_BATCH=${7:-2}
PAIR_PYTHON=${PAIR_PYTHON:-python}
PAIR_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$PAIR_CODE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
mkdir -p -- "$(dirname -- "$PAIR_OUTPUT")"
mkdir -- "$PAIR_OUTPUT"
trap 'printf "{\"state\":\"failed\"}\n" > "$PAIR_OUTPUT/run_status.json"' ERR
cp -- "$PAIR_CONFIG" "$PAIR_OUTPUT/config.json"
printf '{"state":"numeric_candidates"}\n' > "$PAIR_OUTPUT/run_status.json"
"$PAIR_PYTHON" -u -m recovery_pair numeric-preview \
  --windows "$PAIR_WINDOWS" --config "$PAIR_OUTPUT/config.json" \
  --output "$PAIR_OUTPUT/numeric"
printf '{"state":"frozen_vision_features"}\n' > "$PAIR_OUTPUT/run_status.json"
"$PAIR_PYTHON" -u -m recovery_pair.vision_trial \
  --windows "$PAIR_WINDOWS" --numeric-preview "$PAIR_OUTPUT/numeric" \
  --checkpoint "$PAIR_CHECKPOINT" --output "$PAIR_OUTPUT/vision" \
  --device "$PAIR_DEVICE" --max-cuda-memory-gb "$PAIR_MEMORY" \
  --batch-size "$PAIR_IMAGE_BATCH" --review-count 0
printf '{"state":"success_admission"}\n' > "$PAIR_OUTPUT/run_status.json"
"$PAIR_PYTHON" -u -m recovery_pair.success_admission \
  --windows "$PAIR_WINDOWS" --feature-trial "$PAIR_OUTPUT/vision" \
  --config "$PAIR_OUTPUT/config.json" --output "$PAIR_OUTPUT/admission"
printf '{"state":"complete"}\n' > "$PAIR_OUTPUT/run_status.json"
