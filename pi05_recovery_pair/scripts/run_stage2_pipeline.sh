#!/usr/bin/env bash
set -euo pipefail
S2_CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$S2_CODE"
export PYTHONPATH="$S2_CODE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} JAX_PLATFORMS=cpu
: "${S2_PYTHON:?Source scripts/your-stage2.local.env}"
: "${S2_RUN:?Set a NEW independent Stage2 output root}"
S2_ACTION=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi
case "$S2_ACTION" in
prepare)
  "$S2_PYTHON" -u -m recovery_pair.stage2_train prepare \
    --stage1-prepared "${S2_STAGE1_PREPARED:?}" --matching "${S2_MATCHING:?}" \
    --config "${S2_CONFIG:?}" --output "$S2_RUN/prepared" "$@"
  ;;
audit)
  "$S2_PYTHON" -m recovery_pair.stage2_train audit --prepared "$S2_RUN/prepared" "$@"
  ;;
train)
  : "${CUDA_VISIBLE_DEVICES:?Select GPUs explicitly after nvidia-smi}"
  S2_ARGS=(--prepared "$S2_RUN/prepared" --config "${S2_CONFIG:?}"
    --stage1-checkpoint "${S2_STAGE1_CHECKPOINT:?}" --base-checkpoint "${S2_BASE:?}"
    --norm-stats "${S2_NORM:?}" --output "$S2_RUN/training"
    --max-cuda-memory-gb "${S2_TRAIN_MEMORY_GIB:-20}"
    --save-every "${S2_SAVE_EVERY:-50}" --keep-last "${S2_KEEP_LAST:-3}")
  if [[ -n ${S2_RESUME:-} ]]; then S2_ARGS+=(--resume "$S2_RESUME"); fi
  if [[ -n ${S2_MAX_UPDATES:-} ]]; then S2_ARGS+=(--max-updates "$S2_MAX_UPDATES"); fi
  if [[ ${S2_NPROC:-1} == 1 && ${S2_NNODES:-1} == 1 ]]; then
    exec "$S2_PYTHON" -u -m recovery_pair.stage2_train train "${S2_ARGS[@]}" "$@"
  fi
  S2_DIST=(--nproc_per_node="${S2_NPROC:-1}")
  if [[ ${S2_NNODES:-1} == 1 ]]; then S2_DIST+=(--standalone); else
    S2_DIST+=(--nnodes="$S2_NNODES" --node_rank="${S2_NODE_RANK:?}" \
      --master_addr="${S2_MASTER_ADDR:?}" --master_port="${S2_MASTER_PORT:-29521}")
  fi
  exec "$S2_PYTHON" -u -m torch.distributed.run "${S2_DIST[@]}" \
    -m recovery_pair.stage2_train train "${S2_ARGS[@]}" "$@"
  ;;
export)
  : "${S2_EXPORT_CHECKPOINT:?Set the exact Stage2 update_*.pt to export}"
  exec "$S2_PYTHON" -u -m recovery_pair.stage2_train export \
    --checkpoint "$S2_EXPORT_CHECKPOINT" --output "$S2_RUN/exported" "$@"
  ;;
export-stage1)
  exec "$S2_PYTHON" -u -m recovery_pair.stage2_train export-stage1 \
    --checkpoint "${S2_STAGE1_CHECKPOINT:?}" --base-checkpoint "${S2_BASE:?}" \
    --norm-stats "${S2_NORM:?}" --output "${S2_STAGE1_EXPORTED:?}" "$@"
  ;;
eval)
  S2_LABEL=${1:-stage2}
  if [[ $# -gt 0 ]]; then shift; fi
  case "$S2_LABEL" in
    fewshot) S2_WEIGHTS=${S2_BASE:?} ;;
    stage1) S2_WEIGHTS=${S2_STAGE1_EXPORTED:?} ;;
    stage2) S2_WEIGHTS="$S2_RUN/exported" ;;
    *) echo 'Expected fewshot, stage1 or stage2' >&2; exit 2 ;;
  esac
  S2_EVAL_PACKAGE=${S2_EVAL_PACKAGE:-$(cd .. && pwd)}
  S2_EVAL_ARGS=(--label "$S2_LABEL" --checkpoint "$S2_WEIGHTS" --norm-stats "${S2_NORM:?}"
    --output "$S2_RUN/evaluation/$S2_LABEL" --gpus "${S2_EVAL_GPUS:?}")
  if [[ -n ${S2_TRAINING_INITIALS:-} && $S2_LABEL != fewshot ]]; then
    S2_EVAL_ARGS+=(--training-initials "$S2_TRAINING_INITIALS")
  fi
  exec "${S2_EVAL_PYTHON:?}" "$S2_EVAL_PACKAGE/stage2_evaluate_parallel.py" "${S2_EVAL_ARGS[@]}" "$@"
  ;;
compare)
  exec "$S2_PYTHON" "${S2_EVAL_PACKAGE:-$(cd .. && pwd)}/stage2_compare.py" \
    --fewshot "$S2_RUN/evaluation/fewshot" --stage1 "$S2_RUN/evaluation/stage1" \
    --stage2 "$S2_RUN/evaluation/stage2" --output "$S2_RUN/comparison" "$@"
  ;;
*) echo "Usage: source scripts/stage2.local.env; bash $0 prepare|audit|train|export|export-stage1|eval [fewshot|stage1|stage2]|compare" >&2; exit 2 ;;
esac
