#!/usr/bin/env bash
set -euo pipefail
: "${PI05_ROOT:?Set the project root}"
: "${PI05_A_CHECKPOINT:?Set model A final checkpoint actor directory}"
: "${PI05_B_CHECKPOINT:?Set model B final checkpoint actor directory}"
: "${PI05_EVAL_OUTPUT:?Set a new evaluation output root}"
: "${PI05_TRAIN_PYTHON:?Set the training Python for result summarization}"
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for label in fewshot a60 b80; do
    case "$label" in
        fewshot) checkpoint="$PI05_ROOT/models/RLinf-Pi05-LIBERO-SFT" ;;
        a60) checkpoint="$PI05_A_CHECKPOINT" ;;
        b80) checkpoint="$PI05_B_CHECKPOINT" ;;
    esac
    bash "$PACKAGE/run_eval.sh" --label "$label" --checkpoint "$checkpoint" \
        --output "$PI05_EVAL_OUTPUT/$label" "$@"
done
"$PI05_TRAIN_PYTHON" -B "$PACKAGE/compare_results.py" --fewshot "$PI05_EVAL_OUTPUT/fewshot" \
    --a60 "$PI05_EVAL_OUTPUT/a60" --b80 "$PI05_EVAL_OUTPUT/b80" \
    --output "$PI05_EVAL_OUTPUT/comparison"
