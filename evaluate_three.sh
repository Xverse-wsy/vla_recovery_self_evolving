#!/usr/bin/env bash
set -euo pipefail
: "${PI05_ROOT:?Set the project root}"
: "${PI05_A_CHECKPOINT:?Set model A final checkpoint actor directory}"
: "${PI05_B_CHECKPOINT:?Set model B final checkpoint actor directory}"
: "${PI05_EVAL_OUTPUT:?Set a new evaluation output root}"
: "${PI05_TRAIN_PYTHON:?Set the training Python for result summarization}"
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_CHECKPOINT="$("$PI05_TRAIN_PYTHON" -c 'import sys; sys.path.insert(0,sys.argv[1]); from settings import load_paths; print(load_paths().base_checkpoint)' "$PACKAGE")"
for label in fewshot a60 b80; do
    case "$label" in
        fewshot) checkpoint="$BASE_CHECKPOINT" ;;
        a60) checkpoint="$PI05_A_CHECKPOINT" ;;
        b80) checkpoint="$PI05_B_CHECKPOINT" ;;
    esac
    ENTRYPOINT=run_eval.sh
    if [[ -n "${PI05_EVAL_GPUS:-}" ]]; then
        ENTRYPOINT=run_eval_parallel.sh
    fi
    bash "$PACKAGE/$ENTRYPOINT" --label "$label" --checkpoint "$checkpoint" \
        --output "$PI05_EVAL_OUTPUT/$label" "$@"
done
if ! "$PI05_TRAIN_PYTHON" -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); sys.exit(0 if all((p/s/"status.json").exists() and json.loads((p/s/"status.json").read_text())["state"]=="complete" for s in ["fewshot","a60","b80"]) else 1)' "$PI05_EVAL_OUTPUT"; then
    echo 'Local workers finished; global results are waiting for other nodes. Merge and compare after all nodes finish.'
    exit 0
fi
"$PI05_TRAIN_PYTHON" -B "$PACKAGE/compare_results.py" --fewshot "$PI05_EVAL_OUTPUT/fewshot" \
    --a60 "$PI05_EVAL_OUTPUT/a60" --b80 "$PI05_EVAL_OUTPUT/b80" \
    --output "$PI05_EVAL_OUTPUT/comparison"
