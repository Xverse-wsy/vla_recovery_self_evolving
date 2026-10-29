#!/usr/bin/env bash
set -euo pipefail
: "${PI05_ROOT:?Set the existing project root}"
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PI05_TRAIN_PYTHON:-$PI05_ROOT/environment-setup/.venv-pi05-sft/bin/python}"
export PI05_EVAL_PYTHON="${PI05_EVAL_PYTHON:-/home/wenchaoxu/miniconda3/envs/corrector-eval/bin/python}"
export PI05_EVAL_GPU="${PI05_EVAL_GPU:-1}"
RUN="$(cat "$PACKAGE/latest_validation.txt")"
exec > >(tee -a "$RUN/validation_eval.log") 2>&1
for label in fewshot a60 b80; do
    if [[ "$label" == fewshot ]]; then
        checkpoint="$PI05_ROOT/models/RLinf-Pi05-LIBERO-SFT"
    else
        checkpoint="$("$PYTHON" -c 'import json,sys; s=json.load(open(sys.argv[1])); assert s["state"]=="complete"; print(s["checkpoint"]+"/actor")' "$RUN/$label/status.json")"
    fi
    resume_args=()
    if [[ -f "$RUN/eval_$label/run.json" ]]; then
        resume_args+=(--resume)
    fi
    bash "$PACKAGE/run_eval.sh" --label "$label" --checkpoint "$checkpoint" \
        --output "$RUN/eval_$label" --trials 1 --tasks 0 "${resume_args[@]}"
done
"$PYTHON" -Bu "$PACKAGE/verify_validation.py" --run "$RUN"
"$PYTHON" -Bu "$PACKAGE/compare_results.py" --fewshot "$RUN/eval_fewshot" \
    --a60 "$RUN/eval_a60" --b80 "$RUN/eval_b80" --output "$RUN/comparison"
echo "VALIDATION_COMPLETE=$RUN"
