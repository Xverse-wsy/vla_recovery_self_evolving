#!/usr/bin/env bash
set -euo pipefail
: "${PI05_ROOT:?Set the existing project root}"
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PI05_TRAIN_PYTHON:-$PI05_ROOT/environment-setup/.venv-pi05-sft/bin/python}"
EVAL_PYTHON="${PI05_EVAL_PYTHON:-/home/wenchaoxu/miniconda3/envs/corrector-eval/bin/python}"
GPUS="${PI05_VALIDATION_GPUS:-1,3}"
RUN="$PI05_ROOT/training_runs/pi05_direct_validation_$(date +%Y%m%d_%H%M%S)"
mkdir "$RUN"
printf '%s\n' "$RUN" > "$PACKAGE/latest_validation.txt"
exec > >(tee -a "$RUN/validation.log") 2>&1
export PYTHONPATH="$PACKAGE:$PI05_ROOT/RLinf${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LIBERO_CONFIG_PATH="$PI05_ROOT/libero-resources/config"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_TYPE=standard ROBOT_PLATFORM=LIBERO
echo "VALIDATION_OUTPUT=$RUN"
BASE_CHECKPOINT="$("$PYTHON" "$PACKAGE/settings.py" base_checkpoint)"
CUDA_VISIBLE_DEVICES='' "$PYTHON" -Bu "$PACKAGE/validate_data.py" --output "$RUN/data"
CUDA_VISIBLE_DEVICES='' "$PYTHON" -Bu "$PACKAGE/validate_sampler.py" --output "$RUN/data/sampler16_validation.json"
CUDA_VISIBLE_DEVICES='' "$PYTHON" -Bu "$PACKAGE/validate_parallel.py" --output "$RUN/data/parallel_validation.json"
unset CUDA_VISIBLE_DEVICES RAY_ADDRESS
for experiment in a60 b80; do
    "$PYTHON" -Bu "$PACKAGE/train.py" --experiment "$experiment" --gpus "$GPUS" \
        --smoke --output "$RUN/$experiment"
done
for label in fewshot a60 b80; do
    if [[ "$label" == fewshot ]]; then
        checkpoint="$BASE_CHECKPOINT"
    else
        checkpoint="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"]+"/actor")' "$RUN/$label/status.json")"
    fi
    PI05_EVAL_PYTHON="$EVAL_PYTHON" bash "$PACKAGE/run_eval_parallel.sh" --gpus "$GPUS" \
        --label "$label" --checkpoint "$checkpoint" --output "$RUN/eval_$label" \
        --trials 1 --tasks 0
done
"$PYTHON" -Bu "$PACKAGE/verify_validation.py" --run "$RUN"
"$PYTHON" -Bu "$PACKAGE/compare_results.py" --fewshot "$RUN/eval_fewshot" \
    --a60 "$RUN/eval_a60" --b80 "$RUN/eval_b80" --output "$RUN/comparison"
echo "VALIDATION_COMPLETE=$RUN"
