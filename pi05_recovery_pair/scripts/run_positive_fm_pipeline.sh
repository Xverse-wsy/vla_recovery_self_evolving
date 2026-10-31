#!/usr/bin/env bash
# Current positive-only pipeline. Each stage must be explicitly invoked.
set -euo pipefail
PAIR_STAGE=${1:-help}
PAIR_CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PAIR_CODE"
export PYTHONPATH="$PAIR_CODE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export JAX_PLATFORMS=cpu
PAIR_PYTHON=${PAIR_PYTHON:-python}
if [[ "$PAIR_STAGE" == help || "$PAIR_STAGE" == --help ]]; then
  echo "Usage: source scripts/your.local.env; bash $0 prepare|train|export"
  echo 'prepare: build libraries, match CAP successes, prepare ALL admitted positives, deep audit'
  echo 'train: explicitly start native FM (one machine, one or more GPUs); no heldout/early stopping'
  echo 'export: materialize PAIR_EXPORT_CHECKPOINT into a new full-weights directory'
  exit 0
fi
: "${PAIR_RUN:?Set a new absolute output directory}"
: "${PAIR_DATA_ROOT:?Set local data root}"
: "${PAIR_CHECKPOINT:?Set ORIGINAL RLinf few-shot checkpoint}"
: "${PAIR_NORM_STATS:?Set ORIGINAL norm_stats.json}"
# Normalize relative paths before producing immutable artifacts.
PAIR_RUN=$("$PAIR_PYTHON" -c 'import pathlib,sys;print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$PAIR_RUN")
case "$PAIR_STAGE" in
prepare)
  : "${PAIR_VISION_GPU:?Select one physical GPU for frozen image features}"
  PAIR_EXPERT_DATASET=${PAIR_EXPERT_DATASET:-$PAIR_DATA_ROOT/dataset_subsets/rlinf_pi05_libero_fewshot40_20261021}
  test -f "$PAIR_EXPERT_DATASET/episodes.json"
  PAIR_SOURCES=${PAIR_SOURCES:-configs/data_sources.example.json}
  PAIR_CONFIG=${PAIR_CONFIG:-configs/libero_pi05_positive_fm.json}
  # Validate paths/config before creating a run; preserve input files.
  "$PAIR_PYTHON" - "$PAIR_SOURCES" "$PAIR_CONFIG" "$PAIR_DATA_ROOT" "$PAIR_NORM_STATS" "$PAIR_CHECKPOINT" <<'PY'
import json,sys
from pathlib import Path
from recovery_pair.config import Config
sources,algorithm,root,norm,checkpoint=map(Path,sys.argv[1:])
c=Config.load(algorithm)
assert c.pair_batch==0 and c.eta==0,'Use the positive-FM configuration'
assert c.fm_epochs==50,'Current stage 1 requires 50 complete epochs'
assert c.cap_success_admission=='segment_suffix'
assert root.is_dir() and norm.is_file() and (checkpoint/'model.safetensors').is_file()
s=json.loads(sources.read_text())
configured=Path(s['norm_stats']).expanduser()
assert (configured if configured.is_absolute() else root/configured).resolve()==norm.resolve(), 'Source and training norm paths differ'
PY
  mkdir -p -- "$(dirname -- "$PAIR_RUN")"
  mkdir -- "$PAIR_RUN"
  cp -- "$PAIR_CONFIG" "$PAIR_RUN/algorithm.json"
  "$PAIR_PYTHON" - "$PAIR_SOURCES" "$PAIR_RUN" <<'PY'
import json,sys
from pathlib import Path
source,out=map(Path,sys.argv[1:])
s=json.loads(source.read_text())
s['algorithm_config']=str(out/'algorithm.json')
(out/'sources.json').write_text(json.dumps(s,indent=2))
PY
  "$PAIR_PYTHON" -u -m recovery_pair build-libraries \
    --sources-config "$PAIR_RUN/sources.json" --data-root "$PAIR_DATA_ROOT" \
    --output "$PAIR_RUN/banks"
  CUDA_VISIBLE_DEVICES="$PAIR_VISION_GPU" PAIR_PYTHON="$PAIR_PYTHON" \
    bash scripts/run_success_admission.sh \
    "$PAIR_RUN/banks/windows/windows.jsonl" "$PAIR_CHECKPOINT" \
    "$PAIR_RUN/algorithm.json" "$PAIR_RUN/matching" cuda:0 \
    "${PAIR_VISION_MEMORY_GIB:-8}" "${PAIR_VISION_BATCH:-2}"
  "$PAIR_PYTHON" -u -m recovery_pair prepare-positive-fm \
    --admission "$PAIR_RUN/matching/admission" --config "$PAIR_RUN/algorithm.json" \
    --all-train --expert-dataset "$PAIR_EXPERT_DATASET" --expected-expert-episodes 40 \
    --output "$PAIR_RUN/prepared"
  "$PAIR_PYTHON" -u -m recovery_pair audit --prepared "$PAIR_RUN/prepared" \
    --source-norm-stats "$PAIR_NORM_STATS"
  ;;
train)
  : "${CUDA_VISIBLE_DEVICES:?Select available training GPUs explicitly}"
  PAIR_NPROC=${NPROC_PER_NODE:-1}
  "$PAIR_PYTHON" - "$PAIR_RUN/prepared" "$PAIR_NPROC" "$CUDA_VISIBLE_DEVICES" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);n=int(sys.argv[2]);gpus=sys.argv[3].split(',')
assert n>0 and n==len(gpus) and len(set(gpus))==n,'NPROC_PER_NODE must equal the selected GPU count'
r=json.loads((p/'report.json').read_text())
assert r['split_policy']=='all_train_v1' and not r['validation_enabled']
from recovery_pair.config import Config
c=Config.load(p.parent/'algorithm.json')
assert c.fm_epochs==50,'Re-prepare with the 50-epoch configuration'
b=r['training_budget']
assert b['sampling_policy']=='positive_epoch_permutation_v1' and b['target_epochs']==50
assert b['windows_per_epoch']==r['positive_windows'] and not b['drop_last'] and not b['pad_last']
assert b['expected_updates']==((r['positive_windows']+c.fm_batch-1)//c.fm_batch)*50
print('Full epoch training budget:',json.dumps(b),flush=True)
PY
  PAIR_ARGS=(--prepared "$PAIR_RUN/prepared" --config "$PAIR_RUN/algorithm.json"
    --checkpoint "$PAIR_CHECKPOINT" --norm-stats "$PAIR_NORM_STATS"
    --output "$PAIR_RUN/training" --mode fm_only
    --save-every "${PAIR_SAVE_EVERY:-50}" --keep-last "${PAIR_KEEP_LAST:-3}"
    --max-cuda-memory-gb "${PAIR_TRAIN_MEMORY_GIB:-17}")
  if [[ -n "${PAIR_RESUME:-}" ]]; then PAIR_ARGS+=(--resume "$PAIR_RESUME"); fi
  if [[ -n "${PAIR_MAX_UPDATES:-}" ]]; then PAIR_ARGS+=(--max-updates "$PAIR_MAX_UPDATES"); fi
  if [[ "$PAIR_NPROC" == 1 ]]; then
    exec "$PAIR_PYTHON" -u -m recovery_pair train "${PAIR_ARGS[@]}"
  else
    exec "$PAIR_PYTHON" -u -m torch.distributed.run --standalone \
      --nproc_per_node="$PAIR_NPROC" -m recovery_pair train "${PAIR_ARGS[@]}"
  fi
  ;;
export)
  : "${PAIR_EXPORT_CHECKPOINT:?Select a completed training update_NNNNNN.pt}"
  exec "$PAIR_PYTHON" -u -m recovery_pair export \
    --checkpoint "$PAIR_EXPORT_CHECKPOINT" --output "$PAIR_RUN/exported"
  ;;
*) echo "Unknown stage: $PAIR_STAGE. Expected prepare|train|export" >&2; exit 2 ;;
esac
