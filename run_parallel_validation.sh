#!/usr/bin/env bash
set -euo pipefail
: "${PI05_ROOT:?Set the project root}"
: "${PI05_EVAL_PYTHON:?Set the evaluation Python}"
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GPUS="${PI05_VALIDATION_GPUS:-1,3}"
RUN="$PI05_ROOT/training_runs/pi05_parallel_validation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
printf '%s\n' "$RUN" > "$PACKAGE/latest_parallel_validation.txt"
exec > >(tee -a "$RUN/validation.log") 2>&1
echo "PARALLEL_VALIDATION_OUTPUT=$RUN"
BASE_CHECKPOINT="$("$PI05_EVAL_PYTHON" "$PACKAGE/settings.py" base_checkpoint)"
"$PI05_EVAL_PYTHON" "$PACKAGE/validate_parallel.py" --output "$RUN/cpu_validation.json"
args=(--gpus "$GPUS" --label fewshot --checkpoint "$BASE_CHECKPOINT"
      --output "$RUN/evaluation" --tasks 0 --trials 1)
bash "$PACKAGE/run_eval_parallel.sh" "${args[@]}"
"$PI05_EVAL_PYTHON" - "$RUN" <<'PY'
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1])
files=list((r/'evaluation/episodes').glob('*/*/*'))
assert len(files)==8, 'Expected four videos and four result JSONs'
snapshot={str(p.relative_to(r)):[p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()] for p in files}
(r/'before_resume.json').write_text(json.dumps(snapshot))
PY
bash "$PACKAGE/run_eval_parallel.sh" "${args[@]}" --resume
"$PI05_EVAL_PYTHON" - "$RUN" <<'PY'
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1])
before=json.loads((r/'before_resume.json').read_text())
after={str(p.relative_to(r)):[p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()] for p in (r/'evaluation/episodes').glob('*/*/*')}
assert before==after, 'Resume changed completed episodes or videos'
status=json.loads((r/'evaluation/status.json').read_text())
assert status['state']=='complete' and status['summary']['episodes']==4
assert status['parallel']['exactly_once']
report=dict(state='complete', real_episodes=4, videos=4,
            global_shards=status['parallel']['num_shards'],
            duplicate_episodes=0, resume_preserves_results_and_videos=True)
(r/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
PY
echo "PARALLEL_VALIDATION_COMPLETE=$RUN"
