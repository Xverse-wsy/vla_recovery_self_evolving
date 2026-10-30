"""A800 bounded CPU import audit. Keeps all unreviewed local behavior UNKNOWN."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from recovery_pair.config import Config
from recovery_pair.data import manifest_segmented,manifest_autonomous,build_windows,QuantileAssets
from recovery_pair.features import read_start_image
from recovery_pair.schema import save_windows
from recovery_pair.source_identity import verify_installed_rlinf
from recovery_pair.openpi_backend import RLINF_COMMIT
import rlinf

ROOT=Path('/home/wenchaoxu/wsy/vla-recovery-posttrain-loop')
OUT=HERE/'outputs/real_import_v1'
manifest=[]
for path in ['recovery_success20_segmented','recovery_failure20_segmented']:
    manifest.extend(manifest_segmented(ROOT/'dataset_subsets/pi05_recovery_segmented'/path))
manifest.extend(manifest_autonomous(ROOT/'dataset_subsets/pi05_autonomous20_matched_recovery20_20261029/autonomous_success20_lerobot'))
instructions=defaultdict(set)
for e in manifest:
    instructions[e['task_id']].add(e['instruction'])
if any(len(v)!=1 for v in instructions.values()):
    raise RuntimeError('Canonical task identity maps to conflicting instructions')
config=Config()
norm=ROOT/'models/RLinf-Pi05-LIBERO-SFT/physical-intelligence/libero/norm_stats.json'
records=build_windows(manifest,QuantileAssets(norm),config,OUT)
counts=Counter((r.split,r.controller,r.local_quality) for r in records)
state_only=[r for r in records if r.source.get('state_only')]
for r in [records[0],records[-1],state_only[0]]:
    for view,ref in r.image_refs.items():
        if read_start_image(ref).shape!=(256,256,3):
            raise RuntimeError('Unexpected actual start RGB shape')
report=dict(state='real_import_verified',segments=len(manifest),
    parent_episodes=len({r.episode_id for r in records}),tasks=len(instructions),windows=len(records),
    counts={'/'.join(k):v for k,v in counts.items()},handoff_states=len(state_only),
    normalization_sha256=QuantileAssets(norm).sha,
    source_identity=verify_installed_rlinf(rlinf,RLINF_COMMIT),
    local_labels='unknown pending local behavior evidence',
    no_matching_or_training=True)
(OUT/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
