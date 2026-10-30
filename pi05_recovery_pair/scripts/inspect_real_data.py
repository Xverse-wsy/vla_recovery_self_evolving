"""Read-only A800 metadata probe. No training, GPU allocation, replay or credentials."""
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT=Path('/home/wenchaoxu/wsy/vla-recovery-posttrain-loop')
DATASETS={
    'autonomous20':ROOT/'dataset_subsets/pi05_autonomous20_matched_recovery20_20261029/autonomous_success20_lerobot',
    'cap_success':ROOT/'dataset_subsets/pi05_recovery_segmented/recovery_success20_segmented',
    'cap_failure':ROOT/'dataset_subsets/pi05_recovery_segmented/recovery_failure20_segmented',
}
output={'datasets':{},'packages':{}}
for package in ('torch','numpy','pyarrow','pillow','omegaconf'):
    try:
        output['packages'][package]=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        output['packages'][package]=None
spec=importlib.util.find_spec('rlinf')
if spec and spec.origin:
    repository=next((p for p in Path(spec.origin).parents if (p/'.git').exists()),None)
    output['rlinf_path']=spec.origin
    if repository:
        output['rlinf_commit']=subprocess.check_output(['git','-C',str(repository),'rev-parse','HEAD'],text=True).strip()
for name,root in DATASETS.items():
    info=json.loads((root/'meta/info.json').read_text())
    provenance=[json.loads(x) for x in (root/'meta/provenance.jsonl').read_text().splitlines() if x.strip()]
    first=provenance[0]
    eid=int(first.get('episode_index',first.get('exported_episode_index',0)))
    parquet=root/info['data_path'].format(episode_chunk=eid//info['chunks_size'],episode_index=eid)
    import pyarrow.parquet as pq
    table=pq.read_table(parquet,columns=['state','actions'])
    selection=['episode_index','exported_episode_index','parent_episode_index','start','end','owner',
        'suite','task_id','trial_id','instruction','original_episode','action_mapping','canonical_task_index',
        'category','completion_label','native_control_hz','logical_dataset_fps','prefix_steps']
    rec=first.get('source_record',{})
    output['datasets'][name]=dict(root=str(root),episodes=info.get('total_episodes'),
        frames=info.get('total_frames'),fps=info.get('fps'),features=info.get('features'),
        provenance_count=len(provenance),provenance_keys=list(first),
        provenance_example={k:first[k] for k in selection if k in first},
        source_record_keys=list(rec),source_record_example={k:rec[k] for k in
            ['source_episode','suite','evaluation_task_id','trial_id','instruction','handoff_step','success'] if k in rec},
        parquet=str(parquet),schema=str(table.schema),first_rows=table.slice(0,2).to_pylist(),
        annotation_files=[str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and
            any(x in p.name.lower() for x in ['quality','annotat','local_label','review'])][:30])
    if name.startswith('cap_'):
        summary=root/Path(first['action_mapping']).parent/'segments.json'
        output['datasets'][name]['summary_exists']=summary.is_file()
        if summary.is_file():
            parent=json.loads(summary.read_text())
            output['datasets'][name]['prefix_steps']=parent['prefix_steps']
output['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.free,memory.total',
    '--format=csv,noheader,nounits'],text=True).strip()
print(json.dumps(output,ensure_ascii=False,indent=2))
