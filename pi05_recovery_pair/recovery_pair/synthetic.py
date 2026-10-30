"""Explicit synthetic CPU fixture. Never substituted for real robot validation."""
from dataclasses import replace
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from .config import Config, MatchThresholds
from .data import QuantileAssets, build_windows, write_jsonl
from .schema import save_windows
from .pipeline import prepare
from .toy import ToyFlow
from .trainer import train


def fixture(output, config=None):
    root=Path(output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    config=config or replace(Config(), max_episode_reuses=2, fm_only_updates=3,
        coverage=MatchThresholds(proprio_rms=.5,image_cosine=.1),
        pairing=MatchThresholds(position_m=.01,rotation_deg=5.,proprio_rms=.2,image_cosine=.05))
    norm=root/'norm_stats.json'
    norm.write_text(json.dumps({'norm_stats':{k:{'q01':[-1]*d,'q99':[1]*d}
        for k,d in [('actions',7),('state',8)]}}))
    image=io.BytesIO()
    Image.new('RGB',(16,16),(80,90,100)).save(image,format='PNG')
    manifest=[]
    for task in range(6):
        for owner,quality in [('VLA','positive'),('VLA','negative'),('code','positive'),('code','negative')]:
            eid=f'task{task}-{owner}-{quality}'
            path=root/(eid+'.parquet')
            rows=[dict(state=[0.,0.,.2,0.,0.,0.,.02,.02],
                actions=[.6 if quality=='positive' else -.6]*7,
                image={'bytes':image.getvalue(),'path':None},
                wrist_image={'bytes':image.getvalue(),'path':None}) for _ in range(12)]
            pq.write_table(pa.Table.from_pylist(rows),path)
            context=dict(phase='approach',target_object='block',gripper_state='open',
                held_object='empty',contact_state='none',progress=.2,critical_state_verified=True,
                progress_scorer_id='synthetic_score_v1',evidence={'source':'synthetic_fixture'})
            manifest.append(dict(segment_id=eid,parquet=str(path),task_id=f'task{task}',
                episode_id=eid,initial_group_id=f'initial{task}',initialization_verified=True,
                event_id='event-'+eid,controller=owner,instruction='place block',
                parent_start_step=0,length=12,state_layout='eef_xyz_axisangle_gripper2',
                handoff_start=owner=='code',
                annotations=[dict(start=0,end=12,local_quality=quality,
                    quality_evidence=dict(verified_local_actions=True,negative_behavior_confirmed=quality=='negative',
                                          reviewer='synthetic_generator',evidence='synthetic_only'))],
                start_context={str(i):context for i in range(12)}))
    assets=QuantileAssets(norm)
    records=build_windows(manifest,assets,config,root/'windows')
    for r in records:
        r.visual_features=dict(encoder_sha256='synthetic_frozen_features',
            calibration_id='synthetic_no_real_encoder',views={'front':[1.,0.], 'wrist':[1.,0.]})
    save_windows(root/'windows/windows.jsonl',records)
    report=prepare(records,config,root/'prepared')
    (root/'config.json').write_text(json.dumps(config.to_dict(),indent=2))
    return config, records, report


def loader(records):
    return {'state':torch.tensor([r.proprio for r in records],dtype=torch.float32)}


def smoke(output,updates=2):
    config,records,report=fixture(output)
    torch.manual_seed(config.seed)
    backend=ToyFlow(config.horizon,config.action_dim,steps=config.flow_steps)
    result=train(backend,loader,Path(output)/'prepared',config,Path(output)/'run',
        max_updates=updates,save_every=1,validate_every=1)
    return dict(synthetic_only=True,index=report,training=result)
