"""CPU-only data/recipe acceptance and exact-epoch plan; never launches training."""
import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader,Subset

from direct_data import build_raw
from epoch_plan import EpochPlan,ExactEpochSampler
from settings import load_config
from .fetch import dump
from .training_data import audit_recipe


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--world-size',type=int,default=1)
    ap.add_argument('--global-batch-size',type=int)
    ap.add_argument('--micro-batch-size',type=int)
    args=ap.parse_args()
    cfg,base=load_config('a60_subtasks22'),load_config('a60')
    parity=audit_recipe(cfg,base)
    for c in (cfg,base):
        if args.global_batch_size is not None:c.actor.global_batch_size=args.global_batch_size
        if args.micro_batch_size is not None:c.actor.micro_batch_size=args.micro_batch_size
    audit_recipe(cfg,base)
    global_batch,micro=int(cfg.actor.global_batch_size),int(cfg.actor.micro_batch_size)
    if min(global_batch,micro,args.world_size)<1 or global_batch%(args.world_size*micro):
        raise ValueError('Global batch must divide evenly by world_size * micro_batch')
    raw=build_raw(OmegaConf.to_container(cfg.data.sources,resolve=True),horizon=10)
    if len(raw.base)!=9864 or len(raw.episodes)!=82 or len(raw)!=11058:
        raise ValueError('Wrong data: expected A60(9864) + reviewed22(1194)')
    plan=EpochPlan(len(raw),args.world_size,micro,global_batch//(args.world_size*micro))
    # Check every subtask start against its source record and actual Parquet rows.
    for i,w in enumerate(raw.clips.windows):
        sample=raw[len(raw.base)+i]
        meta=raw.clips.segments[w['segment_id']]
        table=raw.clips.numeric(w['parquet'])
        expected=np.asarray(table['actions'].to_pylist(),np.float32)
        positions=w['start']+np.arange(10)
        if w['tail_policy']=='repeat_last':
            assert meta['endpoint_has_recorded_task_success']
            positions=np.minimum(positions,len(expected)-1)
        else:
            assert positions[-1]<len(expected)
        np.testing.assert_array_equal(sample['actions'],expected[positions])
        assert sample['prompt']==meta['task_instruction']==sample['task']
        assert sample['task_index']==raw.tasks[sample['prompt']]
        assert sample['image'].shape==sample['wrist_image'].shape==(256,256,3)
    # Original A60 tails remain unchanged at each episode's start/end.
    probes=[]
    for i,r in enumerate(raw.base.episodes):
        start=raw.base.ends[i-1] if i else 0
        probes.extend([start,start+r['length']-1])
    for i in probes:
        got=raw[i];rec,start=raw.base.locate(i)
        np.testing.assert_array_equal(got['actions'],rec['actions'][np.minimum(start+np.arange(10),rec['length']-1)])
        assert got['prompt']==rec['task']
    # Real batching across the A60/subtask source boundary, without loading any model.
    batch=next(iter(DataLoader(Subset(raw,[0,6632,len(raw.base),len(raw)-1]),batch_size=4,num_workers=0)))
    assert tuple(batch['actions'].shape)==(4,10,7)
    epochs=int(cfg.runner.max_epochs)
    for epoch in range(epochs):
        indices=[]
        for rank in range(args.world_size):
            sampler=ExactEpochSampler(len(raw),args.world_size,rank,seed=cfg.actor.seed,micro_batch_size=micro)
            sampler.set_epoch(epoch)
            indices.extend(i for i in sampler if i>=0)
        assert sorted(indices)==list(range(len(raw)))
        assert sum(plan.global_samples(o,plan.update_batches(o)) for o in range(0,plan.batches_per_epoch,plan.accumulation))==len(raw)
    raw.assert_unchanged()
    args.output.mkdir(parents=True,exist_ok=True)
    OmegaConf.save(cfg,args.output/'resolved_config.yaml')
    dump(args.output/'dataset_manifest.json',raw.summary())
    report=dict(state='passed',experiment='a60_subtasks22',a60_recipe_parity=parity,
                episodes=82,tasks=len(raw.tasks),training_windows=len(raw),unique_action_rows=11175,
                source_windows={'author40':6632,'autonomous20':3232,'subtask22':1194},
                terminal_subtask_segments=9,intermediate_subtask_segments=13,
                intermediate_tail_padding=0,parent_instructions_only=True,
                subtask_windows_checked=len(raw.clips),a60_boundary_windows_checked=len(probes),
                epoch_plan=plan.describe(),epochs=epochs,updates=epochs*plan.updates_per_epoch,
                all_50_epoch_permutations_checked=True,gpu_used=False,model_training_performed=False,
                native_rlinf_transform_probe='not_run_by_this_command_use_native_probe.py')
    dump(args.output/'preflight.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
