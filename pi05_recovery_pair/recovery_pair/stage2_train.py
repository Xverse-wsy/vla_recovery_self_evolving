"""New Stage2 optimizer initialized from Stage1 weights, never Stage1 resume."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .config import Config
from .engine import distributed_info, optimizer_update
from .features import sha256
from .openpi_backend import load_backend, ObservationLoader
from .stage2_data import POLICY, audit, prepare, validate_config
from .stage2_sampler import Stage2Sampler
from .trainer import base_identity, check_sampler, init_distributed

OBJECTIVE = 'stage2_native_fm_full_flow_continuous_pair_v1'


def apply_trainable(backend, weights):
    named = {n:p for n,p in backend.named_parameters() if p.requires_grad}
    if set(weights)!=set(named):
        raise ValueError('Trainable parameter inventory differs from the pinned model')
    with torch.no_grad():
        for name,param in named.items():
            value=weights[name]
            if value.shape!=param.shape or not torch.isfinite(value).all():
                raise ValueError('Invalid trained parameter: '+name)
            param.copy_(value.to(param))


def check_stage1(state, base, *, allow_partial=False, max_updates=None,prepared_sha=None):
    identity=state['identity']
    origin=identity.get('base') or {}
    if (identity.get('mode')!='fm_only' or identity.get('objective_version')!='native_fm_masked_window_mean_v1'
            or any(origin.get(k)!=base[k] for k in ('weights_sha256','norm_sha256'))
            or identity['config'].get('fm_epochs')!=50):
        raise ValueError('Expected the native-FM 50-epoch Stage1 checkpoint from this original base')
    if prepared_sha is not None and identity.get('prepared_sha256')!=prepared_sha:
        raise ValueError('Stage1 checkpoint was trained on a different prepared pool')
    completed=state['sampler'].get('epochs_completed',0)
    if completed!=50 and not (allow_partial and max_updates is not None and 0<max_updates<=2):
        raise ValueError('Stage1 is incomplete; partial initialization is only allowed for <=2-update smoke')
    return dict(stage1_updates=state['update'],stage1_epochs_completed=completed,
                partial_stage1_smoke=completed!=50)


def memory_limit(device, cap):
    if not str(device).startswith('cuda'):
        return
    torch.cuda.set_device(torch.device(device))
    free,total=torch.cuda.mem_get_info()
    if cap<=0 or (cap+4)*2**30>free:
        raise RuntimeError('Need the allocator cap plus 4GiB free margin')
    torch.cuda.set_per_process_memory_fraction(cap*2**30/total)


def train_loop(backend, loader, records, pairs, fm_ids, config, output, identity, *,
               resume=None,max_updates=None,save_every=50,keep_last=3,hash_parameters=False):
    if save_every<1 or keep_last<1 or (max_updates is not None and max_updates<1):
        raise ValueError('Positive checkpoint intervals and update bounds required')
    sampler=Stage2Sampler(records,pairs,fm_ids,config)
    rank,world=distributed_info()
    out=Path(output)
    optimizer=torch.optim.AdamW([p for p in backend.parameters() if p.requires_grad],lr=config.learning_rate,weight_decay=0.)
    if resume:
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        if saved['identity']!=identity:
            raise ValueError('Stage2 resume base/init/data/config/objective differs')
        apply_trainable(backend,saved['trainable'])
        optimizer.load_state_dict(saved['optimizer'])
        sampler.load_state_dict(saved['sampler'])
        if sampler.updates!=saved['update']:
            raise ValueError('Checkpoint and sampler update differ')
    # Audit BEFORE restoring stochastic state: audits must not shift a resumed RNG.
    first=next(r for r in records if r.sample_id in set(fm_ids))
    sampler_check=check_sampler(backend,loader,first,checkpoint_steps=config.checkpoint_sampling)
    if resume:
        torch.set_rng_state(saved['torch_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
        if torch.cuda.is_available() and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state(saved['cuda_rng'])
        del saved
    hashes={}
    if hash_parameters:
        from .fm_validation import parameter_hash
        hashes={'frozen_before':parameter_hash(backend,False),'trainable_before':parameter_hash(backend,True)}
    if rank==0:
        (out/'run.json').write_text(json.dumps(dict(identity=identity,world_size=world,
            sampler_audit=sampler_check,optimizer_reset_at_stage_boundary=True,
            max_episode_reuses=config.max_episode_reuses,expected_updates=sampler.nominal_updates,
            pair_sampling='episode -> event -> negative window -> one reference',
            fm_global_batch=config.fm_batch,pair_global_batch_max=config.pair_batch,
            no_stage2_warmup=True,validation_enabled=False),indent=2))

    def checkpoint():
        if rank!=0:return
        state=dict(identity=identity,update=sampler.updates,
            trainable={n:p.detach().cpu().clone() for n,p in backend.named_parameters() if p.requires_grad},
            optimizer=optimizer.state_dict(),sampler=sampler.state_dict(),torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)
        dest=out/f'update_{sampler.updates:06d}.pt';temp=dest.with_suffix('.tmp')
        torch.save(state,temp);temp.replace(dest)
        for old in sorted(out.glob('update_*.pt'))[:-keep_last]:old.unlink()

    while not sampler.done and (max_updates is None or sampler.updates<max_updates):
        if str(next(backend.parameters()).device).startswith('cuda'):
            free,_=torch.cuda.mem_get_info()
            available=torch.tensor(float(free/2**30),device=next(backend.parameters()).device)
            if world>1:torch.distributed.all_reduce(available,op=torch.distributed.ReduceOp.MIN)
            if available.item()<4:raise RuntimeError('Our training stops to preserve 4GiB shared GPU headroom')
        plan=sampler.draw()
        metric=optimizer_update(backend,optimizer,loader,records,plan,config,
            update_index=sampler.updates-1,total_updates=sampler.nominal_updates,mode='continuous_pair')
        metric.update(update=sampler.updates,fm_count=len(plan['fm_ids']),pair_count=len(plan['pairs']),episode_visits=dict(sampler.visits))
        if rank==0:
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric,allow_nan=False)+'\n')
            print(json.dumps(metric),flush=True)
        if sampler.updates%save_every==0:checkpoint()
    checkpoint()
    if hash_parameters:
        hashes.update(frozen_after=parameter_hash(backend,False),trainable_after=parameter_hash(backend,True))
        if hashes['frozen_before']!=hashes['frozen_after']:raise RuntimeError('Frozen VLM weights changed')
    result=dict(state='complete' if sampler.done else 'stopped',reason='episode_budget_complete' if sampler.done else 'requested_update_limit',
        updates=sampler.updates,episode_visits=sampler.visits,parameter_hashes=hashes,
        checkpoint=str(out/f'update_{sampler.updates:06d}.pt'),training_policy=POLICY)
    if torch.cuda.is_available():result['cuda_peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
    if rank==0:(out/'status.json').write_text(json.dumps(result,indent=2))
    return result


def run_training(prepared,config_path,stage1_checkpoint,base_checkpoint,norm_stats,output,*,
                 resume=None,max_updates=None,allow_partial_stage1=False,memory_gib=20.,save_every=50,keep_last=3,hash_parameters=False):
    config=Config.load(config_path);validate_config(config)
    device=init_distributed();rank,world=distributed_info()
    out=Path(output).resolve()
    inputs=[Path(prepared).resolve(),Path(base_checkpoint).resolve(),Path(stage1_checkpoint).resolve().parent]
    if any(out==p or p in out.parents for p in inputs):raise ValueError('Stage2 training output must be independent of inputs')
    if rank==0:out.mkdir(parents=True,exist_ok=bool(resume))
    if world>1:torch.distributed.barrier()
    try:
        if rank==0:(out/'status.json').write_text(json.dumps(dict(state='auditing_inputs')))
        records,pairs,fm_ids,digest=audit(prepared,config)
        base=base_identity(base_checkpoint,norm_stats)
        first=torch.load(stage1_checkpoint,map_location='cpu',weights_only=False)
        source_report=json.loads((Path(prepared)/'report.json').read_text())
        stage1_info=check_stage1(first,base,allow_partial=allow_partial_stage1,max_updates=max_updates,
                                prepared_sha=source_report['source_stage1_sha256'])
        if any(r.spec.normalization_sha256!=base['norm_sha256'] for r in records):
            raise ValueError('Training normalization differs from the original base')
        stage1_digest=sha256(stage1_checkpoint)
        identity_config=config.to_dict()
        for key in ('fm_micro_batch','pair_micro_batch'):identity_config.pop(key)
        identity=dict(objective_version=OBJECTIVE,mode='continuous_pair',base=base,config=identity_config,
            prepared_sha256=digest,stage1_checkpoint=str(Path(stage1_checkpoint).resolve()),
            stage1_checkpoint_sha256=stage1_digest,stage1_info=stage1_info)
        memory_limit(device,memory_gib)
        torch.manual_seed(config.seed);np.random.seed(config.seed);random.seed(config.seed)
        if rank==0:(out/'status.json').write_text(json.dumps(dict(state='loading_stage1_weights',**stage1_info)))
        backend=load_backend(base_checkpoint,norm_stats,config,device)
        apply_trainable(backend,first['trainable']);del first
        if rank==0:(out/'status.json').write_text(json.dumps(dict(state='training',**stage1_info)))
        result=train_loop(backend,ObservationLoader(backend),records,pairs,fm_ids,config,out,identity,
            resume=resume,max_updates=max_updates,save_every=save_every,keep_last=keep_last,hash_parameters=hash_parameters)
        if sha256(stage1_checkpoint)!=stage1_digest:raise ValueError('Source Stage1 checkpoint changed')
        return result
    except BaseException as exc:
        if rank==0:(out/'status.json').write_text(json.dumps(dict(state='failed',error=repr(exc))))
        raise


def export(checkpoint,output):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    identity=saved['identity']
    if identity.get('objective_version')!=OBJECTIVE:
        raise ValueError('Export requires an independent Stage2 checkpoint')
    base=identity['base']
    if base_identity(base['checkpoint'],base['norm_stats'])!=base:
        raise ValueError('Original base or normalization assets changed')
    out=Path(output).resolve()
    protected=[Path(base['checkpoint']).resolve(),Path(checkpoint).resolve().parent,
               Path(identity['stage1_checkpoint']).resolve().parent]
    if any(out==p or p in out.parents for p in protected):
        raise ValueError('Export must be outside the original base and checkpoint directories')
    out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(identity['config'],indent=2))
    config=Config.load(out/'config.json')
    backend=load_backend(base['checkpoint'],base['norm_stats'],config,'cpu')
    apply_trainable(backend,saved['trainable'])
    torch.save(backend.policy.state_dict(),out/'full_weights.pt')
    (out/'source.json').write_text(json.dumps(dict(identity=identity,source_update=saved['update'],
        source_checkpoint_sha256=sha256(checkpoint),stage='stage2',full_weights_sha256=sha256(out/'full_weights.pt')),indent=2))
    return dict(exported=str(out),source_update=saved['update'])


def export_stage1(checkpoint,base_checkpoint,norm_stats,output,allow_partial=False):
    """Read-only Stage1 export: no temporary files inside its training directory."""
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    base=base_identity(base_checkpoint,norm_stats)
    info=check_stage1(saved,base,allow_partial=allow_partial,max_updates=2 if allow_partial else None)
    out=Path(output).resolve()
    if any(out==p or p in out.parents for p in (Path(checkpoint).resolve().parent,Path(base_checkpoint).resolve())):
        raise ValueError('Export must be outside Stage1 training and original base directories')
    out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(saved['identity']['config'],indent=2))
    config=Config.load(out/'config.json')
    backend=load_backend(base_checkpoint,norm_stats,config,'cpu')
    apply_trainable(backend,saved['trainable'])
    torch.save(backend.policy.state_dict(),out/'full_weights.pt')
    (out/'source.json').write_text(json.dumps(dict(stage='stage1',base=base,stage1_info=info,
        stage1_checkpoint_sha256=sha256(checkpoint),full_weights_sha256=sha256(out/'full_weights.pt')),indent=2))
    return dict(exported=str(out),**info)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    for key in ('stage1-prepared','matching','config','output'):p.add_argument('--'+key,required=True)
    p=sub.add_parser('audit');p.add_argument('--prepared',required=True)
    p=sub.add_parser('train')
    for key in ('prepared','config','stage1-checkpoint','base-checkpoint','norm-stats','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--resume');p.add_argument('--max-updates',type=int)
    p.add_argument('--allow-partial-stage1',action='store_true')
    p.add_argument('--max-cuda-memory-gb',type=float,default=20.)
    p.add_argument('--save-every',type=int,default=50);p.add_argument('--keep-last',type=int,default=3)
    p.add_argument('--hash-parameters',action='store_true')
    p=sub.add_parser('export')
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('export-stage1')
    for key in ('checkpoint','base-checkpoint','norm-stats','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--allow-partial-stage1',action='store_true')
    a=parser.parse_args()
    if a.command=='prepare':result=prepare(a.stage1_prepared,a.matching,Config.load(a.config),a.output)
    elif a.command=='audit':
        records,pairs,fm,digest=audit(a.prepared)
        result=dict(windows=len(records),pairs=len(pairs),fm_windows=len(fm),sha256=digest)
    elif a.command=='export':result=export(a.checkpoint,a.output)
    elif a.command=='export-stage1':result=export_stage1(a.checkpoint,a.base_checkpoint,a.norm_stats,a.output,a.allow_partial_stage1)
    else:
        result=run_training(a.prepared,a.config,a.stage1_checkpoint,a.base_checkpoint,a.norm_stats,a.output,
            resume=a.resume,max_updates=a.max_updates,allow_partial_stage1=a.allow_partial_stage1,
            memory_gib=a.max_cuda_memory_gb,save_every=a.save_every,keep_last=a.keep_last,hash_parameters=a.hash_parameters)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
