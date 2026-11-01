"""Bounded real-GPU Stage2 update/resume/export audit; never a long training run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from .config import Config
from .fm_validation import parameter_hash
from .openpi_backend import load_backend, ObservationLoader
from .stage2_data import audit
from .stage2_train import memory_limit


def validate(prepared,config,stage1_checkpoint,base_checkpoint,norm_stats,output,memory_gib=20.):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    def status(phase,**extra):
        (out/'status.json').write_text(json.dumps(dict(state=phase,**extra),indent=2))
        print(json.dumps(dict(state=phase,**extra)),flush=True)
    common=[sys.executable,'-u','-m','recovery_pair.stage2_train','train','--prepared',str(prepared),
        '--config',str(config),'--stage1-checkpoint',str(stage1_checkpoint),'--base-checkpoint',str(base_checkpoint),
        '--norm-stats',str(norm_stats),'--allow-partial-stage1','--max-cuda-memory-gb',str(memory_gib),
        '--save-every','1','--keep-last','2','--hash-parameters']
    try:
        status('first_update')
        subprocess.run(common+['--output',str(out/'resumed'),'--max-updates','1'],check=True)
        status('resuming_second_update')
        subprocess.run(common+['--output',str(out/'resumed'),'--max-updates','2',
            '--resume',str(out/'resumed/update_000001.pt')],check=True)
        status('continuous_two_update_reference')
        subprocess.run(common+['--output',str(out/'continuous'),'--max-updates','2'],check=True)
        a=json.loads((out/'resumed/status.json').read_text());b=json.loads((out/'continuous/status.json').read_text())
        h=a['parameter_hashes'];hb=b['parameter_hashes']
        if (h['trainable_after']!=hb['trainable_after'] or h['frozen_after']!=hb['frozen_after'] or
                h['frozen_before']!=h['frozen_after'] or hb['frozen_before']!=hb['frozen_after'] or
                hb['trainable_before']==hb['trainable_after']):
            raise RuntimeError('Real-model resume/weight/frozen-parameter validation failed')
        rows=[json.loads(x) for x in (out/'resumed/metrics.jsonl').read_text().splitlines()]
        other=[json.loads(x) for x in (out/'continuous/metrics.jsonl').read_text().splitlines()]
        if rows!=other or any(r['pair_count']==0 for r in rows):
            raise RuntimeError('Pair updates absent or resumed metrics differ')
        status('exporting_reviewed_checkpoint')
        subprocess.run([sys.executable,'-u','-m','recovery_pair.stage2_train','export',
            '--checkpoint',str(out/'resumed/update_000002.pt'),'--output',str(out/'exported')],check=True)
        status('reloading_export_on_gpu')
        memory_limit('cuda:0',memory_gib)
        cfg=Config.load(config)
        model=load_backend(out/'exported',norm_stats,cfg,'cuda:0')
        if parameter_hash(model,True)!=h['trainable_after'] or parameter_hash(model,False)!=h['frozen_after']:
            raise RuntimeError('Export reload did not preserve exact parameters')
        records,_,fm_ids,_=audit(prepared,cfg)
        record=next(r for r in records if r.sample_id==fm_ids[0])
        noise=torch.randn(1,cfg.horizon,cfg.action_dim,device='cuda:0')
        action=model.deployment_sample(ObservationLoader(model)([record]),noise)
        if action.shape!=(1,cfg.horizon,cfg.action_dim) or not torch.isfinite(action).all():
            raise RuntimeError('Reloaded model produced invalid native actions')
        report=dict(state='complete',real_optimizer_updates=4,logical_fm_batch=cfg.fm_batch,
            logical_pair_batch=cfg.pair_batch,resume_matches_continuous_exactly=True,
            frozen_weights_unchanged=True,export_reload_exact=True,native_inference_finite=True,
            init_scope='partial Stage1 checkpoint explicitly permitted for bounded smoke only',
            stage2_loss='native FM plus continuous pair loss through full deployed Euler chain',
            libero_rollout_evaluated=False,formal_training_started=False,
            export=str(out/'exported'),metrics=rows)
        (out/'report.json').write_text(json.dumps(report,indent=2));status('complete')
        return report
    except BaseException as exc:
        status('failed',error=repr(exc));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('prepared','config','stage1-checkpoint','base-checkpoint','norm-stats','output'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--max-cuda-memory-gb',type=float,default=20.)
    a=p.parse_args()
    print(json.dumps(validate(a.prepared,a.config,a.stage1_checkpoint,a.base_checkpoint,a.norm_stats,a.output,a.max_cuda_memory_gb),indent=2))


if __name__=='__main__':main()
