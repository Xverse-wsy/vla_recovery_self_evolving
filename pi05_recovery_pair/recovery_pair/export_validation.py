"""Validate the documented export CLI and reload exported weights on a real GPU."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys

import torch

from .config import Config
from .fm_validation import parameter_hash
from .openpi_backend import load_backend, ObservationLoader
from .pipeline import audit_prepared
from .trainer import base_identity


def validate(validation, output, device='cuda:0', memory_gib=12., existing_export=None):
    previous = Path(validation).resolve()
    report = json.loads((previous/'report.json').read_text())
    if report.get('state') != 'complete' or not report.get('uninterrupted_equivalence_verified'):
        raise ValueError('Complete GPU training review must pass before export validation')
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = out/'status.json'
    try:
        status.write_text(json.dumps({'state':'exporting_with_documented_cli'}))
        checkpoint = previous/'run/update_000002.pt'
        exported = Path(existing_export).resolve() if existing_export else out/'exported'
        if existing_export:
            provenance=json.loads((exported/'source.json').read_text())
            if (provenance['base']!=report['base'] or provenance['prepared_sha256']!=report['prepared_sha256']
                    or provenance['mode']!='fm_only'):
                raise ValueError('Existing export provenance differs from reviewed checkpoint')
        else:
            subprocess.run([sys.executable,'-m','recovery_pair','export','--checkpoint',str(checkpoint),
                            '--output',str(exported)], check=True)
        config = Config.load(exported/'config.json')
        records,_,_=audit_prepared(report['prepared'],config)
        if not device.startswith('cuda'):
            raise ValueError('Export reload review requires real CUDA')
        torch.cuda.set_device(torch.device(device))
        free,total=torch.cuda.mem_get_info()
        limit=int(memory_gib*1024**3)
        if limit<=0 or limit+4*1024**3>free:
            raise RuntimeError('Insufficient free GPU memory for export reload cap plus4GiB margin')
        torch.cuda.set_per_process_memory_fraction(limit/total)
        status.write_text(json.dumps({'state':'reloading_exported_weights_on_gpu'}))
        base=report['base']
        backend=load_backend(exported,base['norm_stats'],config,device)
        if (parameter_hash(backend,True)!=report['final_trainable_sha256']
                or parameter_hash(backend,False)!=report['frozen_sha256']):
            raise RuntimeError('Exported model parameters differ from reviewed final checkpoint')
        loader=ObservationLoader(backend)
        record=next(r for r in records if r.split=='train')
        generator=torch.Generator(device=device).manual_seed(765)
        noise=torch.randn(1,config.horizon,config.action_dim,device=device,generator=generator)
        action=backend.deployment_sample(loader([record]),noise)
        if action.shape!=(1,config.horizon,config.action_dim) or not torch.isfinite(action).all():
            raise RuntimeError('Exported model inference is invalid')
        if base_identity(base['checkpoint'],base['norm_stats'])!=base:
            raise RuntimeError('Original base checkpoint changed')
        result=dict(state='complete', documented_export_cli_passed=not bool(existing_export),
            existing_documented_export_reused=bool(existing_export),
            full_weights_path=str(exported/'full_weights.pt'),
            full_weights_bytes=(exported/'full_weights.pt').stat().st_size,
            trainable_and_frozen_parameters_exactly_match_training=True,
            cuda_reload_passed=True, native_deployment_inference_passed=True,
            generated_action_shape=list(action.shape), original_base_unchanged=True,
            cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            cuda_peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
            scope='Weight export, exact parameter reloading and finite native action inference; no LIBERO simulation')
        (out/'report.json').write_text(json.dumps(result,indent=2))
        status.write_text(json.dumps({'state':'complete'}))
        del backend,loader,action
        gc.collect()
        torch.cuda.empty_cache()
        return result
    except Exception as exc:
        status.write_text(json.dumps({'state':'failed','error':str(exc)}))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--max-cuda-memory-gb',type=float,default=12.)
    parser.add_argument('--existing-export',help='Recheck an already materialized export; provenance and exact parameter hashes must match')
    args=parser.parse_args()
    print(json.dumps(validate(args.validation,args.output,args.device,args.max_cuda_memory_gb,args.existing_export),indent=2))


if __name__=='__main__':
    main()
