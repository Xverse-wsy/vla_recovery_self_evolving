"""Explicit commands; nothing trains on import, indexing, or feature extraction."""
import argparse
import json
from pathlib import Path
import torch
from .config import Config
from .data import QuantileAssets, manifest_segmented, manifest_autonomous, read_jsonl, write_jsonl, build_windows
from .features import extract_features
from .pipeline import prepare, calibrate, audit_prepared
from .schema import load_windows, save_windows
from .libraries import create_libraries,audit_libraries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('default-config')
    p.add_argument('--output',required=True)
    p=subs.add_parser('manifest')
    p.add_argument('--segmented-dataset',action='append',default=[])
    p.add_argument('--autonomous-dataset',action='append',default=[])
    p.add_argument('--output',required=True)
    p=subs.add_parser('build')
    sources=p.add_mutually_exclusive_group(required=True)
    sources.add_argument('--manifest')
    sources.add_argument('--libraries')
    for arg in ('norm-stats','config','output'):
        p.add_argument('--'+arg,required=True)
    p=subs.add_parser('libraries')
    for arg in ('manifest','config','output'):
        p.add_argument('--'+arg,required=True)
    p=subs.add_parser('audit-libraries')
    p.add_argument('--libraries',required=True)
    p=subs.add_parser('build-libraries',help='Portable CPU build from existing classified datasets')
    for arg in ('sources-config','data-root','output'):
        p.add_argument('--'+arg,required=True)
    p=subs.add_parser('features')
    for arg in ('windows','encoder','preprocessing','cache','output'):
        p.add_argument('--'+arg,required=True)
    p.add_argument('--device',default='cpu')
    p=subs.add_parser('numeric-preview',help='Count numerical candidates without image filtering or pool admission')
    for arg in ('windows','config','output'):
        p.add_argument('--'+arg,required=True)
    p.add_argument('--proprio-rms',type=float,help='Optional explicit RMS gate; otherwise use config or report uncalibrated')
    p=subs.add_parser('calibrate')
    for arg in ('windows','reviews','config','output'):
        p.add_argument('--'+arg,required=True)
    p=subs.add_parser('prepare')
    for arg in ('windows','config','output'):
        p.add_argument('--'+arg,required=True)
    p=subs.add_parser('audit')
    p.add_argument('--prepared',required=True)
    p=subs.add_parser('smoke')
    p.add_argument('--output',required=True)
    p.add_argument('--updates',type=int,default=2)
    p=subs.add_parser('train')
    for arg in ('prepared','config','checkpoint','norm-stats','output'):
        p.add_argument('--'+arg,required=True)
    p.add_argument('--resume')
    p.add_argument('--max-updates',type=int)
    p.add_argument('--mode',choices=['continuous_pair','pull_only','fm_only'],default='continuous_pair')
    p.add_argument('--save-every',type=int,default=50)
    p.add_argument('--validate-every',type=int,default=25)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--keep-last',type=int,default=3)
    p=subs.add_parser('export')
    p.add_argument('--checkpoint',required=True,help='trusted local training update_N.pt')
    p.add_argument('--output',required=True)
    p=subs.add_parser('audit-model')
    for arg in ('windows','config','checkpoint','norm-stats','output'):
        p.add_argument('--'+arg,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--max-cuda-memory-gb',type=float,default=16.)
    args=parser.parse_args()
    result=None
    if args.command=='default-config':
        Path(args.output).write_text(json.dumps(Config().to_dict(),indent=2))
    elif args.command=='manifest':
        records=[]
        for path in args.segmented_dataset:
            records.extend(manifest_segmented(path))
        for path in args.autonomous_dataset:
            records.extend(manifest_autonomous(path))
        if not records:
            raise ValueError('Select at least one segmented or autonomous dataset')
        write_jsonl(args.output,records)
        result=dict(segments=len(records),bank_labels='existing classes preserved',
                    training_labels='existing four segment classes; optional finer evidence preserved')
    elif args.command=='libraries':
        result=create_libraries(read_jsonl(args.manifest),Config.load(args.config),args.output)
    elif args.command=='audit-libraries':
        result=dict(segments=len(audit_libraries(args.libraries)),state='library_audit_passed')
    elif args.command=='build-libraries':
        from .library_build import build_existing_libraries
        result=build_existing_libraries(args.sources_config,args.data_root,args.output)
    elif args.command=='build':
        config=Config.load(args.config)
        if args.libraries:
            manifest=audit_libraries(args.libraries)
            stored=Config.load(Path(args.libraries)/'config.json')
            for key in ('seed','split_fractions','conservative_task_split'):
                if getattr(config,key)!=getattr(stored,key):
                    raise ValueError(f'Existing library split configuration differs: {key}')
        else:
            manifest=read_jsonl(args.manifest)
        result=dict(windows=len(build_windows(manifest,QuantileAssets(args.norm_stats),config,args.output)),
                    window_libraries=str(Path(args.output)/'banks'))
    elif args.command=='numeric-preview':
        from .numeric_preview import preview_numeric
        result=preview_numeric(args.windows,Config.load(args.config),args.output,args.proprio_rms)
    elif args.command=='features':
        records=extract_features(load_windows(args.windows),args.encoder,args.preprocessing,args.cache,args.device)
        save_windows(args.output,records)
        result=dict(windows=len(records))
    elif args.command=='calibrate':
        result=calibrate(load_windows(args.windows),read_jsonl(args.reviews),Config.load(args.config),args.output)
    elif args.command=='prepare':
        result=prepare(load_windows(args.windows),Config.load(args.config),args.output)
    elif args.command=='audit':
        records,pairs,digest=audit_prepared(args.prepared)
        result=dict(windows=len(records),pairs=len(pairs),artifact_sha256=digest)
    elif args.command=='smoke':
        from .synthetic import smoke
        result=smoke(args.output,args.updates)
    elif args.command=='train':
        from .trainer import train, init_distributed, base_identity
        from .openpi_backend import load_backend,ObservationLoader
        config=Config.load(args.config)
        device=init_distributed()
        torch.manual_seed(config.seed)
        base=base_identity(args.checkpoint,args.norm_stats)
        # Audit before allocating the model.
        audit_prepared(args.prepared,config)
        backend=load_backend(args.checkpoint,args.norm_stats,config,device)
        result=train(backend,ObservationLoader(backend),args.prepared,config,args.output,base=base,
            resume=args.resume,mode=args.mode,max_updates=args.max_updates,save_every=args.save_every,
            validate_every=args.validate_every,patience=args.patience,keep_last=args.keep_last)
    elif args.command=='export':
        from .trainer import base_identity
        from .openpi_backend import load_backend
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        base=state['identity']['base']
        if not base or base_identity(base['checkpoint'],base['norm_stats'])!=base:
            raise ValueError('Export requires the identical original base checkpoint and normalization asset')
        config_path=Path(args.checkpoint).parent/'export_config.tmp.json'
        config_path.write_text(json.dumps(state['identity']['config']))
        try:
            config=Config.load(config_path)
        finally:
            config_path.unlink()
        backend=load_backend(base['checkpoint'],base['norm_stats'],config,'cpu')
        named=dict(backend.named_parameters())
        with torch.no_grad():
            for name,value in state['trainable'].items():
                named[name].copy_(value.to(named[name]))
        out=Path(args.output)
        out.mkdir(parents=True,exist_ok=False)
        torch.save(backend.policy.state_dict(),out/'full_weights.pt')
        (out/'config.json').write_text(json.dumps(config.to_dict(),indent=2))
        (out/'source.json').write_text(json.dumps(state['identity'],indent=2))
        result=dict(exported=str(out),normalization_asset=base['norm_stats'],
                    note='Evaluator must use identical H/K/model config and original normalization/tokenizer assets.')
    elif args.command=='audit-model':
        import numpy as np
        import pyarrow.parquet as pq
        from .features import read_start_image,sha256
        from .openpi_backend import load_backend,ObservationLoader
        from .trainer import check_sampler
        config=Config.load(args.config)
        dest=Path(args.output)
        if dest.exists():
            raise FileExistsError(dest)
        if args.device.startswith('cuda'):
            if args.max_cuda_memory_gb<=0:
                raise ValueError('Specify a positive allocator memory cap')
            torch.cuda.set_device(torch.device(args.device))
            free,total=torch.cuda.mem_get_info()
            limit=int(args.max_cuda_memory_gb*1024**3)
            if limit+4*1024**3>free:
                raise RuntimeError('Insufficient free GPU memory for cap plus 4GiB margin')
            torch.cuda.set_per_process_memory_fraction(limit/total)
        record=load_windows(args.windows)[0]
        if record.spec.normalization_sha256!=sha256(args.norm_stats):
            raise ValueError('Audit record normalization differs from base asset')
        backend=load_backend(args.checkpoint,args.norm_stats,config,args.device)
        loader=ObservationLoader(backend)
        ref=record.image_refs['front']
        length=min(config.horizon,int(record.payload()['valid_mask'].any(-1).sum()))
        raw=np.asarray(pq.read_table(ref['parquet'],columns=['actions'])['actions'].slice(ref['row'],length).to_pylist(),np.float32)
        processed=backend.policy.input_transform({
            'observation/image':read_start_image(record.image_refs['front'])[None],
            'observation/wrist_image':read_start_image(record.image_refs['wrist'])[None],
            'observation/state':np.asarray([record.proprio],np.float32),
            'prompt':[record.instruction],'actions':raw[None]})
        target=record.payload()['actions'][:length,:7]
        native=processed['actions'][0,:length,:7].numpy()
        if not np.allclose(native,target,atol=2e-5,rtol=2e-5):
            raise RuntimeError('Window normalization differs from native model input pipeline')
        result=dict(sample_id=record.sample_id,checkpoint=args.checkpoint,
            native_normalization_max_error=float(np.max(np.abs(native-target))),
            sampler=check_sampler(backend,loader,record),
            checkpointed_sampler=check_sampler(backend,loader,record,checkpoint_steps=True),
            freeze_report=backend.freeze_report(),no_optimizer_updates=True)
        if args.device.startswith('cuda'):
            result['peak_allocated_gb']=torch.cuda.max_memory_allocated()/1024**3
            result['peak_reserved_gb']=torch.cuda.max_memory_reserved()/1024**3
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(result,indent=2))
    if result is not None:
        print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
