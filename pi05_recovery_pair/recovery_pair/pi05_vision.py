"""Frozen native RLinf SigLIP features from an existing pi05 safetensors asset."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from .features import read_start_image,sha256
from .schema import stable_id


def pooled_unit_features(tokens):
    if tokens.ndim!=3 or not torch.isfinite(tokens).all():
        raise ValueError('Expected finite [batch,patches,features] image tokens')
    pooled=tokens.float().mean(dim=1)
    if (pooled.norm(dim=-1)<=1e-12).any():
        raise ValueError('Zero pooled image feature')
    return torch.nn.functional.normalize(pooled,dim=-1)


class Pi05Vision:
    def __init__(self,checkpoint,device='cpu',max_cuda_memory_gb=8.):
        import rlinf
        from safetensors import safe_open
        from .source_identity import verify_installed_rlinf
        from .openpi_backend import RLINF_COMMIT
        from rlinf.models.embodiment.openpi.modules.siglip import SigLIPViT
        from rlinf.utils.ckpt_convertor.openpi.openpi_pytorch_to_openpi import old_to_new_state_dict
        from openpi.transforms import ResizeImages
        self.source_identity=verify_installed_rlinf(rlinf,RLINF_COMMIT)
        self.device=torch.device(device)
        if self.device.type=='cuda':
            torch.cuda.set_device(self.device)
            free,total=torch.cuda.mem_get_info()
            limit=int(max_cuda_memory_gb*1024**3)
            if limit<=0 or limit+4*1024**3>free:
                raise RuntimeError('Need positive allocator cap and at least 4GiB free margin')
            torch.cuda.set_per_process_memory_fraction(limit/total)
        path=Path(checkpoint).expanduser().resolve()
        if path.is_dir():
            path=path/'model.safetensors'
        self.checkpoint=str(path)
        with safe_open(path,framework='pt',device='cpu') as f:
            keys=[k for k in f.keys() if k.startswith('img.') or
                  k.startswith('paligemma_with_expert.paligemma.model.vision_tower.') or
                  k.startswith('paligemma_with_expert.paligemma.model.multi_modal_projector.')]
            state={k:f.get_tensor(k) for k in keys}
        if not state:
            raise ValueError('No native or official OpenPI image weights found')
        digest=hashlib.sha256()
        for k,t in sorted(state.items()):
            digest.update(k.encode()+str(t.shape).encode()+str(t.dtype).encode())
            digest.update(t.contiguous().view(torch.uint8).numpy().tobytes())
        self.encoder_sha=digest.hexdigest()
        converted=old_to_new_state_dict(state) if any(k.startswith('paligemma_with_expert.') for k in state) else state
        weights={k.removeprefix('img.'):v for k,v in converted.items() if k.startswith('img.')}
        width=weights['head.weight'].shape[0]
        self.encoder=SigLIPViT(variant='So400m/14',pool_type='none',num_classes=width,
            use_gradient_checkpointing=False,dtype_mm='bfloat16' if self.device.type=='cuda' else 'float32')
        self.encoder.load_state_dict(weights,strict=True)
        del weights,converted,state
        self.encoder.requires_grad_(False).eval().to(self.device)
        if self.device.type=='cuda':
            # Match native pi05 mixed precision: stem + position stay fp32.
            self.encoder.encoder.to(torch.bfloat16)
            self.encoder.head.to(torch.bfloat16)
        self.resize=ResizeImages(224,224)
        self.preprocess_spec=dict(resize='openpi.transforms.ResizeImages(224,224)',
            range='uint8/255*2-1',layout='BHWC',random_augmentation=False,rotation=False,
            pooling='mean_projected_patch_tokens_then_L2',views=['front','wrist'],
            source_identity=self.source_identity['source_sha256'],
            converter_sha256=sha256(__import__('inspect').getsourcefile(old_to_new_state_dict)))
        self.preprocess_id=stable_id(self.preprocess_spec)

    def preprocess(self,rgb):
        resized=self.resize({'image':{'base_0_rgb':rgb}})['image']['base_0_rgb']
        return torch.as_tensor(np.asarray(resized).copy(),device=self.device).float()/255*2-1

    def check_preprocess(self,rgb):
        from rlinf.models.embodiment.openpi.modules.model import Observation,preprocess_observation
        resized=self.resize({'image':{'base_0_rgb':rgb}})['image']['base_0_rgb']
        original=Observation.from_dict(dict(image={'base_0_rgb':torch.as_tensor(resized.copy(),device=self.device)[None]},
            image_mask={'base_0_rgb':torch.ones(1,device=self.device,dtype=torch.bool)},
            state=torch.zeros(1,8,device=self.device)))
        native=preprocess_observation(original,train=False,image_keys=('base_0_rgb',)).images['base_0_rgb'][0]
        error=float((native-self.preprocess(rgb)).abs().max())
        if error!=0:
            raise RuntimeError(f'Preprocessing differs from native pipeline: {error}')
        return error

    def extract(self,records,cache_dir,batch_size=2):
        if batch_size<1:
            raise ValueError('batch_size must be positive')
        cache=Path(cache_dir)
        cache.mkdir(parents=True,exist_ok=True)
        pending=[]
        feature_map={}
        preprocessing_error=None
        count=0
        def flush():
            nonlocal count
            if not pending:
                return
            with torch.inference_mode():
                tokens,_=self.encoder(torch.stack([x[1] for x in pending]))
                features=pooled_unit_features(tokens).cpu().numpy()
            for (key,_),feature in zip(pending,features):
                np.save(cache/(key+'.npy'),feature)
                feature_map[key]=feature
            count+=len(pending)
            pending.clear()
        jobs=[]
        ordered=sorted(records,key=lambda r:(str(r.image_refs['front']),r.sample_id))
        for r in ordered:
            refs={}
            for view in ('front','wrist'):
                rgb=read_start_image(r.image_refs[view])
                if preprocessing_error is None:
                    preprocessing_error=self.check_preprocess(rgb)
                rgb_sha=hashlib.sha256(rgb.tobytes()+str(rgb.shape).encode()).hexdigest()
                key=stable_id(self.encoder_sha,self.preprocess_id,rgb_sha)
                refs[view]=key
                if key not in feature_map and not any(k==key for k,_ in pending):
                    dest=cache/(key+'.npy')
                    if dest.exists():
                        feature_map[key]=np.load(dest,allow_pickle=False)
                    else:
                        pending.append((key,self.preprocess(rgb)))
                if len(pending)>=batch_size:
                    flush()
            jobs.append((r,refs))
        flush()
        for r,refs in jobs:
            r.visual_features=dict(encoder_sha256=self.encoder_sha,calibration_id=self.preprocess_id,
                views={v:feature_map[k].tolist() for v,k in refs.items()})
        return dict(image_embeddings_computed=count,unique_images=len(feature_map),
            native_preprocessing_max_abs_error=preprocessing_error,
            vision_parameters=sum(p.numel() for p in self.encoder.parameters()),
            trainable_parameters=sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
            encoder_sha256=self.encoder_sha,preprocessing=self.preprocess_spec,
            source_identity=self.source_identity,checkpoint=self.checkpoint,
            cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3 if self.device.type=='cuda' else None)
