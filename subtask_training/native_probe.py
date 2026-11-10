"""CPU-only native RLinf/OpenPI transform parity probe. Never loads model weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['OMP_NUM_THREADS'] = '2'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples',required=True,type=Path)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--norm-stats',required=True,type=Path)
    ap.add_argument('--reference-dataset',required=True)
    args=ap.parse_args()
    import numpy as np
    import torch
    import jax
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.env_io import EnvIO
    from rlinf.models.embodiment.openpi.transforms.pipeline import build_openpi_transforms
    from openpi.training.data_loader import transform_dataset, _collate_fn
    from openpi.models.model import Observation
    torch.set_num_threads(2)
    kw=dict(norm_stats_path=str(args.norm_stats),extra_delta_transform=False)
    config=get_openpi_config('pi05_libero',model_path=args.checkpoint,batch_size=1,
                            repo_id=args.reference_dataset,data_kwargs=kw)
    data_config=config.data.create(config.assets_dirs,config.model)
    inputs,outputs=build_openpi_transforms(args.checkpoint,'pi05_libero',data_kwargs=kw)
    env=EnvIO()
    env.config_name='pi05_libero'
    env.state_indices=None
    env.device=torch.device('cpu')
    env.action_chunk=10
    env.setup_transforms(inputs,outputs)
    def arr(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    def error(a,b):
        a,b=arr(a),arr(b)
        assert a.shape==b.shape,(a.shape,b.shape)
        return float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
    reports=[]
    with np.load(args.samples,allow_pickle=False) as samples:
        for i,prompt in enumerate(samples['prompt']):
            sample=dict(image=samples['image'][i],wrist_image=samples['wrist_image'][i],
                        state=samples['state'][i],actions=samples['actions'][i],prompt=str(prompt))
            transformed=transform_dataset([sample],data_config)[0]
            official=Observation.from_dict(jax.tree.map(torch.as_tensor,_collate_fn([transformed])))
            via_env=env.env_obs_to_observation(dict(states=sample['state'][None],main_images=sample['image'][None],
                wrist_images=sample['wrist_image'][None],task_descriptions=[str(prompt)]))
            image_errors={}
            for key,v in official.images.items():
                a,b=arr(v),arr(via_env.images[key])
                if a.ndim==4 and a.shape[1]==3:a=np.transpose(a,(0,2,3,1))
                if b.ndim==4 and b.shape[1]==3:b=np.transpose(b,(0,2,3,1))
                image_errors[key]=error(a,b)
            r=dict(sample_id=str(samples['sample_id'][i]),
                   normalized_action_error=error(transformed['actions'],samples['normalized_actions'][i]),
                   state_error=error(official.state,via_env.state),
                   prompt_token_error=error(official.tokenized_prompt,via_env.tokenized_prompt),
                   prompt_mask_equal=np.array_equal(arr(official.tokenized_prompt_mask),arr(via_env.tokenized_prompt_mask)),
                   image_errors=image_errors,
                   image_masks_equal=all(np.array_equal(arr(v),arr(via_env.image_masks[k])) for k,v in official.image_masks.items()))
            assert r['normalized_action_error']<2e-6 and r['state_error']<2e-6
            assert r['prompt_token_error']==0 and r['prompt_mask_equal'] and r['image_masks_equal']
            assert all(v==0 for v in image_errors.values())
            reports.append(r)
    print('NATIVE_SUBTASK_PROBE '+json.dumps(dict(state='passed',samples=reports,
        normalization_sha256=hashlib.sha256(args.norm_stats.read_bytes()).hexdigest(),
        native_training_transforms_checked=True,model_weights_loaded=False,gpu_used=False,training_performed=False)))


if __name__=='__main__':
    main()
