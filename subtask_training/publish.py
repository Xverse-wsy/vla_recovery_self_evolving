"""Publish only the reviewed22 payload, then verify every remote Git/LFS object hash."""
import argparse
import hashlib
import json
from pathlib import Path

from .fetch import dump

REPO='wsy-w1k/rollout'
PREFIX='pi05_subtasks22_20261110'


def payload(root):
    files=[]
    for directory in ('segmented','assets'):
        files.extend(p for p in (root/directory).rglob('*') if p.is_file())
    files.extend(root/p for p in ['README.md','cpu_gradient_sampler.json',
                 'preflight_default/preflight.json','preflight_16rank/preflight.json'])
    return sorted(files)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--verify-revision',help='Verify an existing commit instead of uploading')
    args=ap.parse_args()
    root=args.output.resolve()
    report=json.loads((root/'segmented/build_state.json').read_text())
    validation=json.loads((root/'segmented/validation.json').read_text())
    preflight=json.loads((root/'preflight_default/preflight.json').read_text())
    if report['segments']!=22 or report['training_windows']!=1194 or validation['state']!='passed' or preflight['state']!='passed':
        raise ValueError('Only the accepted and validated original22 dataset can be published')
    from huggingface_hub import HfApi,RepoFile
    api=HfApi()
    files=payload(root)
    if not args.verify_revision:
        commit=api.upload_folder(repo_id=REPO,repo_type='dataset',folder_path=root,path_in_repo=PREFIX,
            allow_patterns=['segmented/**','assets/**','README.md','cpu_gradient_sampler.json',
                            'preflight_default/preflight.json','preflight_16rank/preflight.json'],
            commit_message='Add original22 reviewed subtask training clips and CPU acceptance reports')
        revision=commit.oid
        dump(root/'hf_upload_commit.json',dict(repo=REPO,prefix=PREFIX,revision=revision))
    else:
        revision=args.verify_revision
    remote={f.path:f for f in api.list_repo_tree(REPO,repo_type='dataset',revision=revision,path_in_repo=PREFIX,recursive=True) if isinstance(f,RepoFile)}
    inventory=[]
    for path in files:
        rel=path.relative_to(root).as_posix()
        f=remote[PREFIX+'/'+rel]
        data=path.read_bytes()
        digest=hashlib.sha256(data).hexdigest()
        if f.size!=len(data):raise ValueError('Remote size mismatch: '+rel)
        if f.lfs:
            if f.lfs.sha256!=digest:raise ValueError('Remote LFS hash mismatch: '+rel)
        else:
            git_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            if git_sha!=f.blob_id:raise ValueError('Remote Git blob mismatch: '+rel)
        inventory.append(dict(path=rel,bytes=len(data),sha256=digest))
    unexpected=set(remote)-{PREFIX+'/'+p.relative_to(root).as_posix() for p in files}
    if unexpected:raise ValueError('Unexpected remote payload files: '+str(sorted(unexpected)))
    result=dict(state='verified',repo=REPO,prefix=PREFIX,revision=revision,
                files=len(files),bytes=sum(p.stat().st_size for p in files),
                parquet_files=sum(p.suffix=='.parquet' for p in files),segments=22,
                excluded_added46=True,all_remote_object_hashes_match=True,inventory=inventory)
    dump(root/'hf_verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='inventory'},indent=2))


if __name__=='__main__':
    main()
