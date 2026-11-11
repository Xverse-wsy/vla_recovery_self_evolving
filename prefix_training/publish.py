"""Publish only accepted prefix10 data; verify every resulting HF object hash."""
import argparse
import hashlib
import json
from pathlib import Path

from subtask_training.fetch import dump
from .spec import PREFIX


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--verify-revision')
    a = ap.parse_args()
    root = a.output.resolve()
    build = json.loads((root / 'segmented/build_state.json').read_text())
    validation = json.loads((root / 'segmented/validation.json').read_text())
    preflight = json.loads((root / 'preflight_default/preflight.json').read_text())
    if ((build['segments'], build['unique_action_rows'], build['training_windows']) != (10, 845, 755)
            or validation['state'] != 'passed' or preflight['state'] != 'passed'
            or preflight['training_windows'] != 10619):
        raise ValueError('Only the accepted ten physically-trim15 prefixes can be published')
    patterns = ['segmented/**', 'assets/**', 'review_videos/**', 'README.md',
                'preflight_default/preflight.json', 'preflight_16rank/preflight.json']
    files = [p for folder in ('segmented', 'assets', 'review_videos')
             for p in (root / folder).rglob('*') if p.is_file()]
    files += [root / p for p in ('README.md', 'preflight_default/preflight.json', 'preflight_16rank/preflight.json')]
    from huggingface_hub import HfApi, RepoFile
    api = HfApi()
    if a.verify_revision:
        revision = a.verify_revision
    else:
        commit = api.upload_folder(repo_id='wsy-w1k/rollout', repo_type='dataset',
            folder_path=root, path_in_repo=PREFIX, allow_patterns=patterns,
            commit_message='Add ten short VLA prefixes with physical tail15 trim and A60 training acceptance')
        revision = commit.oid
        dump(root / 'hf_upload_commit.json', dict(repo='wsy-w1k/rollout', prefix=PREFIX, revision=revision))
    remote = {f.path: f for f in api.list_repo_tree('wsy-w1k/rollout', repo_type='dataset',
        revision=revision, path_in_repo=PREFIX, recursive=True) if isinstance(f, RepoFile)}
    if set(remote) != {PREFIX + '/' + p.relative_to(root).as_posix() for p in files}:
        raise ValueError('Missing or extra remote payload')
    inventory = []
    for p in files:
        data = p.read_bytes()
        rel = p.relative_to(root).as_posix()
        f = remote[PREFIX + '/' + rel]
        digest = hashlib.sha256(data).hexdigest()
        if f.size != len(data):
            raise ValueError('Remote size mismatch: ' + rel)
        if f.lfs:
            okay = f.lfs.sha256 == digest
        else:
            okay = f.blob_id == hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if not okay:
            raise ValueError('Remote Git/LFS hash mismatch: ' + rel)
        inventory.append(dict(path=rel, bytes=len(data), sha256=digest))
    result = dict(state='verified', repo='wsy-w1k/rollout', prefix=PREFIX, revision=revision,
        files=len(files), bytes=sum(p.stat().st_size for p in files),
        parquet_files=sum(p.suffix == '.parquet' for p in files), segments=10,
        unique_action_rows=845, prefix_windows=755, full_training_windows=10619,
        all_remote_object_hashes_match=True, inventory=inventory)
    dump(root / 'hf_verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'inventory'}, indent=2))


if __name__ == '__main__':
    main()
