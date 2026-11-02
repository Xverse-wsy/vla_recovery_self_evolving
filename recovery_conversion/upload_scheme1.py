"""Upload only finalized scheme1 assets to an independent existing HF repo prefix.

Credentials come from the existing private HF cache/environment, never arguments.
Each stage has a manifest and verifies remote file sizes/LFS digests before success.
"""
import argparse
import datetime
import os
from pathlib import Path
import re
import tarfile
import time
from urllib.parse import urlsplit

import convert_selected_success as core

REPO = 'wsy-w1k/rollout'
PREFIX = 'pi05_recovery_scheme1_20261102'


def phase(job, state, **fields):
    value = dict(state=state, updated_at=datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=8))).isoformat(), **fields)
    core.save(job / 'status.json', value)
    print('HF_UPLOAD_PHASE', value, flush=True)


def archive(source, destination, names):
    tmp = destination.with_suffix('.building')
    with tarfile.open(tmp, 'w:gz') as tar:
        for name in names:
            path = source / name
            assert path.exists(), path
            tar.add(path, arcname=name, filter=lambda info: None if '__pycache__' in info.name else info)
    tmp.replace(destination)


def upload_once(root):
    from huggingface_hub import HfApi
    output = root / 'dataset_subsets/pi05_recovery_scheme1_stage1_20261102'
    job = root / 'hf_upload_jobs/pi05_recovery_scheme1_20261102'
    job.mkdir(parents=True, exist_ok=True)
    phase(job, 'checking_hf_identity')
    api = HfApi()
    who = api.whoami()
    assert api.repo_info(REPO, repo_type='dataset').id == REPO
    core.save(job / 'identity.json', dict(user=who['name'], repo=REPO, prefix=PREFIX))
    for group in ['success', 'failure']:
        name = 'recovery_' + group + '20'
        if not (output / (name + '_export_complete.json')).exists():
            continue
        for folder in [name, name + '_segmented']:
            marker = job / (folder + '.uploaded.json')
            if marker.exists():
                continue
            source = output / folder
            phase(job, 'hashing_finalized_folder', folder=folder)
            files = sorted(p for p in source.rglob('*') if p.is_file() and '.cache' not in p.relative_to(source).parts)
            # Scan text without ever logging matching content or credentials.
            secret = re.compile(r'hf_[A-Za-z0-9]{25,}|sk-(?:proj-)?[A-Za-z0-9_-]{32,}')
            for p in files:
                if p.suffix in ['.json', '.jsonl', '.py', '.txt', '.md']:
                    assert not secret.search(p.read_text()), 'Possible private credential in ' + str(p.relative_to(source))
            entries = [dict(path=str(p.relative_to(source)), size=p.stat().st_size,
                sha256=core.sha(p)) for p in files]
            phase(job, 'uploading_folder', folder=folder, files=len(entries), bytes=sum(x['size'] for x in entries))
            api.upload_folder(repo_id=REPO, repo_type='dataset', folder_path=str(source),
                path_in_repo=PREFIX + '/' + folder, commit_message='Add fixed scheme1 ' + folder)
            phase(job, 'verifying_remote_folder', folder=folder)
            remote = {str(x.path): x for x in api.list_repo_tree(REPO, repo_type='dataset',
                path_in_repo=PREFIX + '/' + folder, recursive=True) if hasattr(x, 'size')}
            for entry in entries:
                f = remote[PREFIX + '/' + folder + '/' + entry['path']]
                assert f.size == entry['size']
                lfs = getattr(f, 'lfs', None)
                if lfs:
                    digest = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
                    assert digest == entry['sha256']
            core.save(marker, dict(state='verified', repo=REPO, prefix=PREFIX + '/' + folder, files=entries))
            print('HF_FOLDER_VERIFIED', folder, len(entries), flush=True)
    data_done = all((job / (folder + '.uploaded.json')).exists() for folder in
        ['recovery_success20', 'recovery_failure20', 'recovery_success20_segmented', 'recovery_failure20_segmented'])
    if not data_done or not (output / 'summary.json').exists():
        return False
    summary = core.read(output / 'summary.json')
    assert summary['state'] == 'complete'
    if not (job / 'libraries.uploaded.json').exists():
        bundle = output / 'stage1_libraries.tar.gz'
        if not bundle.exists():
            archive(output, bundle, ['banks', 'matching', 'prepared', 'algorithm.json', 'sources.json', 'summary.json'])
        scripts = output / 'rebuild_scripts.tar.gz'
        if not scripts.exists():
            archive(core.HERE, scripts, ['build_scheme1.py', 'convert_selected_success.py', 'convert_fixed_failures.py',
                'run_scheme1_build.sh', 'upload_scheme1.py', 'README_SCHEME1_STAGE1.md', 'tasks.json',
                'historical_engine', 'segmentation_engine'])
        for p in [bundle, scripts, output / 'summary.json', output / 'README.md']:
            api.upload_file(repo_id=REPO, repo_type='dataset', path_or_fileobj=str(p),
                path_in_repo=PREFIX + '/' + p.name, commit_message='Add strict Stage1 library: ' + p.name)
            remote = api.get_paths_info(REPO, [PREFIX + '/' + p.name], repo_type='dataset')[0]
            assert remote.size == p.stat().st_size
            if remote.lfs:
                digest = remote.lfs.get('sha256') if isinstance(remote.lfs, dict) else remote.lfs.sha256
                assert digest == core.sha(p)
        core.save(job / 'libraries.uploaded.json', dict(state='verified', repo=REPO, prefix=PREFIX))
    core.save(job / 'status.json', dict(state='complete', remote_verified=True,
        url='https://huggingface.co/datasets/' + REPO + '/tree/main/' + PREFIX))
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    os.environ['HF_HUB_OFFLINE'] = '0'
    job = root / 'hf_upload_jobs/pi05_recovery_scheme1_20261102'
    job.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 361):
        try:
            if upload_once(root):
                return
            status = dict(state='waiting_for_finalized_assets', attempt=attempt)
        except Exception as exc:
            # Type is sufficient for progress; do not print credentials or response bodies.
            previous = core.read(job / 'status.json') if (job / 'status.json').exists() else {}
            request = getattr(exc, 'request', None)
            status = dict(state='upload_retry_pending', attempt=attempt, error_type=type(exc).__name__,
                failed_phase=previous.get('state'), folder=previous.get('folder'),
                failed_host=urlsplit(getattr(request, 'url', '') or '').hostname)
            from huggingface_hub.utils._http import reset_sessions
            reset_sessions()
        status['updated_at'] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
        core.save(job / 'status.json', status)
        print('HF_UPLOAD_STATUS', status, flush=True)
        time.sleep(60)
    raise SystemExit('Upload retries exhausted; inspect status.json and resume explicitly')


if __name__ == '__main__':
    main()
