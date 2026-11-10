"""Download immutable, already-OSC-converted parents without logging credentials."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import pyarrow.parquet as pq
import requests


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def session():
    s = requests.Session()
    token_path = Path(os.environ.get('HF_TOKEN_PATH', Path.home() / '.cache/huggingface/token'))
    token = os.environ.get('HF_TOKEN') or (token_path.read_text().strip() if token_path.exists() else '')
    if token:
        s.headers['Authorization'] = 'Bearer ' + token
    return s


def download(s, url, target, expected_size=None):
    """Atomic output, bounded retries, resume a partial file; never log signed URLs."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if expected_size is not None and target.stat().st_size != expected_size:
            raise ValueError('Existing source has wrong size: ' + target.name)
        return
    part = target.with_suffix(target.suffix + '.part')
    for attempt in range(4):
        offset = part.stat().st_size if part.exists() else 0
        try:
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            with s.get(url, headers=headers, stream=True, timeout=(20, 90)) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f'HTTP {r.status_code}')
                if r.status_code == 206:
                    if not r.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Unexpected partial range')
                    mode = 'ab'
                else:
                    mode = 'wb'
                with part.open(mode) as f:
                    for block in r.iter_content(1024 * 1024):
                        f.write(block)
            if expected_size is not None and part.stat().st_size != expected_size:
                raise ValueError('Incomplete source download')
            part.replace(target)
            return
        except (requests.RequestException, RuntimeError, ValueError) as e:
            if attempt == 3:
                raise RuntimeError(f'Download failed ({type(e).__name__}): {target.name}') from None
            time.sleep(2 ** attempt)


def fetch_parent(record, source_root, output, expected_sha256):
    cid = record['candidate_id']
    meta = json.loads((source_root / cid / 'source.json').read_text())
    if meta['hf_path'] != record['parent_parquet'] or meta['hf_revision'] != record['hf_revision']:
        raise ValueError('Source revision/path mismatch')
    dest = output / 'parents' / cid / 'episode.parquet'
    s = session()
    base = f"https://huggingface.co/datasets/{record['hf_repo']}/resolve/{record['hf_revision']}/"
    download(s, base + record['parent_parquet'], dest, meta['source_file_bytes'])
    digest = sha256(dest)
    if digest != expected_sha256:
        raise ValueError('Parent bytes do not match pinned HF LFS hash: '+cid)
    pf = pq.ParquetFile(dest)
    expected = {'image', 'wrist_image', 'state', 'actions', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'}
    if set(pf.schema_arrow.names) != expected or pf.metadata.num_rows != meta['row_count']:
        raise ValueError('Full parent schema/length differs from numeric review source')
    original_numeric = pq.read_table(source_root / cid / 'numeric.parquet')
    if not pf.read(columns=original_numeric.column_names).equals(original_numeric):
        raise ValueError('Parent state/actions differ from reviewed version')
    provenance = dict(meta, candidate_id=cid, sha256=digest, expected_hf_lfs_sha256=expected_sha256, hf_repo=record['hf_repo'],
                      numeric_columns_only=False, already_converted_action_space='libero_osc_pose_7',
                      source_conversion_rerun=False)
    dump(dest.parent / 'source.json', provenance)
    print(f'{cid}: complete original RGB/state/action parent ({pf.metadata.num_rows} rows)', flush=True)
    return provenance


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--review', type=Path, required=True)
    ap.add_argument('--numeric-sources', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--fetch-normalization',action='store_true',help='Also acquire the audited pinned base-model norm asset')
    ap.add_argument('--parent-cache',type=Path,help='Reuse already downloaded immutable parents without downloading again')
    args = ap.parse_args()
    manifest = json.loads((args.review / 'manifest.json').read_text())
    records = {r['candidate_id']: r for r in manifest}
    trees=json.loads((args.numeric_sources.parent/'source_trees.json').read_text())
    source_hashes={e['path']:e['lfs']['oid'] for key,tree in trees.items() if key.endswith('_parent')
                   for e in tree if e.get('lfs')}
    args.output.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(args.review / 'manifest.json')
    lock = args.output / 'source_lock.json'
    identity = dict(review_manifest_sha256=source_hash, hf_revisions=sorted({r['hf_revision'] for r in manifest}),
                    parent_ids=sorted(records), format='reviewed_subtask_training_v1')
    if lock.exists() and json.loads(lock.read_text()) != identity:
        raise ValueError('Output belongs to a different selection; choose a new output')
    dump(lock, identity)
    if args.parent_cache:
        for cid in records:
            source=args.parent_cache/'parents'/cid/'episode.parquet'
            target=args.output/'parents'/cid/'episode.parquet'
            if source.exists() and not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                os.link(source,target)
        for group in sorted({r['source_group'] for r in manifest}):
            source=args.parent_cache/'source_metadata'/group
            if source.exists():shutil.copytree(source,args.output/'source_metadata'/group,dirs_exist_ok=True)
        source=args.parent_cache/'assets'
        if source.exists():shutil.copytree(source,args.output/'assets',dirs_exist_ok=True)
    s = session()
    for group in sorted({r['source_group'] for r in manifest}):
        r = next(r for r in manifest if r['source_group'] == group)
        base = f"https://huggingface.co/datasets/{r['hf_repo']}/resolve/{r['hf_revision']}/"
        group_root = r['parent_parquet'].split('/data/')[0]
        for name in ('info.json', 'tasks.jsonl', 'episodes.jsonl'):
            download(s, base + group_root + '/meta/' + name, args.output / 'source_metadata' / group / name)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_parent, r, args.numeric_sources, args.output,source_hashes[r['parent_parquet']]) for r in records.values()]
        results, errors = [], []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                errors.append(f'{type(e).__name__}: {e}')
        if errors:
            raise RuntimeError('\n'.join(errors))
    dump(args.output / 'parent_conversion_report.json', dict(state='complete', parents=sorted(results, key=lambda x:x['candidate_id']),
         method='materialize_previously_OSC_converted_original_training_parquets', new_replay=False))
    if args.fetch_normalization:
        repo='RLinf/RLinf-Pi05-LIBERO-SFT'
        rev='45ccfcc4e28634f1576ebf78cab0fbe2fd82432d'
        rel='physical-intelligence/libero/norm_stats.json'
        asset=args.output/'assets/norm_stats.json'
        download(s,f'https://huggingface.co/{repo}/resolve/{rev}/{rel}',asset)
        digest=sha256(asset)
        if digest!='dae37d79a22108af83df9189c6710a3ec8e077d65b28e34bdf1da724e5ae30f1':
            raise ValueError('Base model normalization asset differs from audited training asset')
        dump(asset.parent/'normalization_source.json',dict(hf_repo=repo,revision=rev,path=rel,sha256=digest))


if __name__ == '__main__':
    main()
