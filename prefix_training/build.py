"""Materialize the fixed ten prefixes from pinned, already-trim10 source Parquets.

Only FIVE extra rows are cut from those sources. No conversion, image rotation,
normalization, action interpolation or source overwrite is performed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from subtask_training.fetch import dump, download, session, sha256
from .spec import ACTION_ROWS, HERE, SEGMENTS, WINDOWS, records, selection, selection_sha


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def acquire(cache):
    """Download fixed HF blobs and verify their published Git/LFS hashes."""
    from huggingface_hub import HfApi
    spec = selection()
    paths = {r['source_parquet'] for r in spec['segments']}
    for r in spec['segments']:
        root = r['source_parquet'].split('/data/')[0]
        paths.update(root + '/meta/' + s for s in ('provenance.jsonl', 'tasks.jsonl', 'info.json'))
    api = HfApi()
    files = api.get_paths_info(spec['hf_repo'], sorted(paths), repo_type='dataset', revision=spec['source_revision'])
    if {f.path for f in files} != paths:
        raise ValueError('Missing source paths at fixed HF revision')

    def fetch(f):
        target = cache / f.path
        s = session()
        try:
            download(s, f"https://huggingface.co/datasets/{spec['hf_repo']}/resolve/{spec['source_revision']}/{f.path}",
                     target, f.size)
        finally:
            s.close()
        data = target.read_bytes()
        if f.lfs:
            expected = f.lfs.sha256
            actual = sha256(target)
        else:
            import hashlib
            expected = f.blob_id
            actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if actual != expected:
            raise ValueError('Source object hash mismatch: ' + f.path)
        print('Verified source: ' + f.path, flush=True)
        return f.path, sha256(target)

    with ThreadPoolExecutor(max_workers=3) as pool:
        hashes = dict(pool.map(fetch, files))
    dump(cache / 'hf_sources_verified.json', dict(revision=spec['source_revision'], files=hashes))
    return hashes


def materialize(cache, output, source_hashes):
    spec = selection()
    dest = output / 'segmented'
    if dest.exists():
        raise FileExistsError('Use a new output; preserve existing datasets')
    dest.mkdir(parents=True)
    dump(dest / 'build_state.json', dict(state='building'))
    tasks = {s: i for i, s in enumerate(sorted(r['task_instruction'] for r in spec['segments']))}
    windows, provenance, episodes, checks, hashes = [], [], [], [], {}
    offset = 0
    for eid, r in enumerate(spec['segments']):
        src = cache / r['source_parquet']
        if sha256(src) != source_hashes[r['source_parquet']]:
            raise ValueError('Source changed before materialization')
        group = cache / r['source_parquet'].split('/data/')[0]
        meta_path = group / 'meta/provenance.jsonl'
        for name in ('provenance.jsonl', 'tasks.jsonl', 'info.json'):
            rel = (group / 'meta' / name).relative_to(cache).as_posix()
            if sha256(cache / rel) != source_hashes[rel]:
                raise ValueError('Source metadata changed')
        m = next(x for x in records(meta_path) if x['episode_index'] == r['source_segment_episode_index'])
        a, b = r['retained_parent_action_rows']
        if (m['category'] != 'vla_progress_candidate' or m['owner'] != 'pi05'
                or m['instruction'] != r['task_instruction']
                or (m['suite'], m['task_id'], m['trial_id']) != (r['suite'], r['task_id'], r['trial_id'])
                or m['parent_episode_index'] != r['source_parent_episode_index']
                or [m['start'], m['end']] != r['source_parent_rows_after_trim10']
                or m['untrimmed_parent_range'] != r['untrimmed_parent_range']
                or m['discarded_tail_actions_and_start_frames'] != 10
                or m['end'] - b != 5 or m['untrimmed_parent_range'][1] - b != 15
                or m['task_completed_in_segment']):
            raise ValueError('Source trim, task or controller provenance mismatch')
        parent = pq.read_table(src)
        n = r['action_count']
        if parent.num_rows != n + 5 or b - a != n or r['retained_segment_rows'] != [0, n]:
            raise ValueError('Source was not already trimmed by ten; refusing a double trim')
        table = parent.slice(0, n)
        actions = np.asarray(table['actions'].to_pylist(), np.float32)
        state = np.asarray(table['state'].to_pylist(), np.float32)
        task_map = {t['task_index']: t['task'] for t in records(group / 'meta/tasks.jsonl')}
        if (actions.shape != (n, 7) or state.shape != (n, 8)
                or not np.isfinite(actions).all() or not np.isfinite(state).all()
                or any(task_map[t] != r['task_instruction'] for t in table['task_index'].to_pylist())):
            raise ValueError('Original action/state/task contract failed')
        np.testing.assert_array_equal(parent['frame_index'].to_numpy(), np.arange(n + 5))
        np.testing.assert_array_equal(parent['episode_index'].to_numpy(), [r['source_segment_episode_index']] * (n + 5))
        for key, values in dict(frame_index=list(range(n)), episode_index=[eid] * n,
                                index=list(range(offset, offset + n)), task_index=[tasks[r['task_instruction']]] * n).items():
            idx = table.schema.get_field_index(key)
            table = table.set_column(idx, table.schema.field(key), pa.array(values, type=table.schema.field(key).type))
        rel = f'data/chunk-000/episode_{eid:06d}.parquet'
        path = dest / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, row_group_size=64)
        restored = pq.read_table(path)
        for col in ('image', 'wrist_image', 'state', 'actions', 'timestamp'):
            if not restored[col].equals(parent[col].slice(0, n)):
                raise ValueError('Materialization changed original training columns')
        hashes[rel] = sha256(path)
        meta = dict(r, dataset_episode_index=eid, parquet=rel, controller='VLA',
                    padding_policy='full_real_only', endpoint_has_recorded_task_success=False,
                    action_rows_half_open=[a, b], horizon=10, split='train',
                    local_quality='existing_vla_progress_candidate',
                    label_basis='existing causal WCM boundary; remove original final fifteen rows',
                    source_parquet_sha256=source_hashes[r['source_parquet']])
        provenance.append(meta)
        episodes.append(dict(episode_index=eid, tasks=[r['task_instruction']], length=n))
        checks.append(dict(segment_id=r['segment_id'], source_sha256=sha256(src),
                           original_training_columns_exact=True, source_rows=parent.num_rows,
                           retained_rows=n, extra_rows_removed=5, total_original_trim=15))
        for start in range(n - 9):
            windows.append(dict(sample_id=f"{r['segment_id']}@{start:06d}", segment_id=r['segment_id'],
                dataset_episode_index=eid, parquet=rel, start=start, horizon=10,
                real_steps=10, repeated_steps=0, tail_policy='full_real_only',
                parent_action_rows_half_open=[a + start, a + start + 10],
                task_instruction=r['task_instruction'], suite=r['suite'], task_id=r['task_id'],
                controller='VLA', split='train'))
        offset += n
    if (len(provenance), offset, len(windows)) != (SEGMENTS, ACTION_ROWS, WINDOWS):
        raise ValueError('Unexpected selected prefix size')
    jsonl(dest / 'windows.jsonl', windows)
    jsonl(dest / 'meta/provenance.jsonl', provenance)
    jsonl(dest / 'meta/episodes.jsonl', episodes)
    jsonl(dest / 'meta/tasks.jsonl', [dict(task_index=i, task=s) for s, i in tasks.items()])
    shutil.copy2(HERE / 'selection.json', dest / 'meta/selection.json')
    source_info = json.loads((group / 'meta/info.json').read_text())
    source_info.update(total_episodes=10, total_frames=845, total_tasks=10, total_chunks=1,
                       total_videos=0, splits={'train': '0:10'}, chunks_size=1000,
                       data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
                       native_control_hz=20, training_horizon=10,
                       training_window_policy='must_use_windows_jsonl_full_real_only')
    dump(dest / 'meta/info.json', source_info)
    dump(dest / 'source_identity.json', dict(revision=spec['source_revision'], files=source_hashes, checks=checks))
    dump(dest / 'build_state.json', dict(state='complete', policy='vla_prefix10_tail15_v1',
        selection_sha256=selection_sha(), segments=10, tasks=10, unique_action_rows=845,
        training_windows=755, horizon=10, training_start_stride=1,
        padded_action_positions=0, total_original_trim=15, additional_source_trim=5,
        source_action_reconversion=False, model_training_performed=False, all_train=True,
        file_sha256=hashes))
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-cache', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--offline', action='store_true', help='Use an existing HF-verified source cache')
    args = ap.parse_args()
    hashes = (json.loads((args.source_cache / 'hf_sources_verified.json').read_text())['files']
              if args.offline else acquire(args.source_cache))
    print(materialize(args.source_cache, args.output, hashes))


if __name__ == '__main__':
    main()
