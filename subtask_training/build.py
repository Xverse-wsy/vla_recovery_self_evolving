"""Slice original training Parquets by reviewed intervals, then index real H-step windows."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .fetch import dump, sha256


def jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))


def boundary_key(row):
    return row['owner'], row.get('source_turn'), row.get('source_program')


def build(review, sources, root, horizon, tail_policy='full_real_only'):
    if horizon <= 0:
        raise ValueError('horizon must be positive')
    if tail_policy not in ('full_real_only','success_endpoint_repeat'):
        raise ValueError('Unknown tail policy')
    lock = json.loads((root / 'source_lock.json').read_text())
    if lock['review_manifest_sha256'] != sha256(review / 'manifest.json'):
        raise ValueError('Reviewed boundaries changed since download')
    if not (root / 'parent_conversion_report.json').exists():
        raise ValueError('Materialize and verify all parents first')
    dest = root / 'segmented'
    if dest.exists():
        raise FileExistsError('Use a new output; segmented data is immutable once built')
    dest.mkdir()
    dump(dest / 'build_state.json', dict(state='building'))
    manifest = json.loads((review / 'manifest.json').read_text())
    instructions = sorted({r['task_instruction'] for r in manifest})
    tasks = {instruction: i for i, instruction in enumerate(instructions)}
    seen, tables, mappings = defaultdict(set), {}, {}
    provenance, episodes, windows, skipped, hashes = [], [], [], [], {}
    dataset_index = 0
    for eid, r in enumerate(manifest):
        cid, sid = r['candidate_id'], r['segment_id']
        if not (r.get('review_is_manual') and r.get('continuous_progress_visual_review')=='pass'
                or r.get('manual_review') and r.get('local_quality')=='manually_reviewed_progress'):
            raise ValueError('Only explicitly reviewed continuous-progress ranges may become positives')
        if cid not in tables:
            parent_path = root / 'parents' / cid / 'episode.parquet'
            tables[cid] = pq.read_table(parent_path)
            mappings[cid] = [json.loads(s) for s in (sources / cid / 'action_provenance.jsonl').read_text().splitlines() if s.strip()]
        parent, mapping = tables[cid], mappings[cid]
        a, b = r['action_rows_half_open']
        if not 0 <= a < b <= parent.num_rows or len(mapping) != parent.num_rows:
            raise ValueError('Invalid original action row range: ' + sid)
        selected = mapping[a:b]
        terminal=bool(r.get('endpoint_has_recorded_task_success'))
        if terminal != bool(selected[-1]['success_after_action']):
            raise ValueError('Reviewed task-completion label disagrees with converted replay')
        if any(x['success_after_action'] for x in selected[:-1]):
            raise ValueError('Review includes actions after task first completed')
        repeat=terminal and tail_policy=='success_endpoint_repeat'
        keys = {boundary_key(x) for x in selected}
        if len(keys) != 1:
            raise ValueError('Reviewed interval crosses controller/program boundary: ' + sid)
        if any(x['action_space']!='osc_pose' or x['owner'] not in ('pi05','capx','capx_converted') for x in selected):
            raise ValueError('Expected VLA/CAP converted OSC actions, not joint targets')
        run=r['controller_runs'][0]
        actual_owner,actual_turn,actual_program=next(iter(keys))
        if (run['owner'],run['source_turn'])!=(actual_owner,actual_turn):
            raise ValueError('Reviewed controller/code identity differs from parent provenance')
        if (Path(actual_program).name if actual_program else None)!=(Path(run['program']).name if run['program'] else None):
            raise ValueError('Reviewed program differs from original code')
        if seen[cid].intersection(range(a, b)):
            raise ValueError('Overlapping original action rows: ' + sid)
        seen[cid].update(range(a, b))
        if [x['frame_index'] for x in selected] != list(range(a, b)):
            raise ValueError('Conversion row mapping mismatch')
        table = parent.slice(a, b - a)
        n = table.num_rows
        original_tasks={x['task_index']:x['task'] for x in
                        map(json.loads,(root/'source_metadata'/r['source_group']/'tasks.jsonl').read_text().splitlines())}
        if any(original_tasks[t]!=r['task_instruction'] for t in table['task_index'].to_pylist()):
            raise ValueError('Review text differs from original parent-trajectory instruction')
        state = np.asarray(table['state'].to_pylist(), np.float32)
        actions = np.asarray(table['actions'].to_pylist(), np.float32)
        if state.shape != (n, 8) or actions.shape != (n, 7) or not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError('State/action dimensions or finite-value contract failed')
        if not np.array_equal(actions, np.asarray([x['action'] for x in selected], np.float32)):
            raise ValueError('Converted action provenance differs from Parquet')
        for col, values in dict(frame_index=list(range(n)), episode_index=[eid] * n,
                                index=list(range(dataset_index, dataset_index + n)),
                                task_index=[tasks[r['task_instruction']]] * n,
                                timestamp=[i / r['dataset_timestamp_fps'] for i in range(n)]).items():
            idx = table.schema.get_field_index(col)
            table = table.set_column(idx, table.schema.field(col), pa.array(values, type=table.schema.field(col).type))
        rel = f'data/chunk-{eid // 1000:03d}/episode_{eid:06d}.parquet'
        path = dest / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, row_group_size=64)
        hashes[rel] = sha256(path)
        owner, turn, program = next(iter(keys))
        record = dict(r, dataset_episode_index=eid, parquet=rel, split='train',
                      controller='VLA' if owner == 'pi05' else 'code', original_controller=owner,
                      source_turn=turn, source_program=program,
                      original_indices_rebased=True, original_timestamp_start=float(parent['timestamp'][a].as_py()),
                      padding_policy='repeat_last' if repeat else 'full_real_only', horizon=horizon,
                      trainable_window_count=n if repeat else max(0, n - horizon + 1),
                      local_quality='manually_reviewed_progress',
                      label_basis='existing_manual_continuous_progress_review',
                      # Dataset is independently usable through its window loader; no old Stage1 asset was changed.
                      directly_loadable_by_stage1=False)
        provenance.append(record)
        episodes.append(dict(episode_index=eid, tasks=[r['task_instruction']], length=n))
        provdir = dest / 'provenance' / sid
        jsonl(provdir / 'action_provenance.jsonl', selected)
        dump(provdir / 'segment.json', record)
        if program:
            original = sources / cid / 'programs' / Path(program).name
            if original.exists():
                shutil.copy2(original, provdir / original.name)
        if n < horizon and not repeat:
            skipped.append(dict(segment_id=sid, action_count=n, reason='fewer_than_H_real_actions'))
        for start in range(n if repeat else max(0, n - horizon + 1)):
            real_steps=min(horizon,n-start)
            windows.append(dict(sample_id=f'{sid}@{a + start:06d}', segment_id=sid, candidate_id=cid,
                                dataset_episode_index=eid, parquet=rel, start=start, horizon=horizon,
                                parent_action_rows_half_open=[a + start, a + start + real_steps],
                                real_steps=real_steps,repeated_steps=horizon-real_steps,
                                tail_policy='repeat_last' if repeat else 'full_real_only',
                                task_instruction=r['task_instruction'], suite=r['suite'], task_id=r['task_id'],
                                controller=record['controller'], source_turn=turn, split='train'))
        dataset_index += n
    (dest / 'meta').mkdir(exist_ok=True)
    shutil.copy2(review / 'manifest.json', dest / 'meta' / 'review_manifest.json')
    jsonl(dest / 'meta' / 'provenance.jsonl', provenance)
    jsonl(dest / 'meta' / 'episodes.jsonl', episodes)
    jsonl(dest / 'meta' / 'tasks.jsonl', [dict(task_index=i, task=s) for i, s in enumerate(instructions)])
    jsonl(dest / 'windows.jsonl', windows)
    dump(dest / 'skipped_short_segments.json', skipped)
    # Same embedded-image schema as the source; metadata indicates that full-window selection is mandatory.
    src_info = json.loads((root / 'source_metadata' / provenance[0]['source_group'] / 'info.json').read_text())
    src_info.update(total_episodes=len(episodes), total_frames=dataset_index, total_tasks=len(tasks), total_chunks=1,
                    total_videos=0, splits={'train': f'0:{len(episodes)}'},
                    data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
                    training_window_policy='must_use_windows_jsonl_source_aware_tails',
                    training_horizon=horizon, native_control_hz=20, review_manifest_sha256=lock['review_manifest_sha256'])
    dump(dest / 'meta' / 'info.json', src_info)
    summary = dict(state='complete', review_manifest_sha256=lock['review_manifest_sha256'], parents=len(tables),
                   tasks=len(tasks), segments=len(episodes), trainable_segments=len(episodes)-len(skipped),
                   short_segments=len(skipped), unique_action_rows=dataset_index, training_windows=len(windows),
                   supervised_action_positions=len(windows)*horizon, horizon=horizon,
                   training_start_stride=1, padded_action_positions=sum(w['repeated_steps'] for w in windows),
                   tail_policy=tail_policy,terminal_segments=sum(r['padding_policy']=='repeat_last' for r in provenance),all_train=True,
                   controller_windows=dict(Counter(w['controller'] for w in windows)),
                   file_sha256=hashes, source_action_reconversion=False, simulation_performed=False,
                   original_stage1_modified=False, model_training_performed=False)
    dump(dest / 'build_state.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k != 'file_sha256'}, ensure_ascii=False, indent=2))
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--review', required=True, type=Path)
    ap.add_argument('--numeric-sources', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path, help='Same root as fetch output')
    ap.add_argument('--horizon', type=int, default=10)
    ap.add_argument('--tail-policy',choices=['full_real_only','success_endpoint_repeat'],default='full_real_only')
    args = ap.parse_args()
    build(args.review, args.numeric_sources, args.output, args.horizon,args.tail_policy)


if __name__ == '__main__':
    main()
