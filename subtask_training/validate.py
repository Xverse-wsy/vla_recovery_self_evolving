"""Independently audit all original rows and actually iterate the training loader."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import DataLoader, Subset

from .build import boundary_key
from .dataset import SubtaskDataset, decode, env_batch, normalized_actions
from .fetch import dump, sha256


def numpy_collate(samples):
    # Match OpenPI's worker-side NumPy collation; create torch tensors in the main
    # process instead of relying on platform-specific shared tensor daemons.
    return {k:([s[k] for s in samples] if isinstance(samples[0][k],str)
               else np.stack([s[k] for s in samples])) for k in samples[0]}


def validate(root, review, norm_stats=None):
    dest = root / 'segmented'
    build = json.loads((dest / 'build_state.json').read_text())
    records = [json.loads(s) for s in (dest / 'meta/provenance.jsonl').read_text().splitlines()]
    review_records = {r['segment_id']:r for r in json.loads((review/'manifest.json').read_text())}
    assert build['state'] == 'complete'
    assert build['review_manifest_sha256'] == sha256(review/'manifest.json')
    info = json.loads((dest/'meta/info.json').read_text())
    tasks = {r['task_index']: r['task'] for r in [json.loads(s) for s in (dest/'meta/tasks.jsonl').read_text().splitlines()]}
    parent_tables, seen, pixels, image_shapes = {}, defaultdict(set), 0, Counter()
    rows, eligible_rows, expected_windows, segment_stats = 0, 0, {}, []
    for r in records:
        cid, sid = r['candidate_id'], r['segment_id']
        original_review = review_records[sid]
        assert original_review['action_rows_half_open'] == r['action_rows_half_open']
        assert r['local_quality']=='manually_reviewed_progress'
        assert original_review.get('local_quality')=='manually_reviewed_progress' or original_review.get('continuous_progress_visual_review')=='pass'
        a, b = r['action_rows_half_open']
        if cid not in parent_tables:
            pp = root/'parents'/cid/'episode.parquet'
            pm = json.loads((pp.parent/'source.json').read_text())
            assert sha256(pp) == pm['sha256']
            parent_tables[cid] = pq.read_table(pp)
        parent = parent_tables[cid]
        parent_tasks={x['task_index']:x['task'] for x in
                      map(json.loads,(root/'source_metadata'/r['source_group']/'tasks.jsonl').read_text().splitlines())}
        assert all(parent_tasks[t]==r['task_instruction'] for t in parent['task_index'].slice(a,b-a).to_pylist())
        path = dest/r['parquet']
        assert sha256(path) == build['file_sha256'][r['parquet']]
        table = pq.read_table(path)
        n = b-a
        assert table.num_rows == n
        for col in ('state','actions','image','wrist_image'):
            assert table[col].equals(parent[col].slice(a,n)), (sid,col)
        assert table['frame_index'].to_pylist() == list(range(n))
        assert table['episode_index'].to_pylist() == [r['dataset_episode_index']]*n
        assert table['index'].to_pylist() == list(range(rows,rows+n))
        assert len(set(table['task_index'].to_pylist())) == 1
        assert tasks[table['task_index'][0].as_py()] == r['task_instruction']
        assert np.allclose(table['timestamp'].to_numpy(),np.arange(n)/info['fps'],atol=1e-5,rtol=1e-6)
        assert not seen[cid].intersection(range(a,b))
        seen[cid].update(range(a,b))
        mapping = [json.loads(s) for s in (dest/'provenance'/sid/'action_provenance.jsonl').read_text().splitlines()]
        assert len(mapping) == n and len({boundary_key(s) for s in mapping}) == 1
        assert [s['frame_index'] for s in mapping] == list(range(a,b))
        assert np.array_equal(table['actions'].to_pylist(),np.asarray([s['action'] for s in mapping],np.float32))
        for column in ('image','wrist_image'):
            for im in table[column].to_pylist():
                rgb = decode(im)
                assert rgb.shape == (256,256,3) and rgb.dtype == np.uint8
                image_shapes[str(rgb.shape)] += 1
                pixels += 1
        h = build['horizon']
        repeat=r['padding_policy']=='repeat_last'
        assert not repeat or r['endpoint_has_recorded_task_success'] and mapping[-1]['success_after_action']
        starts = list(range(n if repeat else max(0,n-h+1)))
        expected_windows[sid] = starts
        eligible_rows += n if starts else 0
        segment_stats.append(dict(segment_id=sid, candidate_id=cid, suite=r['suite'], task_id=r['task_id'],
                                  controller=r['controller'], action_count=n, trainable_windows=len(starts),
                                  task_instruction=r['task_instruction']))
        rows += n
    dataset = SubtaskDataset(dest)
    actual_windows = defaultdict(list)
    for w in dataset.windows:
        actual_windows[w['segment_id']].append(w['start'])
        r = next(r for r in records if r['segment_id']==w['segment_id'])
        assert w['parent_action_rows_half_open'] == [r['action_rows_half_open'][0]+w['start'], min(r['action_rows_half_open'][1],r['action_rows_half_open'][0]+w['start']+build['horizon'])]
        assert w['parquet'] == r['parquet'] and w['task_instruction'] == r['task_instruction']
    assert {k:actual_windows[k] for k in expected_windows} == expected_windows
    assert len({w['sample_id'] for w in dataset.windows}) == len(dataset)
    count, action_positions, batches = 0, 0, 0
    # Iterate every sample through the real PyTorch loader and compare its action/state/RGB origins.
    for batch in DataLoader(dataset,batch_size=17,shuffle=False,num_workers=0):
        env = env_batch(batch)
        assert env['main_images'].shape[1:] == (256,256,3)
        assert env['wrist_images'].shape[1:] == (256,256,3)
        assert env['states'].shape[1:] == (8,)
        assert batch['actions'].shape[1:] == (build['horizon'],7)
        assert np.isfinite(batch['actions'].numpy()).all()
        for j in range(len(batch['prompt'])):
            w = dataset.windows[count+j]
            r = next(r for r in records if r['segment_id']==w['segment_id'])
            parent = parent_tables[r['candidate_id']]
            start = w['parent_action_rows_half_open'][0]
            end=w['parent_action_rows_half_open'][1]
            raw=np.asarray(parent['actions'].slice(start,end-start).to_pylist(),np.float32)
            if len(raw)<build['horizon']:
                assert r['padding_policy']=='repeat_last' and r['endpoint_has_recorded_task_success']
                raw=np.concatenate([raw,np.repeat(raw[-1:],build['horizon']-len(raw),axis=0)])
            assert np.array_equal(batch['actions'][j].numpy(),raw)
            assert np.array_equal(env['states'][j],parent['state'][start].as_py())
            assert np.array_equal(env['main_images'][j],decode(parent['image'][start].as_py()))
            assert np.array_equal(env['wrist_images'][j],decode(parent['wrist_image'][start].as_py()))
            assert env['task_descriptions'][j] == w['task_instruction']
        if norm_stats:
            normalized, mask, norm_hash = normalized_actions(batch['actions'].numpy(),norm_stats)
            assert normalized.shape == (*batch['actions'].shape[:2],32)
            assert np.isfinite(normalized).all() and not normalized[...,7:].any()
            assert mask[...,:7].all() and not mask[...,7:].any()
        count += len(batch['prompt'])
        action_positions += batch['actions'].shape[0]*build['horizon']
        batches += 1
    # Cross-process serialization, image decoding and collating are also exercised.
    probes = sorted({0,len(dataset)//2,len(dataset)-1})
    multiprocess_count = sum(len(b['prompt']) for b in DataLoader(Subset(dataset,probes),batch_size=2,num_workers=2,
                            multiprocessing_context='spawn',collate_fn=numpy_collate))
    assert count == len(dataset) == build['training_windows'] and multiprocess_count == len(probes)
    assert rows == build['unique_action_rows'] and action_positions == build['supervised_action_positions']
    short = [s for s in segment_stats if s['trainable_windows']==0]
    summary = dict(state='passed', parents=len(parent_tables), segments=len(records), tasks=len(tasks),
                   trainable_segments=len(records)-len(short), retained_short_segments=short,
                   unique_action_rows=rows, unique_actions_covered_by_full_windows=eligible_rows,
                   image_bytes_exactly_match_original=True, decoded_original_rgb_images=pixels,
                   decoded_shapes=dict(image_shapes), state_and_action_max_error=0,
                   independently_checked_windows=count, training_loader_batches=batches,
                   supervised_action_positions=action_positions,padded_positions=build['padded_action_positions'],
                   repeated_positions_only_at_final_task_success=True,intermediate_padded_positions=0,
                   dense_stride=1,horizon=build['horizon'],crossed_boundaries=0,
                   multiprocessing_loader_samples=multiprocess_count,
                   dataset_fps=info['fps'],native_control_hz=info['native_control_hz'],resampled=False,
                   model_training_performed=False,native_model_forward_tested=False,
                   normalization_asset_checked=bool(norm_stats))
    summary['instructions_exactly_match_parent_task_metadata']=True
    if norm_stats:
        summary['normalization_sha256']=norm_hash
    dump(dest/'validation.json',summary)
    dump(dest/'segment_statistics.json',segment_stats)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--review',required=True,type=Path)
    ap.add_argument('--norm-stats',type=Path)
    args=ap.parse_args()
    validate(args.output,args.review,args.norm_stats)


if __name__=='__main__':
    main()
