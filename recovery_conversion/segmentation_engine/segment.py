"""Split existing converted recoveries; never replay, replan, or train.

All ranges are zero-based action rows, end exclusive. Read README before use.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path('/home/wenchaoxu/wsy/vla-recovery-posttrain-loop')
DETECTOR_REL = Path('wcm_pi05_monitor/outputs/hybrid_independent_gpu1_20261019_172047/runtime/trend.py')
SOURCES = {
    'recovery_success20': 'dataset_subsets/pi05_long_recovery20_20261028/recovery_success20_lerobot',
    'recovery_failure20': 'dataset_subsets/pi05_recovery_failure20_20261029/recovery_failure20_lerobot',
}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def lines(path, values):
    Path(path).write_text(''.join(json.dumps(v, ensure_ascii=False, allow_nan=False) + '\n' for v in values))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def snapshot(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def warning_boundary(handoff, scores):
    """Return row of FIRST action producing the confirmed abnormal observation.

    Score observation k is after action k, i.e. action row k-1. This is a
    detector-condition boundary, not a ground-truth physical failure onset.
    """
    end = int(handoff['step'])
    prefix = [s for s in scores if int(s['observation_index']) <= end]
    assert [int(s['observation_index']) for s in prefix] == list(range(end + 1))
    last = prefix[-1]
    assert last['reason'] == handoff['alarm']['reason']
    assert last['diagnostic'] == handoff['alarm']['diagnostic']
    reason = last['reason']
    info = dict(reason=reason, handoff_step=end, weak_label=True,
                physical_failure_onset_verified=False, alarm_diagnostic=last['diagnostic'])
    if reason == 'sudden_drop':
        start_obs = end
    elif reason in ('trend_flattening', 'sustained_decline'):
        field = 'flattening_actions' if reason == 'trend_flattening' else 'decline_actions'
        count = int(last['diagnostic'][field])
        assert count >= int(last['diagnostic']['confirmation_actions'])
        start_obs = end - count + 1
        assert start_obs >= 1
        for j in range(start_obs, end + 1):
            assert int(prefix[j]['diagnostic'][field]) == j - start_obs + 1
        assert int(prefix[start_obs - 1]['diagnostic'][field]) == 0
    else:
        return None, dict(info, boundary_known=False, method='retain_unsplit_vla_prefix')
    return start_obs - 1, dict(info, boundary_known=True,
                               first_abnormal_observation=start_obs,
                               warning_action_rows=[start_obs - 1, end])


def segment_ranges(mapping, prefix, boundary):
    assert 0 <= prefix <= len(mapping)
    spans = []
    if boundary is None:
        if prefix:
            spans.append(dict(start=0, end=prefix, owner='pi05', category='vla_boundary_uncertain', turn=None))
    else:
        assert 0 <= boundary < prefix
        if boundary:
            spans.append(dict(start=0, end=boundary, owner='pi05', category='vla_progress_candidate', turn=None))
        spans.append(dict(start=boundary, end=prefix, owner='pi05', category='vla_wcm_warning', turn=None))
    assert all(m['owner'] == 'pi05' and m['source_turn'] is None for m in mapping[:prefix])
    seen = set()
    i = prefix
    while i < len(mapping):
        m = mapping[i]
        assert m['owner'] in ('capx', 'capx_converted') and m['source_turn'] is not None
        turn = int(m['source_turn'])
        assert turn not in seen, 'Program reappeared non-contiguously'
        seen.add(turn)
        j = i + 1
        while j < len(mapping) and mapping[j]['source_turn'] == turn:
            assert mapping[j]['owner'] in ('capx', 'capx_converted')
            j += 1
        completed = any(x['success_after_action'] for x in mapping[i:j])
        spans.append(dict(start=i, end=j, owner='capx', category='capx_program', turn=turn,
                          completion_label='完成任务' if completed else '没有完成任务'))
        i = j
    assert spans and spans[0]['start'] == 0 and spans[-1]['end'] == len(mapping)
    assert all(a['end'] == b['start'] for a, b in zip(spans, spans[1:]))
    for i, m in enumerate(mapping):
        assert m['frame_index'] == i and m['observation_index'] == i and m['next_observation_index'] == i + 1
        if m['inserted_compensation_action']:
            assert i and m['source_action_step'] == mapping[i - 1]['source_action_step']
            assert m['source_turn'] == mapping[i - 1]['source_turn'] and m['owner'] == mapping[i - 1]['owner']
            assert not any(s['start'] == i for s in spans)
    return spans


def statistics(payload, names):
    """Exact pixel moments using Pillow histograms; existing LeRobot convention."""
    import io
    import numpy as np
    from PIL import Image, ImageStat
    out = {}
    for name in names:
        if name in ('image', 'wrist_image'):
            sums, squares = np.zeros(3), np.zeros(3)
            low, high = np.full(3, np.inf), np.full(3, -np.inf)
            pixels = 0
            for row in payload:
                with Image.open(io.BytesIO(row[name]['bytes'])) as im:
                    assert im.size == (256, 256) and im.mode == 'RGB'
                    stat = ImageStat.Stat(im)
                    sums += np.asarray(stat.sum) / 255
                    squares += np.asarray(stat.sum2) / (255 * 255)
                    low = np.minimum(low, np.asarray(stat.extrema)[:, 0] / 255)
                    high = np.maximum(high, np.asarray(stat.extrema)[:, 1] / 255)
                    pixels += im.width * im.height
            mean = sums / pixels
            std = np.sqrt(np.maximum(0, squares / pixels - mean ** 2))
            vals = [v.reshape(3, 1, 1).tolist() for v in [low, high, mean, std]]
        else:
            a = np.asarray([r[name] for r in payload], dtype=np.float64)
            vals = [np.atleast_1d(v).tolist() for v in [a.min(0), a.max(0), a.mean(0), a.std(0)]]
        out[name] = dict(zip(['min', 'max', 'mean', 'std'], vals), count=[len(payload)])
    return out


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve(source, value):
    p = Path(value)
    return p if p.is_absolute() else source / p


def detector_path(root):
    """Use the recorded run's frozen detector, independently of copied log paths."""
    path = Path(root) / DETECTOR_REL
    if not path.is_file():
        raise FileNotFoundError(f'Frozen experiment detector is required: {path}')
    return path


def export(source, dest, limit, root):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    assert source != dest and not dest.exists()
    temp = dest.with_name(dest.name + '.building')
    assert not temp.exists(), f'Inspect incomplete export: {temp}'
    (temp / 'meta').mkdir(parents=True)
    frozen = detector_path(root)
    detector_module = load_module('frozen_trend', frozen)
    shutil.copy2(frozen, temp / 'frozen_trend.py')
    info = read(source / 'meta/info.json')
    parents = rows(source / 'meta/provenance.jsonl')
    assert len(parents) == 20
    if limit:
        parents = parents[:limit]
    episodes, epstats, provenance, parent_reports, empty_programs = [], [], [], [], []
    unchanged = snapshot(source)
    offset, input_frames = 0, 0
    for parent in parents:
        pid = parent['exported_episode_index']
        relative = info['data_path'].format(episode_chunk=pid // info['chunks_size'], episode_index=pid)
        table = pq.read_table(source / relative)
        n = len(table)
        rec = parent['source_record']
        original = Path(rec['source_episode'])
        original_report = read(original / 'report.json')
        report_path = resolve(source, parent.get('conversion_report', parent.get('replay_report')))
        report = read(report_path)
        mapping_path = resolve(source, parent['action_provenance'])
        mapping = rows(mapping_path)
        prefix = int(report['prefix_steps'])
        expected_success = source.name.startswith('recovery_success')
        # Some old converter input records contain a placeholder success=True.
        # Use actual original report and converted transition outcomes instead.
        assert bool(original_report['success']) == expected_success == bool(report['success'])
        assert report['state'] == 'complete' and n == len(mapping) == report['executed_steps']
        assert prefix == rec['handoff_step'] and report['prefix_state_error_max'] == 0
        successes = [i for i, m in enumerate(mapping) if m['success_after_action']]
        assert successes == ([n - 1] if expected_success else [])
        actions = np.array(table['actions'].to_pylist(), np.float32)
        np.testing.assert_array_equal(actions, np.asarray([m['action'] for m in mapping], np.float32))
        assert [m['source_action_step'] for m in mapping[:prefix]] == list(range(1, prefix + 1))
        handoff = read(original / 'handoff.json')
        scores = rows(original / 'wcm/scores.jsonl')
        boundary, diagnostic = warning_boundary(handoff, scores)
        # Recompute the actual frozen detector on the unchanged VLA prefix.
        detector = detector_module.TrendDetector()
        for s in scores[:prefix + 1]:
            reason, diag = detector.update(s['value'], s['observation_index'], terminal=bool(s['success'] or s['truncated']))
            assert reason == s['reason'] and diag == s['diagnostic']
        diagnostic.update(frozen_detector=str(frozen), frozen_detector_sha256=sha(frozen),
                          prefix_diagnostics_recomputed_and_matched=True)
        spans = segment_ranges(mapping, prefix, boundary)
        audit_rel = f'provenance/parent_{pid:06d}'
        audit = temp / audit_rel
        audit.mkdir(parents=True)
        shutil.copy2(report_path, audit / 'conversion_report.json')
        shutil.copy2(mapping_path, audit / 'action_provenance.jsonl')
        shutil.copy2(original / 'report.json', audit / 'original_report.json')
        shutil.copy2(original / 'handoff.json', audit / 'handoff.json')
        shutil.copy2(original / 'wcm/scores.jsonl', audit / 'original_wcm_scores.jsonl')
        save(audit / 'warning_boundary.json', diagnostic)
        (audit / 'programs').mkdir()
        (audit / 'feedback').mkdir()
        used_turns = {s['turn'] for s in spans if s['owner'] == 'capx'}
        for program in sorted((original / 'capx').glob('turn_*.py')):
            turn = int(program.stem.split('_')[-1])
            shutil.copy2(program, audit / 'programs' / program.name)
            for suffix in ['stdout.txt', 'stderr.txt']:
                feedback = program.with_name(f'{program.stem}_{suffix}')
                if feedback.exists():
                    shutil.copy2(feedback, audit / 'feedback' / feedback.name)
            if turn not in used_turns:
                empty_programs.append(dict(parent_episode=pid, turn=turn,
                    program=f'{audit_rel}/programs/{program.name}',
                    status='no_actions_in_converted_trajectory',
                    note='May have had no original actions, or fall after converted early success; no empty episode emitted.'))
        parent_summary = dict(parent_episode=pid, suite=parent['suite'],
            task_id=rec['evaluation_task_id'], trial_id=rec['trial_id'], instruction=rec['instruction'],
            original_episode=str(original), original_success=bool(original_report['success']),
            converted_success=bool(report['success']), converted_frames=n,
            prefix_steps=prefix, first_success_action_row=successes[0] if successes else None,
            boundary=diagnostic, segments=[])
        for span in spans:
            eid = len(episodes)
            start, end = span['start'], span['end']
            segment = table.slice(start, end - start)
            payload = segment.to_pylist()
            for j, row in enumerate(payload):
                row.update(episode_index=eid, frame_index=j, index=offset + j,
                           timestamp=np.float32(j / info['fps']).item())
            new = pa.Table.from_pylist(payload, schema=table.schema)
            for col in ['image', 'wrist_image', 'state', 'actions', 'task_index']:
                assert segment[col].equals(new[col]), col
            output = temp / f'data/chunk-{eid // 1000:03d}/episode_{eid:06d}.parquet'
            output.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(new, output, compression='zstd', row_group_size=64)
            assert pq.read_table(output).equals(new)
            completed = span.get('completion_label') == '完成任务'
            meta = dict(episode_index=eid, tasks=[rec['instruction']], length=end - start,
                        success=completed, category=span['category'],
                        completion_label=span.get('completion_label'), parent_episode_index=pid)
            episodes.append(meta)
            epstats.append(dict(episode_index=eid, stats=statistics(payload, table.schema.names)))
            subset = mapping[start:end]
            rel = f'{audit_rel}/segment_{eid:06d}_mapping.jsonl'
            lines(temp / rel, [dict(m, segment_episode_index=eid, segment_action_row=j,
                                   parent_action_row=start + j) for j, m in enumerate(subset)])
            copied_program = None
            if span['owner'] == 'capx':
                copied_program = f'{audit_rel}/programs/turn_{span["turn"]:02d}.py'
                assert (temp / copied_program).is_file()
                for m in subset:
                    assert sha(Path(m['source_program'])) == sha(temp / copied_program)
            item = dict(**span, episode_index=eid, parent_episode_index=pid,
                suite=parent['suite'], task_id=rec['evaluation_task_id'],
                trial_id=rec['trial_id'], canonical_task_index=parent['canonical_task_index'],
                instruction=rec['instruction'], parent_dataset=str(source), parent_parquet=relative,
                original_episode=str(original), original_success=bool(original_report['success']),
                converted_parent_success=expected_success, task_completed_in_segment=completed,
                converted_action_range=[start, end],
                original_action_steps_inclusive=[subset[0]['source_action_step'], subset[-1]['source_action_step']],
                original_observation_indices=[start, end],
                inserted_compensation_actions=sum(bool(m['inserted_compensation_action']) for m in subset),
                program=copied_program, feedback_directory=f'{audit_rel}/feedback',
                action_mapping=rel, global_row_range=[offset, offset + end - start],
                weak_vla_label=span['owner'] == 'pi05', wcm_reevaluated=False,
                original_wcm_scores=f'{audit_rel}/original_wcm_scores.jsonl',
                original_wcm_valid_for_converted_segment=span['owner'] == 'pi05',
                native_control_hz=20, logical_dataset_fps=info['fps'],
                segment_boundary_is_environment_terminal=bool(subset[-1]['terminated']),
                parent_recorded_truncated_after_action=bool(subset[-1].get('truncated', False)),
                segmentation_changes_control_actions=False)
            provenance.append(item)
            parent_summary['segments'].append(item)
            offset += end - start
            print('SEGMENT', source.name, pid, eid, span['category'], start, end,
                  span.get('completion_label'), flush=True)
        save(audit / 'segments.json', parent_summary)
        parent_reports.append(parent_summary)
        input_frames += n
        save(temp / 'progress.json', dict(state='exporting',parents=len(parent_reports),
                                         segments=len(episodes), frames=offset))
    assert offset == input_frames
    assert sum(x['task_completed_in_segment'] for x in provenance) == (len(parents) if source.name.startswith('recovery_success') else 0)
    assert unchanged == snapshot(source), 'Source dataset changed'
    save(temp / 'source_checksums.json', unchanged)
    count = len(episodes)
    info.update(total_episodes=count, total_frames=offset, total_chunks=(count + 999) // 1000,
                chunks_size=1000, total_videos=0, splits=dict(train=f'0:{count}'))
    save(temp / 'meta/info.json', info)
    for name, data in [('episodes', episodes), ('episodes_stats', epstats), ('provenance', provenance)]:
        lines(temp / 'meta' / f'{name}.jsonl', data)
    shutil.copy2(source / 'meta/tasks.jsonl', temp / 'meta/tasks.jsonl')
    lines(temp / 'programs_without_converted_actions.jsonl', empty_programs)
    categories = dict(Counter(x['category'] for x in provenance))
    labels = dict(Counter(x['completion_label'] for x in provenance if x['owner'] == 'capx'))
    save(temp / 'selection.json', dict(
        vla_progress_candidate=[x['episode_index'] for x in provenance if x['category'] == 'vla_progress_candidate'],
        vla_wcm_warning=[x['episode_index'] for x in provenance if x['category'] == 'vla_wcm_warning'],
        capx_completed=[x['episode_index'] for x in provenance if x.get('completion_label') == '完成任务'],
        capx_not_completed=[x['episode_index'] for x in provenance if x.get('completion_label') == '没有完成任务']))
    result = dict(state='exported_pending_loader_validation', path=str(dest), parent_episodes=len(parents),
        episodes=count, frames=offset, categories=categories, capx_completion_labels=labels,
        input_frames=input_frames, action_partition_exact=True, images_state_actions_unchanged=True,
        source_dataset_unchanged=True, source_sha256=unchanged,
        programs_without_converted_actions=len(empty_programs), wcm_reevaluated=False,
        frozen_detector=str(frozen), frozen_detector_sha256=sha(frozen),
        training_started=False, cuda_used=False)
    save(temp / 'segmentation_report.json', result)
    (temp / 'README.md').write_text(
        '# Program-level recovery segments (LeRobot v2.1)\n\n'
        'Each episode is a contiguous segment of an existing converted parent. Images, state and '
        '7D OSC actions are unchanged; only episode/frame/global indices and logical timestamp restart. '
        'VLA progress/warning are weak WCM labels, not proven physical failure onset. '
        'CAP-X labels are only 完成任务 / 没有完成任务, from converted LIBERO transitions. '
        '没有完成任务 does not mean no useful progress. Parent final failure/success is separate.\n\n'
        'Each executed Python program is one segment; inserted controls stay with that program. '
        'No empty episodes. programs_without_converted_actions.jsonl also includes original programs '
        'not reached after converted early success. All source programs and stdout/stderr are copied. '
        'meta/provenance.jsonl records task/trial, parent ranges, program and segment mappings. '
        'selection.json provides category-specific episode IDs.\n\n'
        'Native simulation20Hz; fps10 is logical indexing inherited from the source, without resampling. '
        'Use original RLinf normalization. Horizon10 must pad inside each new episode. '
        'A segment end is not an invented environment termination. For RL use mapped terminated/truncated '
        'flags, not segment length. Original WCM curves are not scores for converted CAP-X observations. '
        'Original VLA prefixes are unchanged and their WCM scores remain aligned. '
        'Contains unfinished/failed examples; not automatically added to successful SFT. No training/API/GPU calls.\n')
    temp.rename(dest)
    return result


def validate(dest, root):
    import numpy as np
    from direct_data import DirectEpisodes
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    import pyarrow.parquet as pq
    report = read(dest / 'segmentation_report.json')
    provenance = rows(dest / 'meta/provenance.jsonl')
    ds = DirectEpisodes([dict(name=dest.name, path=str(dest), episodes=report['episodes'])], cache_row_groups=4)
    lr = LeRobotDataset(str(dest), delta_timestamps={'actions': [i / 10 for i in range(10)]})
    assert len(ds) == len(lr) == report['frames']
    assert lr.num_episodes == len(ds.episodes) == report['episodes']
    windows = 0
    start = 0
    parent_key, parent_table = None, None
    for eid, rec in enumerate(ds.episodes):
        length = rec['length']
        assert int(lr.episode_data_index['from'][eid]) == start
        assert int(lr.episode_data_index['to'][eid]) == start + length
        for local in sorted(set([0, length // 2, *range(max(0, length - 10), length)])):
            sample, sample_lr = ds[start + local], lr[start + local]
            expected = rec['actions'][np.minimum(local + np.arange(10), length - 1)]
            np.testing.assert_array_equal(sample['actions'], expected)
            np.testing.assert_array_equal(sample_lr['actions'].numpy(), expected)
            np.testing.assert_array_equal(sample_lr['actions_is_pad'].numpy(), np.arange(local, local + 10) >= length)
            np.testing.assert_array_equal(sample_lr['state'].numpy(), sample['state'])
            assert sample_lr['task'] == sample['prompt']
            if local in (0, length - 1):
                for key in ['image', 'wrist_image']:
                    decoded = (sample_lr[key].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                    np.testing.assert_array_equal(decoded, sample[key])
            windows += 1
        # Recheck each published row against parent for ALL immutable columns.
        p = provenance[eid]
        key = (p['parent_dataset'], p['parent_parquet'])
        if key != parent_key:
            parent_table = pq.read_table(Path(key[0]) / key[1])
            parent_key = key
        parent = parent_table.slice(p['start'], length)
        published = pq.read_table(rec['path'])
        for key in ['image', 'wrist_image', 'state', 'actions', 'task_index']:
            assert published[key].equals(parent[key])
        start += length
    ds.assert_unchanged()
    assert snapshot(Path(provenance[0]['parent_dataset'])) == read(dest / 'source_checksums.json')
    report.update(state='complete', direct_reader_verified=True, lerobot_reader_verified=True,
                  windows_checked=windows, horizon10_never_crosses_segments=True)
    save(dest / 'segmentation_report.json', report)
    save(dest / 'validation_report.json', report)
    return report


def main():
    os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--resume', action='store_true', help='Validate existing exports; never overwrite them')
    args = ap.parse_args()
    root = args.root
    sys.path.insert(0, str(root / 'training_configs/pi05_ablation_direct_20261029'))
    out = args.out or root / 'dataset_subsets/pi05_recovery_segmented'
    if args.pilot:
        out = out / 'pilot'
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, rel in SOURCES.items():
        dest = out / f'{name}_segmented'
        save(out / 'status.json', dict(state='processing', current_dataset=str(dest), datasets=results,
                                     training_started=False, cuda_used=False))
        if not args.validate_only and not (args.resume and dest.exists()):
            export(root / rel, dest, 1 if args.pilot else None, root)
        result = validate(dest, root)
        results.append(result)
        save(out / 'status.json', dict(state='validating', datasets=results, training_started=False))
        print('DATASET_VALIDATED', dest, result['episodes'], result['frames'], flush=True)
    # One actual RLinf/OpenPI transformed training batch per new dataset.
    validator_dir = root / 'training_configs/pi05_matched17_osc_split_20261028'
    sys.path.insert(0, str(validator_dir))
    validator = load_module('segment_loader_audit', validator_dir / 'validate_conversion.py')
    save(out / 'summary.json', dict(datasets=[dict(path=r['path'], episodes=r['episodes'], frames=r['frames']) for r in results]))
    validator.EXPORT = out
    loaders = validator.audit_loader()
    result = dict(state='complete', pilot=args.pilot, datasets=results, rlinf_loaders=loaders,
                  training_started=False, cuda_used=False)
    save(out / 'summary.json', result)
    save(out / 'status.json', result)
    print('SEGMENTATION_VALIDATED', json.dumps({k: v for k, v in result.items() if k != 'datasets'}), flush=True)


if __name__ == '__main__':
    main()
