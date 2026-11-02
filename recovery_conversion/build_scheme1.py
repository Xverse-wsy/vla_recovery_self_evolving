"""Export fixed recovery40, trim VLA progress tails, build strict Stage1 banks.

No optimizer/training, API calls or new robot actions. Conversion is a prerequisite.
Only the frozen policy-visible instruction/state8/front+wrist RGB enter matching.
"""
import argparse
from collections import Counter, defaultdict
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

import convert_selected_success as core


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


segment = load_module('scheme1_existing_segmentation', core.HERE / 'segmentation_engine/segment.py')


def trim_progress_span(span, count=10):
    out = dict(span)
    if span['category'] == 'vla_progress_candidate':
        out['untrimmed_parent_range'] = [span['start'], span['end']]
        out['end'] = max(span['start'], span['end'] - count)
        out['discarded_parent_range'] = [out['end'], span['end']]
        out['discarded_tail_actions_and_start_frames'] = span['end'] - out['end']
    return out


def png(rgb):
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format='PNG', compress_level=1)
    return dict(bytes=buf.getvalue(), path=None)


def trajectory_rows(path):
    """Yield the PRE-action observation and corresponding recorded action."""
    meta = core.read(path / 'manifest.json')
    with np.load(path / 'initial.npz', allow_pickle=False) as d:
        before = {k: d[k].copy() for k in d.files}
    count = 0
    for part in meta['shards']:
        f = path / part['file']
        assert core.sha(f) == part['sha256']
        with np.load(f, allow_pickle=False) as d:
            # NpzFile does not cache decoded members: decode each shard field ONCE.
            arrays = {k: d[k].copy() for k in d.files if k.startswith(('raw/', 'policy/', 'sim/'))}
            actions = d['transition/action'].copy()
            for j, action in enumerate(actions):
                yield before, action.copy()
                before = {k: values[j].copy() for k, values in arrays.items()}
                count += 1
    assert count == meta['steps']


def parent_table(trajectory, schema, task):
    sys.path.insert(0, str(core.ENGINE))
    from convert_recovery import state8
    payload = []
    for obs, action in trajectory_rows(trajectory):
        state = np.asarray(obs['policy/state'], np.float32).reshape(-1) if 'policy/state' in obs else state8(
            {k[4:]: v for k, v in obs.items() if k.startswith('raw/')})
        assert state.shape == (8,) and np.isfinite(state).all()
        action = np.asarray(action[:7], np.float32)
        assert action.shape == (7,) and np.isfinite(action).all()
        payload.append(dict(image=png(obs['raw/agentview_image'][::-1, ::-1]),
            wrist_image=png(obs['raw/robot0_eye_in_hand_image'][::-1, ::-1]),
            state=state.tolist(), actions=action.tolist(), timestamp=0., frame_index=0,
            episode_index=0, index=0, task_index=task))
    return pa.Table.from_pylist(payload, schema=schema)


class DatasetWriter:
    def __init__(self, path, base):
        assert not path.exists(), 'Use a new output; do not overwrite a dataset: ' + str(path)
        self.path, self.base = path, base
        (path / 'meta').mkdir(parents=True)
        self.info = core.read(base / 'meta/info.json')
        self.schema = pq.read_schema(base / 'data/chunk-000/episode_000000.parquet')
        self.episodes, self.stats, self.provenance = [], [], []
        self.offset = 0

    def add(self, table, provenance, instruction, **metadata):
        eid, n = len(self.episodes), len(table)
        assert n > 0
        values = table.to_pylist()
        for i, row in enumerate(values):
            row.update(episode_index=eid, frame_index=i, index=self.offset + i,
                timestamp=float(np.float32(i / self.info['fps'])))
        remapped = pa.Table.from_pylist(values, schema=self.schema)
        for name in ['image', 'wrist_image', 'state', 'actions', 'task_index']:
            assert table[name].equals(remapped[name]), name
        dest = self.path / ('data/chunk-%03d/episode_%06d.parquet' % (eid // 1000, eid))
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(remapped, dest, compression='zstd', row_group_size=64)
        assert pq.read_table(dest).equals(remapped)
        self.episodes.append(dict(episode_index=eid, tasks=[instruction], length=n, **metadata))
        self.stats.append(dict(episode_index=eid, stats=segment.statistics(values, self.schema.names)))
        self.provenance.append(dict(provenance, episode_index=eid,
            global_row_range=[self.offset, self.offset + n]))
        self.offset += n
        return eid, dest

    def finish(self):
        n = len(self.episodes)
        self.info.update(total_episodes=n, total_frames=self.offset, total_chunks=(n + 999) // 1000,
            total_videos=0, chunks_size=1000, splits=dict(train='0:%d' % n))
        core.save(self.path / 'meta/info.json', self.info)
        for name, values in [('episodes', self.episodes), ('episodes_stats', self.stats), ('provenance', self.provenance)]:
            segment.lines(self.path / 'meta' / (name + '.jsonl'), values)
        shutil.copy2(self.base / 'meta/tasks.jsonl', self.path / 'meta/tasks.jsonl')
        core.save(self.path / 'status.json', dict(state='complete', episodes=n, frames=self.offset))


def export_group(root, output, group, pilot=False):
    name = 'recovery_' + group + '20'
    if (output / (name + '_export_complete.json')).exists():
        assert core.read(output / (name + '_export_complete.json'))['state'] == 'complete'
        return
    raw = root / 'dataset_subsets/pi05_recovery_raw_scheme1_20261102_v1'
    work = root / 'trajectory_conversion_tests' / ('pi05_scheme1_' + group + '20_20261102')
    assert core.read(work / 'status.json')['state'] == 'complete'
    selected = core.rows(raw / (name + '_raw/episodes.jsonl'))
    if pilot:
        selected = [next(x for x in selected if x['dataset_episode_index'] == (18 if group == 'success' else 7))]
    base = root / 'dataset_subsets/pi05_autonomous20_matched_recovery20_20261029/autonomous_success20_lerobot'
    writer = DatasetWriter(output / name, base)
    pieces = DatasetWriter(output / (name + '_segmented'), base)
    tasks = {(x['suite'], x['evaluation_task_id']): x for x in core.read(core.HERE / 'tasks.json')}
    frozen = root / segment.DETECTOR_REL
    detector_module = load_module('scheme1_frozen_trend', frozen)
    shutil.copy2(frozen, pieces.path / 'frozen_trend.py')
    audits, excluded = [], []
    for item in selected:
        pid = item['dataset_episode_index']
        status = core.read(work / 'results' / ('episode_%06d.json' % pid))
        snapshot = raw / item['snapshot_relative_path']
        original = core.read(snapshot / 'report.json')
        task = tasks[(item['suite'], item['evaluation_task_id'])]
        startup = status.get('conversion_status') == 'not_applicable_CAP_startup_failure'
        if startup:
            trajectory = snapshot / 'trajectory'
            mapping = []
            for i, (_, action) in enumerate(trajectory_rows(trajectory)):
                mapping.append(dict(frame_index=i, observation_index=i, next_observation_index=i + 1,
                    owner='pi05', source_turn=None, source_program=None, copied_program=None,
                    source_action_step=i + 1, inserted_compensation_action=False,
                    action=np.asarray(action[:7], np.float32).tolist(), success_after_action=False,
                    terminated=False, truncated=False))
            converted_success, report = None, dict(prefix_steps=item['vla_steps'], executed_steps=len(mapping))
        else:
            retained = Path(status['retained_episode'])
            trajectory = retained / 'trajectory'
            report = core.read(retained / 'report.json')
            checked = core.read(retained / 'independent_replay_report.json')
            assert checked['success'] == report['success'] and checked['max_sim_state_error'] < 1e-8
            assert checked['action_record_sha256'] == core.sha(retained / 'states_actions.npz')
            assert checked['source_report_sha256'] == core.sha(retained / 'report.json')
            mapping = core.rows(retained / 'action_provenance.jsonl')
            converted_success = report['success']
        prefix = int(report['prefix_steps'])
        table = parent_table(trajectory, writer.schema, task['canonical_task_index'])
        assert len(table) == len(mapping) == report['executed_steps']
        np.testing.assert_array_equal(np.asarray(table['actions'].to_pylist(), np.float32),
            np.asarray([m['action'] for m in mapping], np.float32))
        original_episode = item['source_episode']
        rec = dict(suite=item['suite'], evaluation_task_id=item['evaluation_task_id'], trial_id=item['trial_id'],
            canonical_task_index=task['canonical_task_index'], instruction=item['instruction'],
            source_episode=str(snapshot), original_source_episode=original_episode, handoff_step=prefix)
        audit_rel = 'provenance/parent_%06d' % pid
        audit = pieces.path / audit_rel
        (audit / 'programs').mkdir(parents=True)
        for fname in ['report.json', 'handoff.json']:
            shutil.copy2(snapshot / fname, audit / ('original_' + fname))
        shutil.copy2(snapshot / 'wcm/scores.jsonl', audit / 'original_wcm_scores.jsonl')
        for f in (snapshot / 'capx').glob('turn_*'):
            if f.suffix == '.py' or f.name.endswith(('_stdout.txt', '_stderr.txt')):
                shutil.copy2(f, audit / 'programs' / f.name)
        if not startup:
            shutil.copy2(retained / 'report.json', audit / 'conversion_report.json')
            shutil.copy2(retained / 'independent_replay_report.json', audit / 'independent_replay_report.json')
        segment.lines(audit / 'action_provenance.jsonl', mapping)
        scores = core.rows(snapshot / 'wcm/scores.jsonl')
        handoff = core.read(snapshot / 'handoff.json')
        boundary, diag = segment.warning_boundary(handoff, scores)
        detector = detector_module.TrendDetector()
        for s in scores[:prefix + 1]:
            reason, diagnostic = detector.update(s['value'], s['observation_index'], terminal=bool(s['success'] or s['truncated']))
            assert reason == s['reason'] and diagnostic == s['diagnostic']
        diag.update(prefix_diagnostics_recomputed_and_matched=True, frozen_detector_sha256=core.sha(frozen))
        spans = segment.segment_ranges(mapping, prefix, boundary)
        original_turns = [p['turn'] for p in status.get('boundary_audit', {}).get('program_ranges', [])
            if p['start'] <= item['original_steps'] <= p['end']]
        if not original_turns:
            # The successful source's audit is authoritative; original failures have no successful program.
            original_success_turn = None
        else:
            original_success_turn = original_turns[0]
        export_eid = len(writer.episodes)
        fullprov = dict(source_record=rec, exported_episode_index=export_eid, source_selection_index=pid, suite=item['suite'],
            canonical_task_index=task['canonical_task_index'], original_episode=original_episode,
            original_success=original['success'], converted_success=converted_success,
            original_recording_complete=item['recording_audit']['complete'],
            startup_only=startup, source_trajectory_manifest_sha256=core.sha(trajectory / 'manifest.json'))
        writer.add(table, fullprov, item['instruction'], original_success=original['success'],
            converted_success=converted_success, startup_only=startup)
        video = snapshot / 'execution.mp4' if startup else retained / 'execution.mp4'
        if video.exists():
            video_dest = writer.path / 'review_videos' / ('episode_%06d.mp4' % pid)
            video_dest.parent.mkdir(exist_ok=True)
            try:
                __import__('os').link(video, video_dest)
            except OSError:
                shutil.copy2(video, video_dest)
        parent_audit = dict(parent_episode=pid, prefix_steps=prefix, boundary=diag,
            original_success=original['success'], converted_success=converted_success,
            original_episode=original_episode, original_success_turn=original_success_turn, segments=[])
        for rawspan in spans:
            span = trim_progress_span(rawspan)
            if span.get('discarded_tail_actions_and_start_frames'):
                excluded.append(dict(parent_episode=pid, **span))
            start, end = span['start'], span['end']
            if start == end:
                continue
            subset = mapping[start:end]
            completed = span.get('completion_label') == '完成任务'
            eligible = bool(completed and original['success'] and original_success_turn == span['turn'])
            eid = len(pieces.episodes)
            local_map = audit_rel + '/segment_%06d_mapping.jsonl' % eid
            segment.lines(pieces.path / local_map, [dict(m, parent_action_row=start + j,
                segment_action_row=j, segment_episode_index=eid) for j, m in enumerate(subset)])
            prov = dict(span, suite=item['suite'], task_id=item['evaluation_task_id'],
                trial_id=item['trial_id'], canonical_task_index=task['canonical_task_index'], instruction=item['instruction'],
                parent_episode_index=pid, parent_dataset=str(writer.path),
                parent_parquet=writer.info['data_path'].format(episode_chunk=export_eid // 1000, episode_index=export_eid),
                original_episode=original_episode, original_success=original['success'],
                converted_parent_success=converted_success, task_completed_in_segment=completed,
                cap_matching_eligible=eligible, original_success_turn=original_success_turn,
                action_mapping=local_map, native_control_hz=20, logical_dataset_fps=writer.info['fps'],
                program=(audit_rel + '/programs/turn_%02d.py' % span['turn']) if span['turn'] is not None else None)
            pieces.add(table.slice(start, end - start), prov, item['instruction'], category=span['category'],
                success=completed, completion_label=span.get('completion_label'), parent_episode_index=pid)
            parent_audit['segments'].append(prov)
        core.save(audit / 'segments.json', parent_audit)
        audits.append(parent_audit)
        core.save(output / 'export_progress.json', dict(group=group, completed_parents=pid + 1, total=20))
        print('SCHEME1_PARENT_EXPORTED', group, pid, len(table), flush=True)
    for ds in [writer, pieces]:
        ds.finish()
    categories = {name: [p['episode_index'] for p in pieces.provenance if p['category'] == name]
        for name in ['vla_progress_candidate', 'vla_wcm_warning']}
    categories.update(capx_completed=[p['episode_index'] for p in pieces.provenance if p['task_completed_in_segment']],
        capx_not_completed=[p['episode_index'] for p in pieces.provenance
            if p['owner'] == 'capx' and not p['task_completed_in_segment']])
    core.save(pieces.path / 'selection.json', categories)
    segment.lines(pieces.path / 'discarded_vla_progress_tails.jsonl', excluded)
    core.save(pieces.path / 'segmentation_report.json', dict(state='complete', parent_episodes=len(selected),
        episodes=len(pieces.episodes), frames=pieces.offset, categories={k: len(v) for k, v in categories.items()},
        discarded_tail_actions=sum(p['discarded_tail_actions_and_start_frames'] for p in excluded),
        cap_success_matching_eligible_segments=sum(p['cap_matching_eligible'] for p in pieces.provenance),
        frozen_detector_sha256=core.sha(frozen), prefix_diagnostics_verified=True,
        native_control_hz=20, logical_dataset_fps=writer.info['fps'], training_started=False))
    core.save(output / (name + '_export_complete.json'), dict(state='complete', parents=len(selected), pilot=pilot))


def algorithm(pair_code):
    value = core.read(pair_code / 'configs/libero_pi05_positive_fm.json')
    if value.get('training_recipe') != 'stage1_a60_sft_v1':
        raise ValueError('Use the A60-aligned full-model Stage1 code before building new assets')
    value['coverage'] = dict(position_m=.01, rotation_deg=5., proprio_rms=.10, image_cosine=.002, max_candidates=50)
    value['pairing'] = dict(value['coverage'])
    value['split_fractions'] = [1., 0., 0.]
    value['cap_success_admission'] = 'segment_suffix'
    return value


def validate_exports(output, pair_code):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    sys.path.insert(0, str(pair_code.parent))
    from direct_data import DirectEpisodes
    reports = []
    for group in ['success', 'failure']:
        for ending in ['', '_segmented']:
            path = output / ('recovery_' + group + '20' + ending)
            count = core.read(path / 'meta/info.json')['total_episodes']
            direct = DirectEpisodes([dict(name=path.name, path=str(path), episodes=count)], cache_row_groups=2)
            lr = LeRobotDataset(str(path), delta_timestamps={'actions': [i / 10 for i in range(10)]})
            assert len(lr) == len(direct)
            offsets = [0] + direct.ends[:-1]
            probes = 0
            for n in sorted({0, count - 1}):
                length = direct.episodes[n]['length']
                for local in sorted({0, length - 1, max(0, length - 10)}):
                    a, b = direct[offsets[n] + local], lr[offsets[n] + local]
                    np.testing.assert_array_equal(a['actions'], b['actions'].numpy())
                    np.testing.assert_array_equal(a['state'], b['state'].numpy())
                    assert a['prompt'] == b['task']
                    for view in ['image', 'wrist_image']:
                        rgb = (b[view].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                        np.testing.assert_array_equal(a[view], rgb)
                    probes += 1
            direct.assert_unchanged()
            reports.append(dict(dataset=path.name, episodes=count, frames=len(direct), boundary_probes=probes,
                real_lerobot_and_direct_readers_agree=True, images_state_actions_equal=True))
    core.save(output / 'loader_validation.json', dict(state='complete', datasets=reports))


def build_banks(root, output, pair_code):
    from recovery_pair.library_build import build_existing_libraries
    validate_exports(output, pair_code)
    config = output / 'algorithm.json'
    core.save(config, algorithm(pair_code))
    norm = root / 'models/RLinf-Pi05-LIBERO-SFT/physical-intelligence/libero/norm_stats.json'
    # Original deployment keeps the same assets under its configured model path.
    if not norm.exists():
        candidates = list((root / 'models').glob('*/physical-intelligence/libero/norm_stats.json'))
        assert len(candidates) == 1, candidates
        norm = candidates[0]
    core.save(output / 'sources.json', dict(segmented_datasets=[str(output / ('recovery_' + g + '20_segmented')) for g in ['success', 'failure']],
        autonomous_datasets=[str(root / 'dataset_subsets/pi05_autonomous20_matched_recovery20_20261029/autonomous_success20_lerobot')],
        norm_stats=str(norm), algorithm_config=str(config), expected_image_shape=[256, 256, 3]))
    build_existing_libraries(output / 'sources.json', root, output / 'banks')


def strict_matching(root, output, device, memory):
    import recovery_pair.matching as matching
    from recovery_pair.config import Config
    from recovery_pair.schema import load_windows, save_windows
    from recovery_pair.pi05_vision import Pi05Vision
    from recovery_pair.data import write_jsonl
    config = Config.load(output / 'algorithm.json')
    all_records = load_windows(output / 'banks/windows/windows.jsonl')
    # Existing classes remain in the original bank; gate only CAP admission candidates.
    records = [r for r in all_records if r.controller == 'VLA' or
        (r.training_quality == 'positive' and r.source['provenance']['cap_matching_eligible'])]
    original_numeric = matching.numeric_match
    def with_gripper(q, r, thresholds):
        distances, reason = original_numeric(q, r, thresholds)
        if reason is None and np.max(np.abs(np.asarray(q.proprio)[6:] - np.asarray(r.proprio)[6:])) > .005:
            return None, 'gripper_gap_over_5mm_per_finger'
        return distances, reason
    matching.numeric_match = with_gripper
    groups = defaultdict(list)
    for r in records:
        if r.controller == 'VLA' and r.training_quality == 'positive':
            groups[(r.split, matching.instruction_key(r))].append(r)
    needed, candidates = set(), []
    for q in records:
        if q.controller != 'code':
            continue
        found = []
        for ref in groups[(q.split, matching.instruction_key(q))]:
            d, reason = with_gripper(q, ref, config.coverage)
            if reason is None:
                score = sum(d[k] / getattr(config.coverage, k) for k in ['position_m', 'rotation_deg', 'proprio_rms'])
                found.append((score, ref.sample_id, d))
        found = sorted(found)[:50]
        if found:
            needed.add(q.sample_id)
            needed.update(x[1] for x in found)
        candidates.append(dict(sample_id=q.sample_id, candidates=[dict(reference_id=sid, distances=d) for _, sid, d in found]))
    matching_out = output / 'matching'
    matching_out.mkdir(exist_ok=False)
    write_jsonl(matching_out / 'numeric_candidates.jsonl', candidates)
    feature_report = None
    if needed:
        checkpoint = root / 'models/RLinf-Pi05-LIBERO-SFT'
        encoder = Pi05Vision(checkpoint, device, memory)
        feature_report = encoder.extract([r for r in records if r.sample_id in needed], matching_out / 'feature_cache', 2)
        del encoder
    matcher = matching.FrozenMatcher(records, config)
    for suffix in matcher.success_segments:
        r = next(r for r in records if r.sample_id == suffix['anchor_id'])
        assert suffix['segment_row_range'][1] == pq.ParquetFile(r.image_refs['front']['parquet']).metadata.num_rows
        assert r.source['provenance']['cap_matching_eligible']
    admission = matching_out / 'admission'
    admission.mkdir()
    save_windows(admission / 'positive_pool.jsonl', matcher.positive)
    save_windows(admission / 'admitted_cap_success_windows.jsonl', [r for r in matcher.positive if r.controller == 'code'])
    for name, values in [('success_matches', matcher.success_matches), ('cap_success_suffixes', matcher.success_segments),
                         ('rejections', matcher.rejections), ('coverage', [dict(sample_id=k, **v) for k, v in matcher.coverage.items()])]:
        write_jsonl(admission / (name + '.jsonl'), values)
    core.save(admission / 'config.json', config.to_dict())
    core.save(admission / 'report.json', dict(state='complete', observation_policy=matching.OBSERVATION_POLICY,
        positive_pool_windows=len(matcher.positive), vla_positive_windows=len(matcher.vla_success),
        directly_matched_cap_windows=len(matcher.success_matches), admitted_cap_success_segments=len(matcher.success_segments),
        admitted_cap_action_steps=sum(s['action_steps'] for s in matcher.success_segments),
        coverage=config.to_dict()['coverage'], gripper_max_abs_delta_m=.005, cap_success_admission='segment_suffix',
        feature_report=feature_report, excluded_cap_success_windows=len([r for r in all_records if r.controller == 'code' and
            r.training_quality == 'positive' and not r.source['provenance']['cap_matching_eligible']]), training_started=False))
    print('SCHEME1_ADMISSION', core.read(admission / 'report.json'), flush=True)


def prepare(root, output):
    from recovery_pair.config import Config
    from recovery_pair.positive_fm import prepare_positive_fm, audit_source_alignment
    from recovery_pair.schema import load_windows
    config = Config.load(output / 'algorithm.json')
    report = prepare_positive_fm(output / 'matching/admission', config, output / 'prepared', all_train=True,
        expert_dataset=root / 'dataset_subsets/rlinf_pi05_libero_fewshot40_20261021', expected_expert_episodes=40)
    sources = core.read(output / 'sources.json')
    audit = audit_source_alignment(load_windows(output / 'prepared/training_pool.jsonl'), sources['norm_stats'])
    core.save(output / 'prepared/source_alignment_audit.json', audit)
    core.save(output / 'summary.json', dict(state='complete', training_started=False,
        segmentation={g: core.read(output / ('recovery_' + g + '20_segmented/segmentation_report.json')) for g in ['success', 'failure']},
        admission=core.read(output / 'matching/admission/report.json'), prepared=report, source_alignment=audit))
    (output / 'README.md').write_text(
        '# Scheme1 recovery40 and strict Stage1 fragment library\n\n'
        'Fixed original-success20 (converted12 success/8 failure) and fixed original-failure20. '
        'Original and converted outcomes are separate; every outcome is retained. '
        'Two original-failure episodes have CAP startup failures, no CAP actions and incomplete '
        'original recorder completion; only their saved VLA actions/observations are exported, '
        'explicitly marked startup_only, without inventing CAP actions or a replay outcome.\n\n'
        'Load recovery_success20/recovery_failure20 as separate LeRobot v2.1 datasets. '
        'They contain real aligned front/wrist PNGs, state8 and OSC action7. '
        'Native control20Hz; logical fps10 is inherited without temporal resampling. '
        'review_videos contains converted videos (original VLA-only video for startup-only cases).\n\n'
        'The *_segmented directories reuse the frozen original WCM warning onset and one '
        'segment per action-bearing CAP program. Progress-prefix segments physically discard '
        'their final10 actions AND their starting image/state frames; discarded ranges are audited. '
        'Unknown VLA boundaries are not upgraded to positives. CAP completion labels come from '
        'actual converted LIBERO transitions. Only original-success/converted-success programs '
        'whose successful code round agrees may query positive admission.\n\n'
        'Matching: same public instruction; policy state8 position<=1cm, rotation<=5deg, '
        'quantile-normalized state RMS<=0.10, per-finger position gap<=5mm; max50 numerical '
        'candidates. Frozen native pi05 SigLIP front AND wrist cosine distance<=0.002. '
        'No progress score, object pose, contact/held truth or future image enters retrieval. '
        'Match stride10; earliest matched start admits the same successful program suffix '
        'through actual task success. Later suffix states are NOT claimed independently matched.\n\n'
        'Dense Stage1 starts every retained frame; intermediate VLA requires10 real actions '
        '(no padding); successful autonomous/CAP/expert terminal windows repeat last action. '
        'Expert40 is training-only, never in the matching pool. All admitted positives are train. '
        'No model training/optimizer update has been started.\n\n'
        'stage1_libraries.tar.gz contains the four original banks, strict admission and audited '
        'positive-only prepared library. Indexes retain A800 absolute provenance; on a new machine '
        'rebuild banks/matching/prepared using rebuild_scripts.tar.gz, the adaptation repo, '
        'the downloaded segmented directories, existing autonomous20/expert40, the base checkpoint '
        'and original normalization. See README_SCHEME1_STAGE1.md in the scripts archive.\n')
    core.save(output / 'status.json', dict(state='complete', training_started=False,
        positive_windows=report['positive_windows'], supervised_unique_real_frames=report['unique_real_frames']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--pair-code', type=Path, required=True)
    p.add_argument('--stage', choices=['export', 'validate', 'banks', 'match', 'prepare', 'align-training'], required=True)
    p.add_argument('--group', choices=['success', 'failure'])
    p.add_argument('--pilot', action='store_true', help='Independent short success + startup-failure real-data probe')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--memory-gib', type=float, default=6.)
    args = p.parse_args(); root = args.root.resolve()
    output = root / 'dataset_subsets/pi05_recovery_scheme1_stage1_20261102'
    if args.pilot:
        output = output / 'pilot'
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.pair_code.resolve()))
    if args.stage == 'export':
        for group in ([args.group] if args.group else ['success', 'failure']):
            export_group(root, output, group, pilot=args.pilot)
    elif args.stage == 'validate':
        validate_exports(output, args.pair_code.resolve())
    elif args.stage == 'banks':
        build_banks(root, output, args.pair_code.resolve())
    elif args.stage == 'match':
        strict_matching(root, output, args.device, args.memory_gib)
    elif args.stage == 'align-training':
        from align_stage1_a60 import align
        print(json.dumps(align(root, output, args.pair_code.resolve()), indent=2))
    else:
        prepare(root, output)


if __name__ == '__main__':
    main()
