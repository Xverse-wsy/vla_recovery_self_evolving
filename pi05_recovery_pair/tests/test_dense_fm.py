from dataclasses import replace
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from recovery_pair.data import build_windows, QuantileAssets, write_jsonl
from recovery_pair.features import sha256
from recovery_pair.matching import OBSERVATION_POLICY
from recovery_pair.pipeline import audit_prepared
from recovery_pair.positive_fm import prepare_positive_fm, audit_source_alignment
from recovery_pair.schema import save_windows


def dense_fixture(tmp, config, *, label_cuts=False):
    config = replace(config, pair_batch=0, eta=0, cap_success_admission='segment_suffix')
    norm = tmp/'norm.json'
    norm.write_text(json.dumps({'norm_stats': {k: {'q01': [-1]*d, 'q99': [1]*d}
                                               for k,d in [('actions',7), ('state',8)]}}))
    manifest = []
    for segment, category, controller, length in [
        ('progress', 'vla_progress_candidate', 'VLA', 23),
        ('short', 'vla_progress_candidate', 'VLA', 4),
        ('auto', 'autonomous_episode_success', 'VLA', 12),
        ('cap', 'capx_completed', 'code', 23)]:
        path = tmp/(segment+'.parquet')
        pq.write_table(pa.Table.from_pylist([dict(
            state=[j*.001,0,0,0,0,0,.02,.02], actions=[j*.01]*7,
            image={'bytes': b'fixture', 'path': None}, wrist_image={'bytes': b'fixture', 'path': None})
            for j in range(length)]), path)
        manifest.append(dict(segment_id=segment, parquet=str(path), length=length,
            task_id=segment, episode_id=segment, initial_group_id=segment, event_id=segment,
            controller=controller, instruction='place block', parent_start_step=100,
            state_layout='eef_xyz_axisangle_gripper2', split='train',
            bank_class='D_vla_positive' if controller=='VLA' else 'D_code_positive',
            bank_label={'source_category': category},
            source={'normalization_asset': str(norm)}, annotations=[]))
        if label_cuts and segment=='progress':
            manifest[-1]['annotations'] = [dict(start=a, end=b, local_quality='positive',
                event_id='event-'+str(a), quality_evidence={'verified_local_actions': True})
                for a,b in [(0,12),(12,23)]]
    windows = build_windows(manifest, QuantileAssets(norm), config, tmp/'sparse')
    # Admit only the last 13 real CAP actions, never its unmatched prefix.
    positive = [r for r in windows if r.controller != 'code' or r.source['segment_row'] >= 10]
    cap = [r for r in positive if r.controller=='code']
    admission = tmp/'admission'
    admission.mkdir()
    save_windows(admission/'positive_pool.jsonl', positive)
    write_jsonl(admission/'cap_success_suffixes.jsonl', [dict(
        episode_id='cap', segment_id='cap', segment_row_range=[10,23], anchor_id=cap[0].sample_id,
        admitted_window_ids=[r.sample_id for r in cap], admission_kind='successful_segment_suffix')])
    (admission/'config.json').write_text(json.dumps(config.to_dict()))
    (admission/'report.json').write_text(json.dumps(dict(state='complete',
        observation_policy=OBSERVATION_POLICY, positive_pool_windows=len(positive))))
    return config, admission, norm


def test_dense_every_frame_alignment_and_source_specific_tails(tmp_path, config):
    config, admission, norm = dense_fixture(tmp_path, config)
    digest = sha256(admission/'positive_pool.jsonl')
    report = prepare_positive_fm(admission, config, tmp_path/'dense', all_train=True)
    records, pairs, _ = audit_prepared(tmp_path/'dense', config)
    by_segment = {k: sorted([r for r in records if r.source['segment_id']==k],
                           key=lambda r:r.source['segment_row']) for k in ('progress','short','auto','cap')}
    assert [r.source['segment_row'] for r in by_segment['progress']] == list(range(14))
    assert not by_segment['short']
    assert [r.source['segment_row'] for r in by_segment['auto']] == list(range(12))
    assert [r.source['segment_row'] for r in by_segment['cap']] == list(range(10,23))
    assert all(r.source['fm_repeated_steps']==0 for r in by_segment['progress'])
    assert report['positive_windows']==39 and report['training_start_stride']==1
    assert report['matching_state_stride']==10 and pairs==[] and not report['validation_enabled']
    # Overlap adds supervision, not new recorded environment frames.
    assert report['unique_real_frames']==23+12+13
    assert report['supervised_action_slots']==390
    assert report['repeated_action_steps']==90
    assert sha256(admission/'positive_pool.jsonl')==digest
    source_audit = audit_source_alignment(records, norm)
    assert source_audit['max_real_action_error']==0
    assert source_audit['original_parquets']==3
    assert all(r.split=='train' for r in records)
    middle = by_segment['progress'][3]
    assert middle.start_step==103 and middle.image_refs['front']['row']==3
    assert middle.proprio[0]==pytest.approx(.003)
    raw = np.asarray([[j*.01]*7 for j in range(3,13)],np.float32)
    np.testing.assert_array_equal(middle.payload()['actions'][:,:7],QuantileAssets(norm).normalize('actions',raw))
    last = by_segment['cap'][-1]
    assert last.source['fm_real_steps']==1 and last.source['fm_repeated_steps']==9
    assert last.payload()['valid_mask'][:,:7].all()
    assert not last.payload()['valid_mask'][:,7:].any()


def test_dense_rejects_missing_manifest_and_changed_cap_anchor(tmp_path, config):
    config, admission, _ = dense_fixture(tmp_path, config)
    suffix = json.loads((admission/'cap_success_suffixes.jsonl').read_text())
    suffix['segment_row_range'][0]=0
    write_jsonl(admission/'cap_success_suffixes.jsonl',[suffix])
    with pytest.raises(ValueError, match='scope boundaries'):
        prepare_positive_fm(admission, config, tmp_path/'bad')
    (tmp_path/'sparse/input_manifest.jsonl').unlink()
    with pytest.raises(ValueError, match='input_manifest'):
        prepare_positive_fm(admission, config, tmp_path/'missing')


def test_dense_audit_rejects_holes_and_source_manifest_edits(tmp_path, config):
    from recovery_pair.positive_fm import audit_positive_fm
    config, admission, norm = dense_fixture(tmp_path, config)
    prepare_positive_fm(admission, config, tmp_path/'dense', all_train=True)
    records, _, _ = audit_prepared(tmp_path/'dense',config)
    with pytest.raises(ValueError, match='Dense starts'):
        audit_positive_fm(records[1:],[],config)
    manifest = tmp_path/'sparse/input_manifest.jsonl'
    manifest.write_text(manifest.read_text()+'\n')
    with pytest.raises(ValueError, match='manifest changed'):
        audit_source_alignment(records,norm)


def test_dense_never_crosses_local_label_or_event_boundary(tmp_path, config):
    config, admission, norm = dense_fixture(tmp_path, config, label_cuts=True)
    prepare_positive_fm(admission, config, tmp_path/'dense', all_train=True)
    records, _, _ = audit_prepared(tmp_path/'dense',config)
    progress = [r for r in records if r.source['segment_id']=='progress']
    assert [r.source['segment_row'] for r in progress]==[0,1,2,12,13]
    assert {r.event_id for r in progress}=={'event-0','event-12'}
    assert all(r.source['fm_repeated_steps']==0 for r in progress)
    assert audit_source_alignment(records,norm)['max_real_action_error']==0
