from dataclasses import replace
import json
import numpy as np
import pytest
from recovery_pair.matching import FrozenMatcher,pair_mask,numeric_match
from recovery_pair.splits import split_windows,audit_split
from recovery_pair.schema import Window
from recovery_pair.pipeline import audit_prepared,calibrate
from recovery_pair.synthetic import fixture


def test_no_chain_coverage_only_vla_success_is_reference(window,config):
    vla=window('good','VLA','positive',position=0.)
    bad=window('bad','VLA','negative',position=.05)
    c1=window('c1','code','positive',position=.025)
    c2=window('c2','code','positive',position=.05)
    matcher=FrozenMatcher([vla,bad,c1,c2],config)
    assert 'c1' in matcher.coverage and 'c2' not in matcher.coverage
    assert matcher.frozen_state_ids==('good',)
    assert matcher.coverage['c1']['reference']['sample_id']=='good'
    assert matcher.success_matches[0]['sample_id']=='c1'


def test_max3_references_prefer_episodes_and_traceability(window,config):
    negative=window('negative',quality='negative')
    positives=[window('p'+str(i),episode='ep'+str(i//2)) for i in range(10)]
    matcher=FrozenMatcher([negative]+positives,config)
    pair=matcher.build_pairs()[0]
    assert len(pair['references'])==3
    assert len({r['episode_id'] for r in pair['references']})==3
    assert pair['negative_start']['image_refs']
    assert all(r['distances'] and r['thresholds'] for r in pair['references'])


def test_numeric_then_at_most50_visual_candidates(window,config,monkeypatch):
    import recovery_pair.matching as module
    query=window('negative',quality='negative')
    refs=[window('p'+str(i)) for i in range(80)]
    matcher=FrozenMatcher([query]+refs,config)
    original=module.image_distance
    calls=[]
    def image(a,b):
        calls.append(b.sample_id)
        return original(a,b)
    monkeypatch.setattr(module,'image_distance',image)
    assert len(matcher.match(query,refs,config.pairing))==50
    assert len(calls)==50


def test_privileged_metadata_never_gates_admission_or_pairing(window,config):
    pos=window('vla_success')
    code=window('code_success','code','positive')
    neg=window('negative',quality='negative')
    baseline=FrozenMatcher([pos,code,neg],config)
    baseline_ids=[r['sample_id'] for r in baseline.build_pairs()[0]['references']]
    for r in (pos,code,neg):
        r.phase=None if r is code else 'conflicting-stage'
        r.target_object=None if r is neg else 'unrelated-object'
        r.held_object=None
        r.contact_state='conflicting-contact' if r is code else None
        r.gripper_state=None
        r.source['critical_state_verified']=False
        r.progress=None
    code.object_poses={'oracle_object':{'invalid':'must never read'}}
    code.task_id='different-private-task-metadata'
    code.eef_position_m=[100.,100.,100.]
    code.eef_quaternion_xyzw=[1.,0.,0.,0.]
    matcher=FrozenMatcher([pos,code,neg],config)
    assert 'code_success' in matcher.coverage
    assert [r['sample_id'] for r in matcher.build_pairs()[0]['references']]==baseline_ids
    assert matcher.coverage['code_success']['reference']['distances']==baseline.coverage['code_success']['reference']['distances']


def test_numeric_retrieval_reads_only_public_instruction_and_policy_state(window,config):
    a,b=window('a'),window('b')
    class PublicInputsOnly:
        def __init__(self,r):
            for name in ('split','instruction','spec','proprio','standardized_proprio'):
                setattr(self,name,getattr(r,name))
        def __getattr__(self,name):
            raise AssertionError(f'Matching tried to read non-policy input: {name}')
    assert numeric_match(PublicInputsOnly(a),PublicInputsOnly(b),config.pairing)[1] is None
    b.instruction='open the drawer'
    assert numeric_match(a,b,config.pairing)[1]=='split_or_instruction'


@pytest.mark.parametrize('change,reason',[
    ('position','threshold_position_m'),('rotation','threshold_rotation_deg'),('proprio','threshold_proprio_rms')])
def test_numeric_pose_and_gripper_distances_come_from_policy_proprio(window,config,change,reason):
    a,b=window('a'),window('b')
    if change=='position':
        b.proprio[0]=.02
    elif change=='rotation':
        b.proprio[3]=.2
    else:
        b.standardized_proprio[-1]=10.
    assert numeric_match(a,b,config.pairing)[1]==reason


def test_visual_match_ignores_non_camera_auxiliary_features(window,config):
    from recovery_pair.matching import image_distance
    a,b=window('a'),window('b')
    a.visual_features['views']['oracle_segmentation']=[1.,0.]
    b.visual_features['views']['oracle_segmentation']=[-1.,0.]
    distances,reason=image_distance(a,b)
    assert reason is None and set(distances)=={'front','wrist'}
    assert FrozenMatcher([a,b],config).match(a,[b],config.pairing)


@pytest.mark.parametrize('reference_progress',[None,1.0])
def test_progress_and_scorer_do_not_gate_coverage_or_pairing(window,config,reference_progress):
    neg=window('negative',quality='negative')
    neg.progress=0.0
    neg.source.pop('progress_scorer_id')
    vla=window('vla_success')
    vla.progress=None
    vla.source.pop('progress_scorer_id')
    pos=window('positive','code','positive')
    pos.progress=reference_progress
    pos.source['progress_scorer_id']='different-or-unavailable-scorer'
    distances,reason=numeric_match(neg,pos,config.pairing)
    assert reason is None and 'progress' not in distances
    matcher=FrozenMatcher([neg,vla,pos],config)
    assert 'positive' in matcher.coverage
    pairs=matcher.build_pairs()
    assert len(pairs)==1 and 'positive' in {r['sample_id'] for r in pairs[0]['references']}
    ref=pairs[0]['references'][0]
    assert 'progress' not in ref['distances'] and 'progress' not in ref['thresholds']


def test_mask_excludes_shared_prefix_padding_and_untrusted_steps(window,config):
    p,n=window('p'),window('n',quality='negative')
    payload=n.payload()
    payload['actions'][:2]=p.payload()['actions'][:2]
    payload['trusted_error_mask'][9]=False
    np.savez(n.action_path,**payload)
    mask,reason=pair_mask(p,n,config)
    assert reason is None
    assert not mask[:2].any() and not mask[:,7:].any() and not mask[9].any()
    assert mask[2:9,:7].all()
    payload['actions']=p.payload()['actions']
    np.savez(n.action_path,**payload)
    assert pair_mask(p,n,config)[0] is None


def test_split_connected_initialization_and_episode_never_leaks(window):
    records=[window(str(i),episode=str(i//2)) for i in range(20)]
    records[0].initial_group_id=records[3].initial_group_id
    split_windows(records,conservative_task=False)
    audit_split(records)
    assert len({r.split for r in records[:4]})==1
    records[1].split='test' if records[0].split!='test' else 'train'
    with pytest.raises(ValueError):
        audit_split(records)


def test_weak_labels_cannot_be_used_as_negatives(window):
    r=window('n',quality='negative')
    r.quality_evidence={'wcm_alarm':True}
    with pytest.raises(ValueError):
        r.validate()


def test_actual_parquet_window_import_handoff_and_artifact_audit(tmp_path):
    config,records,report=fixture(tmp_path/'fixture')
    assert report['paired_windows']>0
    assert {r.split for r in records}=={'train','validation','test'}
    handoffs=[r for r in records if r.source.get('handoff_observation')]
    assert handoffs and all(r.controller=='VLA' and not r.payload()['valid_mask'].any() for r in handoffs)
    tails=[r for r in records if r.source['segment_row']==10 and r.local_quality=='positive']
    assert tails and all(r.payload()['valid_mask'].sum()==14 for r in tails)
    assert all(r.image_refs['front']['row']==r.source['segment_row'] for r in records)
    pool,pairs,digest=audit_prepared(tmp_path/'fixture/prepared',config)
    assert pool and pairs and digest
    from recovery_pair.schema import load_windows
    bank=tmp_path/'fixture/windows/banks/train'
    assert all(r.controller=='VLA' for r in load_windows(bank/'S_vla_all.jsonl'))
    assert load_windows(bank/'B_positive.jsonl') and load_windows(bank/'B_negative.jsonl')
    assert load_windows(bank/'D_code_positive.jsonl') and load_windows(bank/'D_code_negative.jsonl')
    # Unknown initialization forces a task-group split, even if requested otherwise.
    records[0].uncertain_initialization=True
    split_windows(records,conservative_task=False)
    assert len({r.split for r in records if r.task_id==records[0].task_id})==1


def test_local_quality_change_is_a_hard_window_boundary(tmp_path):
    from recovery_pair.data import read_jsonl,build_windows,QuantileAssets
    config,records,report=fixture(tmp_path/'original')
    entry=read_jsonl(tmp_path/'original/windows/input_manifest.jsonl')[0]
    entry['annotations']=[dict(start=0,end=4,local_quality='positive',quality_evidence={'verified_local_actions':True}),
        dict(start=4,end=12,local_quality='negative',quality_evidence={'verified_local_actions':True,'negative_behavior_confirmed':True})]
    result=build_windows([entry],QuantileAssets(tmp_path/'original/norm_stats.json'),config,tmp_path/'cut')
    pos=next(r for r in result if r.start_step==0)
    assert pos.payload()['valid_mask'].sum()==4*7
    assert all(r.local_quality=='negative' for r in result if r.start_step>=4)


def test_calibration_never_uses_heldout(window,config,tmp_path):
    a,b=window('a'),window('b',split='test')
    with pytest.raises(ValueError,match='validation/test'):
        calibrate([a,b],[dict(query_id='a',reference_id='b',compatible=True,reviewer='tester')],config,tmp_path/'cal')


def test_unset_thresholds_fail_closed(window):
    from recovery_pair.config import Config
    with pytest.raises(ValueError,match='Calibrate'):
        FrozenMatcher([window('a')],Config())


def test_old_prepared_matches_cannot_silently_keep_privileged_rules(tmp_path):
    fixture(tmp_path/'fixture')
    path=tmp_path/'fixture/prepared/report.json'
    report=json.loads(path.read_text())
    report.pop('observation_policy')
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='policy inputs only'):
        audit_prepared(tmp_path/'fixture/prepared')
