"""Separate Stage2 pool; dense independent starts, explicit experts, real masks."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from recovery_pair.data import build_windows,QuantileAssets
from recovery_pair.features import sha256
from recovery_pair.matching import FrozenMatcher
from recovery_pair.positive_fm import prepare_positive_fm
from recovery_pair.schema import save_windows
from recovery_pair.stage2_matching import (positive_references,dense_negative_queries,
    candidate_index,evaluate,summarize,run)
from test_dense_fm import dense_fixture
from test_expert_fm import expert_fixture


def negative_fixture(tmp,config,norm):
    path=tmp/'negative.parquet'
    pq.write_table(pa.Table.from_pylist([dict(state=[j*.001,0,0,0,0,0,.02,.02],
        actions=[-.5]*7,image={'bytes':b'image','path':None},wrist_image={'bytes':b'wrist','path':None})
        for j in range(8)]),path)
    entry=dict(segment_id='negative',parquet=str(path),length=8,parent_start_step=87,
        controller='VLA',episode_id='failed_episode',task_id='negative',initial_group_id='initial_neg',
        event_id='warning',instruction='place block',split='validation',
        state_layout='eef_xyz_axisangle_gripper2',bank_class='D_vla_negative',
        bank_label={'source_category':'vla_wcm_warning'},annotations=[],source={})
    records=build_windows([entry],QuantileAssets(norm),config,tmp/'negative_sparse')
    assert len(records)==1  # old stride10 misses the remaining 7 observation starts
    return records


def feature(r):
    r.visual_features=dict(encoder_sha256='synthetic',calibration_id='same',
                          views={'front':[1.,0.],'wrist':[1.,0.]})


def test_negative_dense_alignment_real_tails_and_no_inherited_match(tmp_path,config,window):
    norm=tmp_path/'norm.json'
    norm.write_text(json.dumps({'norm_stats':{k:{'q01':[-1]*d,'q99':[1]*d} for k,d in [('state',8),('actions',7)]}}))
    old=negative_fixture(tmp_path,config,norm)
    source_sha=sha256(tmp_path/'negative.parquet')
    negatives,scopes=dense_negative_queries(old,tmp_path/'new')
    assert len(negatives)==8 and len(scopes)==1
    assert [r.start_step for r in negatives]==list(range(87,95))
    assert [r.payload()['valid_mask'].any(-1).sum() for r in negatives]==list(range(8,0,-1))
    for i,r in enumerate(negatives):
        assert r.image_refs['front']['row']==i and r.proprio[0]==pytest.approx(i*.001)
        np.testing.assert_allclose(r.payload()['actions'][:8-i,:7],-.5,atol=1e-6)
        assert not r.payload()['valid_mask'][8-i:].any() and not r.source['stage2_match_inherited']
        assert r.source['stage2_original_split']=='validation' and r.split=='train'
        feature(r)
    positive=window('positive',position=0.)
    positive.spec=negatives[0].spec
    positive.controller='expert'
    positive.bank_class='D_expert_positive'
    positive.source=dict(expert_training_only=True)
    positive.bank_label=dict(source_category='original_expert_demonstration')
    positive.validate()
    feature(positive)
    positive.proprio[6:]=[.02,.02]
    positive.standardized_proprio=positive.proprio.copy()
    candidates=candidate_index(negatives,[positive],config)
    # One neighbor lacks a match: do not inherit a previous start's reference.
    candidates[negatives[1].sample_id]=[]
    pairs,rows=evaluate(negatives,[positive],candidates,config)
    report=summarize(negatives,[positive],pairs,rows)
    assert report['paired_negative_windows']==5  # starts0,2,3,4,5; tails2/1 excluded
    assert report['paired_negative_segments']==report['paired_negative_episodes']==1
    assert report['paired_windows_with_expert_reference']==5
    assert report['unique_effective_negative_action_frames']==8
    assert report['effective_action_slots_union_per_window']>8  # overlapping windows aren't new frames
    assert rows[1]['training_references']==0
    assert sha256(tmp_path/'negative.parquet')==source_sha


def test_experts_only_enabled_in_separate_pool_and_no_repeated_tail_pairing(tmp_path,config):
    c,admission,norm=dense_fixture(tmp_path,config)
    experts=expert_fixture(tmp_path)
    prepared=tmp_path/'stage1'
    prepare_positive_fm(admission,c,prepared,all_train=True,expert_dataset=experts,expected_expert_episodes=2)
    from recovery_pair.pipeline import audit_prepared
    originals,_,_=audit_prepared(prepared)
    before={r.action_path:sha256(r.action_path) for r in originals}
    refs=positive_references(originals,tmp_path/'stage2_refs')
    assert len(refs)==len(originals) and sum(r.controller=='expert' for r in refs)==18
    expert_tail=next(r for r in refs if r.controller=='expert' and r.source['fm_real_steps']==1)
    assert expert_tail.payload()['valid_mask'].any(-1).sum()==1
    assert not expert_tail.payload()['actions'][1:].any()
    assert expert_tail.source['stage2_expert_reference_authorized']
    with pytest.raises(ValueError,match='training-only'): FrozenMatcher(refs,c)
    assert {r.action_path:sha256(r.action_path) for r in originals}==before


def test_synthetic_end_to_end_matching_never_updates_stage1(tmp_path,config):
    c,admission,norm=dense_fixture(tmp_path,config)
    experts=expert_fixture(tmp_path)
    prepared=tmp_path/'stage1'
    prepare_positive_fm(admission,c,prepared,all_train=True,expert_dataset=experts,expected_expert_episodes=2)
    neg=negative_fixture(tmp_path,c,norm)
    windows=tmp_path/'negative_windows.jsonl'
    save_windows(windows,neg)
    immutable={str(p):sha256(p) for root in [prepared,experts] for p in root.rglob('*') if p.is_file()}
    class Encoder:
        def __init__(self,*args): pass
        def extract(self,records,cache,batch):
            for r in records: feature(r)
            return {'synthetic_only':True,'image_embeddings_computed':2*len(records)}
    result=run(prepared,windows,'unused',c,tmp_path/'stage2',device='cpu',encoder_factory=Encoder,expected_expert_episodes=2)
    assert result['negative_query_windows']==8 and result['positive_reference_controllers']['expert']==18
    assert not result['training_started'] and not result['formal_training_prepared']
    assert result['negative_suffix_inheritance'] is False
    assert {p:sha256(p) for p in immutable}==immutable
    with pytest.raises(FileExistsError):
        run(prepared,windows,'unused',c,tmp_path/'stage2',encoder_factory=Encoder)


def test_missing_features_fail_closed(tmp_path,config,window):
    pos=window('pos');neg=window('neg',quality='negative')
    neg.source=dict(segment_id='n',segment_row=0)
    neg.visual_features={}
    with pytest.raises(ValueError,match='Missing candidate features'):
        evaluate([neg],[pos],candidate_index([neg],[pos],config),config)
