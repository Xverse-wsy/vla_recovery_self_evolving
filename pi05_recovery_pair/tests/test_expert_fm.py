from dataclasses import replace
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from recovery_pair.batches import EpisodeEventSampler
from recovery_pair.data import write_jsonl
from recovery_pair.features import sha256
from recovery_pair.matching import FrozenMatcher
from recovery_pair.pipeline import audit_prepared
from recovery_pair.positive_fm import prepare_positive_fm,audit_source_alignment
from test_dense_fm import dense_fixture


def expert_fixture(tmp):
    root=tmp/'expert40'
    (root/'meta').mkdir(parents=True)
    (root/'data').mkdir()
    # Metadata deliberately describes the full source, while episodes.json is sparse.
    (root/'meta/info.json').write_text(json.dumps(dict(total_episodes=1000,fps=10,
        chunks_size=1000,data_path='data/episode_{episode_index:06d}.parquet')))
    (root/'episodes.json').write_text('[2, 807]')
    write_jsonl(root/'meta/tasks.jsonl',[dict(task_index=i,task='expert task '+str(i)) for i in (1,9)])
    for eid,task,n in [(2,1,13),(807,9,5)]:
        pq.write_table(pa.Table.from_pylist([dict(
            state=[j*.001,0,0,0,0,0,.02,.02],actions=[.01*j]*7,
            image={'bytes':b'unchanged-front','path':None},wrist_image={'bytes':b'unchanged-wrist','path':None},
            task_index=task,episode_index=eid,frame_index=j) for j in range(n)]),
            root/f'data/episode_{eid:06d}.parquet')
    return root


def test_experts_are_dense_train_only_and_preserve_sparse_selection(tmp_path,config):
    config,admission,norm=dense_fixture(tmp_path,config)
    experts=expert_fixture(tmp_path)
    originals={str(p):sha256(p) for p in experts.rglob('*') if p.is_file()}
    admission_sha=sha256(admission/'positive_pool.jsonl')
    out=tmp_path/'with_experts'
    report=prepare_positive_fm(admission,config,out,all_train=True,
                              expert_dataset=experts,expected_expert_episodes=2)
    records,pairs,_=audit_prepared(out,config)
    selected=[r for r in records if r.controller=='expert']
    assert len(selected)==18 and report['positive_windows']==57
    assert report['expert_episodes']==2 and report['expert_windows']==18 and not report['expert_in_matching']
    assert {r.source['source_episode_index'] for r in selected}=={2,807}
    assert all(r.split=='train' and r.bank_class=='D_expert_positive' and
               r.label_basis=='original_expert_demonstration' for r in selected)
    episode=sorted([r for r in selected if r.source['source_episode_index']==2],key=lambda r:r.start_step)
    assert [r.start_step for r in episode]==list(range(13))
    assert episode[1].start_time_s==.1 and episode[1].spec.control_hz==20
    assert episode[1].image_refs['front']['row']==1 and episode[1].proprio[0]==pytest.approx(.001)
    assert episode[-1].source['fm_repeated_steps']==9
    tail=episode[-1].payload()
    assert tail['valid_mask'][:,:7].all() and not tail['valid_mask'][:,7:].any()
    assert not tail['real_action_mask'][1:].any()
    np.testing.assert_array_equal(tail['actions'][1:,:7],np.repeat(tail['actions'][:1,:7],9,axis=0))
    assert all(not r.source['fm_repeated_steps'] for r in records
               if r.bank_label.get('source_category')=='vla_progress_candidate')
    assert audit_source_alignment(records,norm)['max_real_action_error']==0
    assert {p:sha256(p) for p in originals}==originals
    assert sha256(admission/'positive_pool.jsonl')==admission_sha
    with pytest.raises(ValueError,match='training-only'):
        FrozenMatcher(records,config)
    sampler=EpisodeEventSampler(records,pairs,replace(config,fm_batch=64))
    plan=sampler.draw()
    by_id={r.sample_id:r for r in records}
    assert sum(by_id[sid].controller=='code' for sid in plan['fm_ids'])==16
    assert any(by_id[sid].controller=='expert' for sid in plan['fm_ids'])
    assert any(by_id[sid].controller=='VLA' for sid in plan['fm_ids'])
    assert not plan['pairs']
    state=sampler.state_dict()
    restored=EpisodeEventSampler(records,[],replace(config,fm_batch=64))
    restored.load_state_dict(state)
    assert restored.draw()==sampler.draw()


def test_experts_reject_wrong_selection_and_preserved_splits(tmp_path,config):
    config,admission,_=dense_fixture(tmp_path,config)
    experts=expert_fixture(tmp_path)
    with pytest.raises(ValueError,match='--all-train'):
        prepare_positive_fm(admission,config,tmp_path/'bad_split',expert_dataset=experts)
    with pytest.raises(ValueError,match='unique expected episode IDs'):
        prepare_positive_fm(admission,config,tmp_path/'bad_count',all_train=True,expert_dataset=experts)
    (experts/'episodes.json').write_text('[2,2]')
    with pytest.raises(ValueError,match='unique expected episode IDs'):
        prepare_positive_fm(admission,config,tmp_path/'duplicate',all_train=True,
                            expert_dataset=experts,expected_expert_episodes=2)


def test_expert_metadata_changes_and_negative_roles_are_rejected(tmp_path,config):
    config,admission,norm=dense_fixture(tmp_path,config)
    experts=expert_fixture(tmp_path)
    out=tmp_path/'with_experts'
    prepare_positive_fm(admission,config,out,all_train=True,expert_dataset=experts,expected_expert_episodes=2)
    records,_,_=audit_prepared(out,config)
    r=next(r for r in records if r.controller=='expert')
    r.bank_class='D_expert_negative'
    with pytest.raises(ValueError,match='positive and explicitly training-only'):
        r.validate()
    r.bank_class='D_expert_positive'
    (experts/'episodes.json').write_text('[807,2]')
    with pytest.raises(ValueError,match='expert source metadata changed'):
        audit_source_alignment(records,norm)
