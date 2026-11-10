import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from subtask_training.dataset import SubtaskDataset
from subtask_training.training_data import audit_recipe,smoke_indices
from direct_data import build_raw,DirectEpisodes
from settings import load_config


@pytest.fixture
def dataset(tmp_path):
    # Distinct action rows expose accidental repetition, trimming or boundary crossing.
    image_type=pa.struct([('bytes',pa.binary()),('path',pa.string())])
    import io
    from PIL import Image
    buf=io.BytesIO();Image.fromarray(np.zeros((256,256,3),np.uint8)).save(buf,format='PNG')
    raw=np.arange(12*7,dtype=np.float32).reshape(12,7)
    table=pa.table({'image':pa.array([{'bytes':buf.getvalue(),'path':None}]*12,type=image_type),
                    'wrist_image':pa.array([{'bytes':buf.getvalue(),'path':None}]*12,type=image_type),
                    'state':pa.array(np.zeros((12,8),np.float32).tolist()),'actions':pa.array(raw.tolist())})
    (tmp_path/'meta').mkdir();pq.write_table(table,tmp_path/'clip.parquet',row_group_size=4)
    def create(terminal):
        meta=dict(segment_id='s',parquet='clip.parquet',task_instruction='original FULL task',
                  dataset_episode_index=0,endpoint_has_recorded_task_success=terminal,
                  padding_policy='repeat_last' if terminal else 'full_real_only')
        starts=range(12 if terminal else 3)
        windows=[dict(meta,sample_id=f's{i}',start=i,horizon=10,real_steps=min(10,12-i),repeated_steps=max(0,i-2),
                      tail_policy=meta['padding_policy']) for i in starts]
        (tmp_path/'meta/provenance.jsonl').write_text(json.dumps(meta)+'\n')
        (tmp_path/'windows.jsonl').write_text(''.join(json.dumps(w)+'\n' for w in windows))
        (tmp_path/'build_state.json').write_text(json.dumps(dict(state='complete',training_windows=len(windows))))
        return SubtaskDataset(tmp_path),raw
    return create


def test_intermediate_uses_all_and_only_full_real_windows(dataset):
    ds,raw=dataset(False)
    assert len(ds)==3
    for i in range(3):np.testing.assert_array_equal(ds[i]['actions'],raw[i:i+10])
    ds.windows[0]['start']=3
    with pytest.raises(ValueError):ds[0]


def test_only_final_success_may_repeat_and_keeps_parent_instruction(dataset):
    ds,raw=dataset(True)
    np.testing.assert_array_equal(ds[-1]['actions'],np.repeat(raw[-1:],10,axis=0))
    assert ds[-1]['prompt']=='original FULL task'
    ds.segments['s']['endpoint_has_recorded_task_success']=False
    with pytest.raises(ValueError):ds[-1]


def test_cannot_enable_repetition_in_middle_by_changing_window_alone(dataset):
    ds,_=dataset(False)
    ds.windows[0].update(start=11,tail_policy='repeat_last',real_steps=1,repeated_steps=9)
    with pytest.raises(ValueError):ds[0]


def test_never_wrap_negative_or_cross_episode(dataset):
    ds,_=dataset(True)
    for index in (-1,12):
        ds.windows[0]['start']=index
        with pytest.raises(ValueError):ds[0]


def test_config_inherits_every_a60_training_setting(monkeypatch):
    monkeypatch.setenv('PI05_ROOT','/tmp/pi05_recipe_test')
    new,base=load_config('a60_subtasks22'),load_config('a60')
    assert audit_recipe(new,base)['model_optimizer_scheduler_runtime_exactly_equal_to_a60']
    assert new.runner.max_epochs==50 and new.actor.global_batch_size==100
    assert new.actor.model.num_steps==5
    new.actor.optim.lr=1e-5
    with pytest.raises(ValueError):audit_recipe(new,base)


def test_legacy_reader_class_is_unchanged_for_existing_sources(monkeypatch):
    seen=[]
    class Sentinel:
        def __init__(self,sources,horizon):seen.append((sources,horizon))
    import direct_data
    monkeypatch.setattr(direct_data,'DirectEpisodes',Sentinel)
    sources=[{'name':'author40'},{'name':'autonomous20'}]
    assert isinstance(build_raw(sources),Sentinel)
    assert seen==[(sources,10)]


def test_smoke_keeps_all_sources_even_when_sorted_uniform_probe_would_drop_new_tail():
    selected=smoke_indices(11058,9864,6632)
    assert len(selected)==len(set(selected))==15
    assert {0,6631,6632,9863,9864,11057}<=set(selected)


def test_non_source_data_runtime_options_must_also_match_a60(monkeypatch):
    monkeypatch.setenv('PI05_ROOT','/tmp/pi05_recipe_test')
    new,base=load_config('a60_subtasks22'),load_config('a60')
    new.data.num_workers=2
    with pytest.raises(ValueError):audit_recipe(new,base)
