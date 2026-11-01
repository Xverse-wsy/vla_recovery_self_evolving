"""Stage2-only evaluation merge/compare rejects mismatched scenarios."""
import json
from pathlib import Path
import sys
import pytest

root=Path(__file__).resolve().parents[2]
if not (root/'stage2_eval_jobs.py').exists():root=root/'pi05_ablation'
sys.path.insert(0,str(root))
from stage2_eval_jobs import merge, save
from stage2_compare import compare


def evaluation(folder,label,wins):
    run=dict(label=label,checkpoint='/weights/'+label,suites=['libero_object'],tasks=[0,1],trials=1,
        flow_steps=3,action_chunk=5,seed=0,max_steps=12,save_video=False,norm_stats='/norm.json',
        training_initials_sha256=None,norm_sha256='same',benchmark='standard_LIBERO',
        wcm_enabled=False,capx_enabled=False,scope='validation_subset')
    save(folder/'parallel.json',dict(run,num_shards=2))
    for rank in range(2):
        worker=folder/'workers'/f'worker_{rank:03d}'
        save(worker/'run.json',dict(run,shard=dict(index=rank,count=2)))
        save(worker/'status.json',dict(state='complete'))
        save(worker/'episodes/libero_object'/f'task_{rank:02d}/trial_00.json',
             dict(state='complete',suite='libero_object',task_id=rank,trial_id=0,steps=12,step_limit=12,
                  success=rank<wins,initial_state_sha256='scene'+str(rank),rollout_training_init_overlap=None))
    return merge(folder,require_complete=True)


def test_merge_two_shards_unknown_overlap_and_same_scene_comparison(tmp_path):
    folders={label:tmp_path/label for label in ('fewshot','stage1','stage2')}
    for i,(label,folder) in enumerate(folders.items()):
        result=evaluation(folder,label,i)
        assert result['summary']['episodes']==2
        assert result['summary']['suites']['libero_object']['rollout_training_init_overlap_unknown']==2
    with pytest.raises(ValueError,match='full 2000'):compare(folders,tmp_path/'bad')
    result=compare(folders,tmp_path/'comparison',allow_subset=True)
    assert [result[k]['success_rate'] for k in folders]==[0.,.5,1.]
    path=folders['stage2']/'episodes/libero_object/task_00/trial_00.json'
    row=json.loads(path.read_text());row['initial_state_sha256']='wrong';save(path,row)
    with pytest.raises(ValueError,match='Different initial state'):compare(folders,tmp_path/'mismatch',True)


def test_merge_rejects_changed_normalization_between_workers(tmp_path):
    evaluation(tmp_path/'eval','stage2',1)
    path=tmp_path/'eval/workers/worker_001/run.json'
    row=json.loads(path.read_text());row['norm_sha256']='changed';save(path,row)
    with pytest.raises(AssertionError,match='different weights'):merge(tmp_path/'eval',True)
