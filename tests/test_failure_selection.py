import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('failure_selection',Path(__file__).resolve().parents[1]/'select_failure_episodes.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_balance_covers_tasks_before_repeating_and_ignores_later_success():
    rows=[]
    for task,total in [('a',8),('b',2),('c',2)]:
        for trial in range(total):
            rows.append(dict(suite='suite',instruction=task,source_episode=f'{task}/{trial}',
                original_success=False,trajectory_exists=True,original_trajectory_complete=True,
                converted_success=True,replay_success=True))
    result=module.select(rows,5,42)
    assert len({r['instruction'] for r in result[:3]})==3
    assert max(r['within_task_selection_round'] for r in result)<=2
    assert result==module.select(list(reversed(rows)),5,42)


def test_unknown_or_successful_outcome_is_not_labelled_failure(tmp_path):
    import json
    for trial,success in enumerate([False,True,None]):
        path=tmp_path/'suite/task_00'/f'trial_{trial:02d}'/'report.json';path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(success=success,used_capx=True,suite='suite',task_id=0,
            trial_id=trial,instruction='task',steps_by_owner={'pi05':10,'capx':0})))
    rows=module.inventory(tmp_path)
    assert len(rows)==1 and rows[0]['original_success'] is False
    assert rows[0]['failure_kind']=='cap_launch_failed_vla_only'


def test_spare_slots_fill_underrepresented_suites_after_covering_every_task():
    from collections import Counter
    rows=[]
    for suite,capacities in [('libero_10',[10,7,3,1,4,6,6,7]),
                             ('libero_goal',[2,3,2,1]),('libero_spatial',[2])]:
        for task,capacity in enumerate(capacities):
            for trial in range(capacity):
                rows.append(dict(suite=suite,instruction=str(task),
                    source_episode=f'{suite}/{task}/{trial}',original_success=False,
                    trajectory_exists=True,original_trajectory_complete=True))
    result=module.select(rows,20,42)
    assert len({(r['suite'],r['instruction']) for r in result[:13]})==13
    task_counts=Counter((r['suite'],r['instruction']) for r in result)
    assert max(task_counts.values())==3
    assert sum(n==3 for n in task_counts.values())==1
    assert Counter(r['suite'] for r in result)==dict(libero_10=10,libero_goal=8,libero_spatial=2)
    assert result==module.select(list(reversed(rows)),20,42)


def test_cli_combines_batches_without_counting_overlapping_roots_twice(tmp_path):
    import json
    import subprocess
    import sys
    roots=[tmp_path/'first',tmp_path/'second']
    for root,suite in zip(roots,['libero_goal','libero_spatial']):
        path=root/suite/'task_00/trial_00/report.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(success=False,used_capx=True,suite=suite,
            task_id=0,trial_id=0,instruction=suite,steps_by_owner={'capx':10})))
    output=tmp_path/'combined'
    args=[sys.executable,str(Path(module.__file__)),
          '--results',str(roots[0]),'--results',str(roots[1]),'--results',str(roots[0]),
          '--output',str(output),'--count','20']
    subprocess.run(args,check=True,capture_output=True,text=True)
    result=json.loads((output/'selection.json').read_text())
    assert result['summary']['available_original_failed_episodes']==2
    assert result['summary']['selected_tasks']==2
    assert len({r['source_episode'] for r in result['selected']})==2
