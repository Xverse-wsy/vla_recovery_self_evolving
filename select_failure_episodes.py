"""Select original failed CAP-intervention episodes with broad task coverage.

Only original report success=False determines the failure label. Conversion,
independent replay and recovery verification are deliberately not consulted.
This writes a selection manifest, not converted training trajectories.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


POLICY = 'original_failure_task_suite_balanced_v2'


def inventory(results):
    rows = []
    for path in sorted(Path(results).glob('*/task_*/trial_*/report.json')):
        data = path.read_bytes()
        report = json.loads(data)
        # Missing/unknown outcomes are not silently labelled failure.
        if report.get('success') is not False or report.get('used_capx') is not True:
            continue
        owners = report.get('steps_by_owner') or {}
        handoff = report.get('handoff') or {}
        trajectory = report.get('trajectory') or {}
        cap_steps = int(owners.get('capx', 0))
        rows.append(dict(source_episode=str(path.parent.resolve()),
            source_report_sha256=hashlib.sha256(data).hexdigest(),
            suite=report['suite'], evaluation_task_id=report['task_id'],
            trial_id=report['trial_id'], instruction=report['instruction'],
            original_success=False, failure_label_basis='original_execution_report',
            original_outcome=report.get('outcome'),
            original_steps=report.get('steps'), vla_steps=int(owners.get('pi05', 0)),
            cap_steps=cap_steps, handoff_step=handoff.get('step'),
            trajectory_exists=(path.parent/'trajectory/manifest.json').is_file(),
            original_trajectory_complete=trajectory.get('complete'),
            failure_kind='cap_executed_task_not_completed' if cap_steps else 'cap_launch_failed_vla_only',
            conversion_result_not_used=True, replay_result_not_used=True))
    return rows


def select(rows, count=20, seed=42):
    if count < 1:
        raise ValueError('count must be positive')
    groups = defaultdict(list)
    for row in rows:
        if row.get('original_success') is not False:
            raise ValueError('Only explicit original failures can enter this pool')
        groups[(row['suite'], ' '.join(row['instruction'].split()))].append(row)
    if len({r['source_episode'] for r in rows}) != len(rows):
        raise ValueError('Duplicate original episodes')

    def tie(value):
        return hashlib.sha256((str(seed)+'|'+str(value)).encode()).hexdigest()

    # Prefer an intact original recording within a task. This is NOT a
    # conversion/replay check and does not change the original failure label.
    for group in groups.values():
        group.sort(key=lambda r: (not r.get('trajectory_exists', False),
            r.get('original_trajectory_complete') is not True,
            tie(r['source_episode'])))
    counts, suite_counts, selected = Counter(), Counter(), []
    # Cover each available task before any repeat. Then prioritize the least
    # represented available suite; within that suite prefer the least repeated
    # task. A scarce suite may need a third example of one task even while a
    # dominant suite still has tasks represented once. Report actual task counts.
    while len(selected) < min(count, len(rows)):
        uncovered=[key for key in groups if counts[key]==0]
        eligible=uncovered or [key for key in groups if counts[key]<len(groups[key])]
        suite=min({key[0] for key in eligible},
                  key=lambda s:(suite_counts[s],tie(s)))
        key=min((key for key in eligible if key[0]==suite),
                key=lambda key:(counts[key],tie(key)))
        row = dict(groups[key][counts[key]])
        counts[key] += 1
        suite_counts[suite] += 1
        row.update(selection_rank=len(selected)+1,
                   within_task_selection_round=counts[key])
        selected.append(row)
    return selected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,action='append',required=True,
                        help='Original eval/results root; repeat to include additional batches')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--count',type=int,default=20)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    # Multiple roots may overlap. Deduplicate the SAME original path, without
    # merging distinct recorded rollouts of the same task/initial scene.
    rows=list({r['source_episode']:r for root in args.results for r in inventory(root)}.values())
    picked=select(rows,args.count,args.seed)
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    tasks=lambda values:len({(r['suite'],' '.join(r['instruction'].split())) for r in values})
    summary=dict(policy=POLICY,requested_episodes=args.count,
        scanned_results_roots=[str(p.resolve()) for p in args.results],
        available_original_failed_episodes=len(rows),available_tasks=tasks(rows),
        available_by_suite=dict(Counter(r['suite'] for r in rows)),
        selected_episodes=len(picked),selected_tasks=tasks(picked),
        selected_by_suite=dict(Counter(r['suite'] for r in picked)),
        max_episodes_per_task=max(Counter((r['suite'],r['instruction']) for r in picked).values(),default=0),
        selection_priorities=['task_coverage','suite_balance','within_suite_task_balance'],
        selected_zero_cap_action_episodes=sum(r['cap_steps']==0 for r in picked),
        selected_incomplete_original_recordings=sum(r['original_trajectory_complete'] is not True for r in picked),
        failure_label='original_execution_success_false',conversion_or_replay_required=False,
        output_kind='selection_manifest_not_converted_dataset',seed=args.seed)
    selection=dict(summary=summary,selected=picked)
    (out/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n')
    (out/'all_original_failures.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    with (out/'selected.csv').open('w',newline='') as f:
        fields=list(picked[0]) if picked else ['source_episode']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(picked)
    with (out/'task_distribution.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['suite','instruction','available','selected'])
        total=Counter((r['suite'],r['instruction']) for r in rows)
        chosen=Counter((r['suite'],r['instruction']) for r in picked)
        for key,n in sorted(total.items()):writer.writerow([*key,n,chosen[key]])
    (out/'README.md').write_text('# 原始失败轨迹按任务均衡选择\n\n'
        '标签只取原始 report.json 的 success=false，且 used_capx=true。'
        '不使用转换或独立回放结果，不生成新的仿真动作。先每任务一条，再优先补当前数量最少的suite；'
        '同suite内优先补入选次数最少且还有候选的任务。\n\n'
        '仅保存选择清单和源路径；不是新的可训练 CAP 转换数据集。'
        'CAP 启动失败而实际 CAP 动作为零的条目明确标为 vla_only，不能声称存在 CAP 执行动作。'
        '原始轨迹保存是否完整另列字段，不等同于任务成败。\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()
