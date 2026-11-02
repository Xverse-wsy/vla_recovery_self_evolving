"""Publish the 20 accepted long-recovery OSC trajectories as LeRobot data."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing
from pathlib import Path
import sys

ROOT=Path('/home/wenchaoxu/wsy/vla-recovery-posttrain-loop')
sys.path.insert(0,str(ROOT/'training_configs/pi05_matched17_compensation_fill_20261028'))
import publish_successes as publication
import pyarrow.parquet as pq

exporter=publication.exporter
WORK=ROOT/'trajectory_conversion_tests/pi05_long_recovery20_20261028'
OUT=ROOT/'dataset_subsets/pi05_long_recovery20_20261028'
PRIOR=ROOT/'dataset_subsets/pi05_matched17_compensated_success_20261028'


def read(p):
    return json.loads(Path(p).read_text())


def export_one(item):
    report=read(item['report']); checked=read(item['independent_report'])
    assert report['success'] and checked['success'] and report['source_steps']>1200
    assert checked['max_sim_state_error']<1e-8
    assert checked['source_report_sha256']==exporter.sha(Path(item['report']))
    assert checked['action_record_sha256']==exporter.sha(Path(item['report']).parent/'states_actions.npz')
    eid=report['episode_index']
    cached=PRIOR/f'candidate_episodes/episode_{eid:06d}'
    if item['reused'] and (cached/'status.json').exists():
        status=read(cached/'status.json')
        assert status['accepted_for_training'] and status['replay_report']==item['report']
        assert exporter.sha(cached/'episode.parquet')==status['sha256']
        assert pq.read_metadata(cached/'episode.parquet').num_rows==report['executed_steps']
        entry=dict(parquet=cached/'episode.parquet',record=item['record'],replay_report=item['report'],
            action_provenance=str(Path(item['report']).parent/'action_provenance.jsonl'),
            added_compensation_steps=report['added_compensation_steps'],stats=read(cached/'stats.json'))
        print('REUSED_EXISTING_PARQUET',eid,flush=True)
    else:
        schema=pq.read_schema(publication.BASE_DATA/'data/chunk-000/episode_000000.parquet')
        entry,status=exporter.export_candidate(report,item['record'],Path(item['replay']),OUT,schema)
        assert status['accepted_for_training']
    entry.update(independent_report=item['independent_report'],replacement=False,reused=item['reused'])
    return entry


def main():
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers',type=int,default=3)
    args=ap.parse_args()
    collection=read(WORK/'summary.json')
    assert collection['state']=='complete' and collection['accepted_count']==20
    selected=collection['accepted']
    assert len(selected)==20
    OUT.mkdir(parents=True,exist_ok=True)
    entries=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(export_one,item) for item in selected]
        for future in as_completed(jobs):
            entries.append(future.result())
            exporter.save_json(OUT/'status.json',dict(state='exporting',completed=len(entries),target=20,training_started=False))
    entries.sort(key=lambda x:(x['record']['suite'],x['record']['evaluation_task_id'],x['record']['trial_id']))
    audit=publication.audit_native(entries)
    assert audit['episodes']==20
    schema=pq.read_schema(publication.BASE_DATA/'data/chunk-000/episode_000000.parquet')
    info=read(publication.BASE_DATA/'meta/info.json')
    tasks=exporter.rows(publication.BASE_DATA/'meta/tasks.jsonl')
    dataset=publication.existing_or_publish(OUT/'recovery_success20_lerobot',entries,schema,info,tasks)
    rows=[]
    selection_by_id={Path(x['report']).as_posix():x for x in selected}
    for eid,entry in enumerate(entries):
        item=selection_by_id[entry['replay_report']]
        report=read(entry['replay_report']); rec=entry['record']
        rows.append(dict(exported_episode_index=eid,source_episode_index=rec['episode_index'],
            suite=rec['suite'],task_id=rec['evaluation_task_id'],trial_id=rec['trial_id'],
            original_total_steps=report['source_steps'],converted_success_steps=report['executed_steps'],
            pi05_steps=report['prefix_steps'],converted_recovery_steps=report['executed_steps']-report['prefix_steps'],
            added_compensation_steps=report['added_compensation_steps'],reused=item['reused'],
            instruction=rec['instruction'],original_episode=rec['source_record']['source_episode'],
            converted_report=entry['replay_report'],independent_report=entry['independent_report'],
            video=str(Path(entry['replay_report']).parent/'execution.mp4')))
    assert len({(x['suite'],x['task_id'],x['trial_id']) for x in rows})==20
    assert all(x['original_total_steps']>1200 for x in rows)
    exporter.save_json(OUT/'selection.json',dict(state='complete',episodes=rows,original_step_threshold=1200))
    with (OUT/'selection.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result=dict(state='validating',target=20,accepted_recoveries=20,reused=collection['reused_count'],
        newly_converted=collection['new_success_count'],datasets=[dataset],method='goal_encoding',
        original_total_steps_strictly_greater_than=1200,failed_candidates_in_training=0,
        source_unchanged=True,training_started=False)
    exporter.save_json(OUT/'summary.json',result)
    exporter.save_json(OUT/'status.json',result)
    publication.validator.EXPORT=OUT
    loaders=publication.validator.audit_loader()
    result['state']='complete'
    exporter.save_json(OUT/'summary.json',result)
    exporter.save_json(OUT/'validation_report.json',dict(**result,native_validation=audit,loaders=loaders))
    exporter.save_json(OUT/'status.json',result)
    (OUT/'README.md').write_text(
        '# 20 successful OSC conversions of long CAP-X recoveries\n\n'
        'Each original complete recovery succeeded and had >1200 total control actions, '
        'including its VLA prefix. Exactly 20 unique source trajectories passed the '
        'goal_encoding conversion and independent saved-action replay. Existing '
        'conversion outcomes were reused. Multiple trials of one task are allowed.\n\n'
        'The converter retains original FK absolute goals, changes joint control to '
        '7D OSC, and inserts bounded actions when the CURRENT replay pose cannot '
        'express the old goal in one legal command. Drift can also trigger these '
        'insertions. No strict 2mm arrival wait or new CAP-X program generation. '
        'Task success does not certify identical movement to the source episode.\n\n'
        'LeRobot v2.1: two lossless RGB views, 8D state, 7D actions, fresh observations '
        'for inserted controls. Native physics runs at 20Hz; fps=10 remains the '
        'existing logical indexing convention, without resampling. Preserve '
        'original RLinf normalization and horizon-10 within-episode padding.\n\n'
        'selection.json/csv records original and converted step counts, insertion '
        'counts, task/trial identifiers, programs/action audit paths and videos. '
        'Failed experiments remain outside training data. No training was started.\n')
    print('LONG20_DATASET_VALIDATED',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
