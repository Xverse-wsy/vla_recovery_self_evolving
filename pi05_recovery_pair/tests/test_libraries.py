from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from recovery_pair.config import Config
from recovery_pair.data import QuantileAssets,build_windows,manifest_segmented,read_jsonl
from recovery_pair.libraries import BANKS,create_libraries,audit_libraries
from recovery_pair.schema import load_windows
from recovery_pair.synthetic import fixture
from recovery_pair.matching import pair_mask
from recovery_pair.library_build import build_existing_libraries
from recovery_pair.pipeline import prepare,audit_prepared


def source_fixture(tmp_path):
    """Existing classified segments deliberately have no new local reviews."""
    config,_,_=fixture(tmp_path/'synthetic')
    entries=read_jsonl(tmp_path/'synthetic/windows/input_manifest.jsonl')
    for e in entries:
        quality='positive' if e['segment_id'].endswith('positive') else 'negative'
        e['bank_class']=f"D_{'vla' if e['controller']=='VLA' else 'code'}_{quality}"
        e['bank_label']={'basis':'existing_fixture_selection','local_actions_verified':False}
        e['annotations']=[]
    return config,entries,QuantileAssets(tmp_path/'synthetic/norm_stats.json')


def test_existing_four_banks_created_before_calibration_and_without_actions_or_source_edits(tmp_path):
    config,entries,_=source_fixture(tmp_path)
    config=replace(config,coverage=Config().coverage,pairing=Config().pairing)
    original=deepcopy(entries)
    report=create_libraries(entries,config,tmp_path/'libraries')
    assert entries==original
    assert report['matching_performed'] is False
    assert report['banks']=={bank:{'segments':6,'action_steps':72} for bank in BANKS}
    assert all((tmp_path/'libraries/segments'/f'{bank}.jsonl').is_file() for bank in BANKS)
    assert not list((tmp_path/'libraries').rglob('*.npz'))
    assert len(audit_libraries(tmp_path/'libraries'))==24
    with pytest.raises(FileExistsError):
        create_libraries(entries,config,tmp_path/'libraries')


def test_window_libraries_inherit_split_and_use_existing_classes_without_forging_local_evidence(tmp_path):
    config,entries,assets=source_fixture(tmp_path)
    create_libraries(entries,config,tmp_path/'libraries')
    split_entries=audit_libraries(tmp_path/'libraries')
    # Change H; it must not reshuffle already grouped episodes.
    changed=replace(config,horizon=6)
    records=build_windows(split_entries,assets,changed,tmp_path/'windows')
    split_of={e['segment_id']:e['split'] for e in split_entries}
    assert all(r.split==split_of[r.source['segment_id']] for r in records)
    assert all(r.local_quality=='unknown' for r in records)
    assert all(not r.payload()['trusted_error_mask'].any() for r in records)
    for bank in BANKS:
        bank_records=load_windows(tmp_path/'windows/banks'/f'{bank}.jsonl')
        assert len(bank_records)==12 and all(r.bank_class==bank for r in bank_records)
        assert all(r.bank_label['basis']=='existing_fixture_selection' for r in bank_records)
    states=[r for r in records if r.source.get('state_only')]
    assert len(states)==12 and all(r.bank_class is None and r.controller=='VLA' for r in states)
    positive=load_windows(tmp_path/'windows/banks/D_vla_positive.jsonl')[0]
    negative=load_windows(tmp_path/'windows/banks/D_vla_negative.jsonl')[0]
    mask,reason=pair_mask(positive,negative,changed)
    assert reason is None and mask.any()
    assert negative.training_quality=='negative' and negative.label_basis=='existing_segment_class'


def test_preassigned_episode_split_leak_is_rejected(tmp_path):
    config,entries,assets=source_fixture(tmp_path)
    create_libraries(entries,config,tmp_path/'libraries')
    entries=audit_libraries(tmp_path/'libraries')
    duplicate=deepcopy(entries[0])
    duplicate['segment_id']='different-segment-same-parent'
    duplicate['split']='test' if entries[0]['split']!='test' else 'train'
    with pytest.raises(ValueError,match='leaks'):
        build_windows(entries+[duplicate],assets,config,tmp_path/'bad_windows')


def test_existing_selection_ids_map_to_banks_and_conflicting_memberships_rejected(tmp_path):
    root=tmp_path/'source'
    (root/'meta').mkdir(parents=True)
    (root/'provenance/parent_000000').mkdir(parents=True)
    (root/'provenance/parent_000000/segments.json').write_text(json.dumps({'prefix_steps':4}))
    records=[]
    for i,category in enumerate(['vla_progress_candidate','vla_wcm_warning','capx_completed','capx_not_completed']):
        code=i>=2
        records.append(dict(episode_index=i,canonical_task_index=5,task_id=0,trial_id=0,
            suite='libero_10',original_episode='original-parent',instruction='put block',
            category='capx_program' if code else category,owner='capx' if code else 'pi05',
            turn=i if code else None,start=i*2,end=i*2+2,
            completion_label=('完成任务' if i==2 else '没有完成任务') if code else None,
            task_completed_in_segment=i==2,action_mapping=f'provenance/parent_000000/segment_{i:06d}_mapping.jsonl'))
    selection={category:[i] for i,category in enumerate(
        ['vla_progress_candidate','vla_wcm_warning','capx_completed','capx_not_completed'])}
    (root/'selection.json').write_text(json.dumps(selection))
    (root/'meta/info.json').write_text(json.dumps({'chunks_size':1000,'data_path':'episode_{episode_index:06d}.parquet'}))
    (root/'meta/provenance.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in records))
    entries=manifest_segmented(root)
    assert [e['bank_class'] for e in entries]==list(BANKS)
    assert all(e['task_id']=='libero_10/task05' for e in entries)
    assert all(not e['bank_label']['local_actions_verified'] for e in entries)
    selection['capx_completed'].append(3)
    (root/'selection.json').write_text(json.dumps(selection))
    with pytest.raises(ValueError):
        manifest_segmented(root)


def portable_sources(tmp_path):
    config,entries,assets=source_fixture(tmp_path)
    source=tmp_path/'new_machine_data'
    source.mkdir()
    (source/'reviewed_manifest.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in entries))
    (source/'norm.json').write_bytes(Path(assets.path).read_bytes())
    settings=tmp_path/'portable_configs'
    settings.mkdir()
    (settings/'algorithm.json').write_text(json.dumps(config.to_dict()))
    path=settings/'sources.json'
    path.write_text(json.dumps(dict(manifests=['reviewed_manifest.jsonl'],norm_stats='norm.json',
        algorithm_config='algorithm.json',expected_segment_counts={bank:6 for bank in BANKS},
        expected_image_shape=[16,16,3])))
    return path,source


def test_portable_entrypoint_resolves_configs_without_a800_or_cwd_dependencies(tmp_path):
    path,source=portable_sources(tmp_path)
    unrelated=tmp_path/'unrelated_cwd'
    unrelated.mkdir()
    script=Path(__file__).resolve().parents[1]/'scripts/build_existing_libraries.py'
    out=tmp_path/'new_machine_output'
    result=subprocess.run([sys.executable,str(script),'--sources-config',str(path),
        '--data-root',str(source),'--output',str(out)],cwd=unrelated,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report=json.loads((out/'report.json').read_text())
    assert report['state']=='complete' and not report['cuda_used'] and not report['training_started']
    assert report['source_metadata_unchanged']
    assert len(audit_libraries(out/'libraries'))==24
    assert report['segment_libraries']['banks']=={bank:dict(segments=6,action_steps=72) for bank in BANKS}
    assert json.loads((out/'status.json').read_text())['state']=='complete'
    with pytest.raises(FileExistsError):
        build_existing_libraries(path,source,out)


def test_portable_build_reports_wrong_source_inventory_without_training(tmp_path):
    path,source=portable_sources(tmp_path)
    settings=json.loads(path.read_text())
    settings['expected_segment_counts']['D_code_negative']=99
    path.write_text(json.dumps(settings))
    out=tmp_path/'failed_inventory'
    with pytest.raises(ValueError,match='bank counts differ'):
        build_existing_libraries(path,source,out)
    status=json.loads((out/'status.json').read_text())
    assert status['state']=='failed' and not (out/'windows').exists()


def test_existing_four_labels_drive_matching_audit_and_training_without_new_reviews(tmp_path):
    config,entries,assets=source_fixture(tmp_path)
    records=build_windows(entries,assets,config,tmp_path/'classified_windows')
    assert all(r.local_quality=='unknown' for r in records)
    for r in records:
        r.visual_features=dict(encoder_sha256='synthetic_test_encoder',calibration_id='synthetic',
                               views={'front':[1.,0.],'wrist':[1.,0.]})
    report=prepare(records,config,tmp_path/'prepared_existing')
    assert report['matched_cap_success_windows']>0 and report['paired_windows']>0
    pool,pairs,_=audit_prepared(tmp_path/'prepared_existing',config)
    matches=read_jsonl(tmp_path/'prepared_existing/success_matches.jsonl')
    assert all(m['code_start']['bank_class']=='D_code_positive' for m in matches)
    assert all(m['vla_start']['bank_class']=='D_vla_positive' for m in matches)
    assert all(p['negative_label_basis']=='existing_segment_class' for p in pairs)
    assert all(p['error_mask_basis']=='existing_segment_class' for p in pairs)
    assert all(not r.payload()['trusted_error_mask'].any() for r in records)
    # Label provenance stays truthful while the user's existing segment labels train.
    from recovery_pair.trainer import train
    from recovery_pair.toy import ToyFlow
    from recovery_pair.synthetic import loader
    result=train(ToyFlow(config.horizon,config.action_dim,steps=config.flow_steps),loader,
        tmp_path/'prepared_existing',config,tmp_path/'classified_train',max_updates=1)
    assert (tmp_path/'classified_train/update_000001.pt').is_file()


def test_conflicting_existing_and_explicit_local_labels_are_rejected(window):
    record=window('conflict',quality='negative')
    record.bank_class='D_vla_positive'
    with pytest.raises(ValueError,match='conflicts'):
        record.validate()
