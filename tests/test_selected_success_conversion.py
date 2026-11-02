import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    'selected_conversion', Path(__file__).resolve().parents[1] / 'recovery_conversion/convert_selected_success.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(tmp_path, success=True, early=False):
    source = tmp_path / 'inputs'
    programs = source / 'ep/capx'
    programs.mkdir(parents=True)
    output = tmp_path / 'converted'
    (output / 'programs').mkdir(parents=True)
    ranges = [dict(turn=0, start=2, end=3, path=programs / 'turn_00.py'),
              dict(turn=1, start=4, end=5, path=programs / 'turn_01.py')]
    for p in ranges:
        p['path'].write_text('code %d' % p['turn'])
        (output / 'programs' / p['path'].name).write_bytes(p['path'].read_bytes())
    # Repeat source action 2 as an inserted compensation action.
    steps = [1, 2, 2, 3] if early else [1, 2, 2, 3, 4, 5]
    trace = []
    for i, step in enumerate(steps):
        turn = None if step == 1 else 0 if step <= 3 else 1
        trace.append(dict(source_action_step=step, source_turn=turn,
            source_program=None if turn is None else str(programs / ('turn_%02d.py' % turn)),
            copied_program=None if turn is None else 'programs/turn_%02d.py' % turn,
            success_after_action=success and i == len(steps) - 1))
    m.save(output / 'report.json', dict(executed_steps=len(steps), success=success))
    (output / 'action_provenance.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in trace))
    helper = SimpleNamespace(SOURCE=source, program_ranges=lambda ep: ranges)
    rec = dict(source_record={}, action_rows=5, handoff_step=1, path='ep')
    return helper, rec, output


def test_separates_failed_then_successful_code_rounds_and_compensation(tmp_path):
    helper, rec, out = fixture(tmp_path)
    audit = m.boundary_audit(helper, rec, out / 'report.json')
    assert [(s['source_turn'], s['converted_start_action'], s['converted_end_action'])
            for s in audit['converted_program_segments']] == [(None, 1, 1), (0, 2, 4), (1, 5, 6)]
    assert audit['success_program_agrees']
    assert not audit['converted_program_segments'][1]['contains_converted_first_success']
    assert audit['converted_program_segments'][2]['contains_converted_first_success']
    assert not audit['whole_episode_positive_label_assigned']


def test_early_converted_success_does_not_relabel_original_failed_program(tmp_path):
    helper, rec, out = fixture(tmp_path, early=True)
    audit = m.boundary_audit(helper, rec, out / 'report.json')
    assert audit['original_success_turn'] == 1
    assert audit['converted_first_success_turn'] == 0
    assert audit['success_program_agrees'] is False


def test_conversion_failure_is_retained_as_separate_outcome(tmp_path):
    helper, rec, out = fixture(tmp_path, success=False)
    audit = m.boundary_audit(helper, rec, out / 'report.json')
    assert audit['original_success'] is True
    assert audit['converted_success'] is False
    assert audit['converted_first_success_turn'] is None
    assert audit['success_program_agrees'] is None
    assert len(audit['converted_program_segments']) == 3


def test_wrong_code_round_or_modified_program_is_rejected(tmp_path):
    helper, rec, out = fixture(tmp_path)
    (out / 'programs/turn_00.py').write_text('different code')
    with pytest.raises(AssertionError):
        m.boundary_audit(helper, rec, out / 'report.json')


def test_retention_keeps_actions_reports_and_replay_after_old_paths_are_deleted(tmp_path):
    original = tmp_path / 'old'
    original.mkdir()
    m.save(original / 'report.json', dict(success=False))
    (original / 'states_actions.npz').write_bytes(b'negative converted episode' * 50000)
    checked = tmp_path / 'checked.json'
    m.save(checked, dict(success=False))
    retained = m.retain_episode(original / 'report.json', checked, tmp_path / 'new', 4)
    assert (retained / 'states_actions.npz').stat().st_ino == (original / 'states_actions.npz').stat().st_ino
    assert (retained / 'report.json').stat().st_ino != (original / 'report.json').stat().st_ino
    old_hash = m.sha(retained / 'states_actions.npz')
    m.shutil.rmtree(original)
    assert m.sha(retained / 'states_actions.npz') == old_hash
    assert m.read(retained / 'report.json')['success'] is False
    assert m.read(retained / 'independent_replay_report.json')['success'] is False
