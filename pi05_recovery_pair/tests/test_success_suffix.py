from dataclasses import replace
import json
import pytest

from recovery_pair.matching import FrozenMatcher
from recovery_pair.success_admission import check_coverage_features
from recovery_pair.pipeline import prepare, audit_prepared
from recovery_pair.data import read_jsonl
from recovery_pair.features import sha256
from recovery_pair.schema import save_windows, load_windows
from recovery_pair.success_admission import admit_success
from recovery_pair.synthetic import fixture


def segment_window(window, sid, row, position, segment='program1', episode='recovery', quality='positive'):
    r = window(sid, 'code', quality, episode=episode, position=position)
    r.start_step = 100 + row
    r.start_time_s = r.start_step / 20
    r.source.update(segment_id=segment, segment_row=row, action_range=[row, row+10])
    return r


def test_suffix_uses_earliest_match_admits_unmatched_tail_and_no_other_program(window, config):
    config = replace(config, cap_success_admission='segment_suffix')
    vla = window('reference')
    before = segment_window(window, 'before', 0, .3)
    anchor = segment_window(window, 'anchor', 10, .01)
    tail = segment_window(window, 'tail', 20, .4)
    later_match = segment_window(window, 'later_match', 30, .02)
    tail2 = segment_window(window, 'tail2', 40, .5)
    different_program = segment_window(window, 'other_program', 0, .4, segment='program2')
    different_episode = segment_window(window, 'other_episode', 0, .4, episode='other')
    failed_program = segment_window(window, 'failed', 0, 0, segment='failed', quality='negative')
    matcher = FrozenMatcher([tail2, later_match, different_episode, before, anchor, vla,
                             tail, different_program, failed_program], config)
    assert set(matcher.coverage) == {'anchor', 'tail', 'later_match', 'tail2'}
    assert {m['sample_id'] for m in matcher.success_matches} == {'anchor', 'later_match'}
    suffix = matcher.success_segments[0]
    assert suffix['segment_row_range'] == [10, 50]
    assert suffix['parent_step_range'] == [110, 150]
    assert suffix['action_steps'] == 40
    assert suffix['anchor_id'] == 'anchor'
    assert matcher.coverage['tail']['directly_matched'] is False
    assert matcher.coverage['tail']['reference_applies_to'] == 'suffix_anchor'
    assert matcher.coverage['later_match']['directly_matched'] is True
    # An admitted tail near another CAP never extends that other program.
    assert different_program not in matcher.positive
    assert failed_program not in matcher.negative
    assert failed_program in matcher.excluded_negative


def test_suffix_rejects_missing_actions_between_windows(window, config):
    config = replace(config, cap_success_admission='segment_suffix')
    anchor = segment_window(window, 'anchor', 0, 0)
    tail = segment_window(window, 'tail', 20, .5)
    with pytest.raises(ValueError, match='contiguous'):
        FrozenMatcher([window('reference'), anchor, tail], config)


def test_missing_segment_identity_never_extends_other_window(window, config):
    config = replace(config, cap_success_admission='segment_suffix')
    anchor = window('anchor', 'code', episode='shared')
    tail = window('tail', 'code', episode='shared', position=.5)
    tail.start_step = 10
    matcher = FrozenMatcher([window('reference'), anchor, tail], config)
    assert set(matcher.coverage) == {'anchor'}


def test_admission_refuses_missing_features_instead_of_silent_rejection(window, config):
    vla = window('reference')
    code = window('anchor', 'code')
    assert check_coverage_features([vla, code], config) == 1
    code.visual_features = {}
    with pytest.raises(ValueError, match='Incomplete coverage candidate features'):
        check_coverage_features([vla, code], config)


def test_prepared_includes_suffix_and_rejects_old_admission_policy(window, config, tmp_path):
    config = replace(config, cap_success_admission='segment_suffix')
    records = [window('reference'), segment_window(window, 'anchor', 0, 0),
               segment_window(window, 'tail', 10, .5), window('negative', quality='negative')]
    out = tmp_path / 'prepared'
    result = prepare(records, config, out)
    assert result['matched_cap_success_windows'] == 1
    assert result['admitted_cap_success_windows'] == 2
    assert len(load_windows(out / 'matched_cap_success_windows.jsonl')) == 1
    assert len(load_windows(out / 'admitted_cap_success_windows.jsonl')) == 2
    assert read_jsonl(out / 'cap_success_suffixes.jsonl')[0]['action_steps'] == 20
    pool, pairs, digest = audit_prepared(out, config)
    assert {r.sample_id for r in pool} == {r.sample_id for r in records}
    assert pairs and digest
    with pytest.raises(ValueError, match='cap_success_admission'):
        audit_prepared(out, replace(config, cap_success_admission='matched_window'))


@pytest.mark.parametrize('truncate_tail', [False, True])
def test_portable_admission_checks_complete_original_program_end(tmp_path, truncate_tail):
    cfg, records, _ = fixture(tmp_path / 'synthetic')
    cfg = replace(cfg, cap_success_admission='segment_suffix')
    if truncate_tail:
        # Keep a legitimate anchor but remove its final two recorded actions from the index.
        target = next(r.source['segment_id'] for r in records
                      if r.controller == 'code' and r.training_quality == 'positive')
        records = [r for r in records if not (r.controller == 'code' and
                   r.source['segment_id'] == target and r.source['segment_row'] == 10)]
    source = tmp_path / 'windows.jsonl'
    save_windows(source, records)
    trial = tmp_path / 'features'
    trial.mkdir()
    save_windows(trial / 'windows_with_features.jsonl', records)
    (trial / 'report.json').write_text(json.dumps(dict(state='complete',
        observation_policy='policy_inputs_only_v1', inputs={'input_sha256': sha256(source)},
        feature_report={'encoder_sha256': 'synthetic_frozen_features'})))
    output = tmp_path / 'admitted'
    if truncate_tail:
        with pytest.raises(ValueError, match='original program segment end'):
            admit_success(source, trial, cfg, output)
        assert json.loads((output / 'status.json').read_text())['state'] == 'failed'
    else:
        report = admit_success(source, trial, cfg, output)
        assert report['admitted_cap_success_segments'] == 6
        assert report['admitted_cap_action_steps'] == 72
        assert report['admitted_cap_windows'] == 12
        assert not report['training_started']
