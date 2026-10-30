from dataclasses import replace
import json
import pytest

from recovery_pair.batches import EpisodeEventSampler
from recovery_pair.config import Config
from recovery_pair.data import read_jsonl
from recovery_pair.matching import FrozenMatcher, NEGATIVE_QUERY_POLICY
from recovery_pair.pipeline import prepare, audit_prepared
from recovery_pair.schema import load_windows


def test_vla_only_queries_keep_cap_success_references_and_preserve_failure_labels(window, config):
    records = [window('vla_positive'), window('cap_positive', 'code'),
               window('vla_negative', quality='negative'),
               window('cap_negative', 'code', 'negative')]
    failed = records[-1]
    failed.bank_class = 'D_code_negative'
    failed.local_quality = 'unknown'
    matcher = FrozenMatcher(records, config)
    assert [r.sample_id for r in matcher.negative] == ['vla_negative']
    assert matcher.excluded_negative == [failed]
    assert {r.sample_id for r in matcher.positive} == {'vla_positive', 'cap_positive'}
    pairs = matcher.build_pairs()
    assert len(pairs) == 1 and pairs[0]['negative_id'] == 'vla_negative'
    assert pairs[0]['negative_controller'] == 'VLA'
    assert pairs[0]['negative_query_policy'] == NEGATIVE_QUERY_POLICY
    assert {r['sample_id'] for r in pairs[0]['references']} == {'vla_positive', 'cap_positive'}
    assert failed.bank_class == 'D_code_negative' and failed.training_quality == 'negative'


def test_prepared_and_sampler_exclude_cap_failure_but_keep_original_inventory(window, config, tmp_path):
    records = [window('positive'), window('vla_negative', quality='negative'),
               window('cap_negative', 'code', 'negative')]
    out = tmp_path / 'prepared'
    report = prepare(records, config, out)
    assert report['accepted_negative'] == 1 and report['excluded_cap_negative_windows'] == 1
    assert report['negative_query_policy'] == NEGATIVE_QUERY_POLICY
    assert len(load_windows(out / 'all_windows.jsonl')) == 3
    assert load_windows(out / 'eligible_banks/train/D_code_negative.jsonl') == []
    assert read_jsonl(out / 'excluded_negative_windows.jsonl')[0]['sample_id'] == 'cap_negative'
    pool, pairs, _ = audit_prepared(out, config)
    assert {r.sample_id for r in pool} == {'positive', 'vla_negative'}
    plan = EpisodeEventSampler(pool, pairs, config).draw()
    assert [p['negative_id'] for p in plan['pairs']] == ['vla_negative']
    invalid = [dict(pairs[0], negative_id='cap_negative')]
    with pytest.raises(ValueError, match='VLA failure queries only'):
        EpisodeEventSampler(records, invalid, config)


def test_cap_failure_only_retains_fm_branch_without_creating_negative_pairs(window, config, tmp_path):
    out = tmp_path / 'prepared'
    report = prepare([window('positive'), window('cap_negative', 'code', 'negative')], config, out)
    assert report['accepted_negative'] == report['paired_windows'] == 0
    pool, pairs, _ = audit_prepared(out, config)
    plan = EpisodeEventSampler(pool, pairs, config).draw()
    assert plan['pairs'] == [] and set(plan['fm_ids']) == {'positive'}


def test_old_prepared_without_vla_only_policy_requires_rebuild(window, config, tmp_path):
    out = tmp_path / 'old_prepared'
    prepare([window('positive'), window('negative', quality='negative')], config, out)
    report = json.loads((out / 'report.json').read_text())
    report.pop('negative_query_policy')
    (out / 'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='VLA-only negative queries'):
        audit_prepared(out, config)
    with pytest.raises(ValueError, match='VLA failures only'):
        replace(Config(), negative_query_policy='all_failures').validate()
