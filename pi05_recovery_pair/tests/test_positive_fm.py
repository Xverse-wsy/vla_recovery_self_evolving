from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from recovery_pair.features import sha256
from recovery_pair.losses import distance_per_window
from recovery_pair.matching import OBSERVATION_POLICY
from recovery_pair.pipeline import audit_prepared
from recovery_pair.positive_fm import (audit_positive_fm, check_training_mode,
                                      prepare_positive_fm, transform_window)
from recovery_pair.schema import save_windows
from recovery_pair.synthetic import loader
from recovery_pair.toy import ToyFlow
from recovery_pair.trainer import train
from recovery_pair.batches import EpisodeEventSampler


def source(window, sid, *, category='vla_progress_candidate', real=10, controller='VLA'):
    r = window(sid, controller)
    r.bank_class = 'D_vla_positive' if controller == 'VLA' else 'D_code_positive'
    r.bank_label = {'source_category': category}
    p = r.payload()
    p['actions'][:real, :7] = np.arange(real, dtype=np.float32)[:, None] / 10
    p['actions'][real:] = 0
    p['valid_mask'][real:] = False
    np.savez(r.action_path, **p)
    return r


def admission(tmp_path, records, config):
    p = tmp_path / 'admission'
    p.mkdir()
    save_windows(p / 'positive_pool.jsonl', records)
    (p / 'config.json').write_text(json.dumps(config.to_dict()))
    (p / 'report.json').write_text(json.dumps(dict(state='complete',
        observation_policy=OBSERVATION_POLICY, positive_pool_windows=len(records))))
    return p


def test_intermediate_vla_only_keeps_full_real_windows(window, tmp_path):
    out = tmp_path / 'new'
    (out / 'actions').mkdir(parents=True)
    short = source(window, 'short', real=9)
    before = sha256(short.action_path)
    assert transform_window(short, out) is None
    assert sha256(short.action_path) == before
    full = source(window, 'full')
    result = transform_window(full, out)
    assert result.source['fm_real_steps'] == 10
    np.testing.assert_array_equal(result.payload()['actions'], full.payload()['actions'])
    assert result.payload()['valid_mask'][:, :7].all()
    assert not result.payload()['valid_mask'][:, 7:].any()


@pytest.mark.parametrize('controller,category', [('code', 'capx_completed'),
                                                ('VLA', 'autonomous_episode_success')])
def test_success_tail_repeats_last_action_and_is_supervised(window, tmp_path, controller, category):
    out = tmp_path / 'new'
    (out / 'actions').mkdir(parents=True)
    r = source(window, 'terminal', controller=controller, category=category, real=2)
    before = sha256(r.action_path)
    result = transform_window(r, out)
    p = result.payload()
    assert result.source['fm_repeated_steps'] == 8
    np.testing.assert_array_equal(p['actions'][2:, :7], np.full((8, 7), .1, np.float32))
    assert p['valid_mask'][:, :7].all()
    assert p['real_action_mask'][:2, :7].all() and not p['real_action_mask'][2:].any()
    # A wrong prediction ONLY on the repeated tail must contribute to FM loss.
    target = torch.tensor(p['actions'][None])
    prediction = target.clone()
    prediction[:, 2:, :7] += 1
    loss, valid = distance_per_window(prediction, target, torch.tensor(p['valid_mask'][None]))
    assert valid.all() and loss.item() == pytest.approx(.8)
    assert sha256(r.action_path) == before


def test_positive_fm_prepared_is_auditable_no_pairs_and_fm_training_resumes(window, config, tmp_path):
    config = replace(config, pair_batch=0, eta=0, fm_batch=8, fm_only_updates=2)
    records = [source(window, 'progress_full'), source(window, 'progress_tail', real=8),
               source(window, 'cap_tail', controller='code', category='capx_completed', real=1),
               source(window, 'auto_tail', category='autonomous_episode_success', real=3)]
    previous = admission(tmp_path, records, config)
    hashes = {r.action_path: sha256(r.action_path) for r in records}
    out = tmp_path / 'prepared'
    report = prepare_positive_fm(previous, config, out)
    assert report['positive_windows'] == 3 and report['dropped_intermediate_vla_windows'] == 1
    assert report['repeated_tail_windows'] == 2 and report['negative_windows'] == report['pairs'] == 0
    accepted, pairs, _ = audit_prepared(out, config)
    assert pairs == [] and all(r.training_quality == 'positive' for r in accepted)
    assert {p: sha256(p) for p in hashes} == hashes
    plan = EpisodeEventSampler(accepted, pairs, config).draw()
    assert len(plan['fm_ids']) == 8 and plan['pairs'] == []
    with pytest.raises(ValueError, match='--mode fm_only'):
        check_training_mode(out, 'continuous_pair')
    with pytest.raises(ValueError, match='--mode fm_only'):
        train(ToyFlow(), loader, out, config, tmp_path / 'wrong_mode')
    model = ToyFlow()
    result = train(model, loader, out, config, tmp_path / 'run', mode='fm_only',
                   max_updates=1, save_every=1)
    assert result['updates'] == 1
    metrics = json.loads((tmp_path / 'run/metrics.jsonl').read_text().splitlines()[0])
    assert metrics['pair_count'] == 0 and metrics['pair'] == metrics['eta'] == 0
    resumed = train(model, loader, out, config, tmp_path / 'run', mode='fm_only',
        resume=tmp_path / 'run/update_000001.pt', save_every=1)
    assert resumed['updates'] == 2
    with pytest.raises(FileExistsError):
        prepare_positive_fm(previous, config, out)


def test_positive_fm_rejects_unknown_sources_negative_records_and_corrupted_tail(window, config, tmp_path):
    (tmp_path / 'actions').mkdir()
    unknown = source(window, 'unknown_source', category='unverified')
    with pytest.raises(ValueError, match='source must identify'):
        transform_window(unknown, tmp_path)
    with pytest.raises(ValueError, match='rejects negative'):
        transform_window(window('negative', quality='negative'), tmp_path)
    r = transform_window(source(window, 'cap', controller='code', category='capx_completed', real=2), tmp_path)
    p = r.payload()
    p['actions'][-1, 0] += 1
    np.savez(r.action_path, **p)
    with pytest.raises(ValueError, match='repeat the last real action'):
        audit_positive_fm([r], [], replace(config, pair_batch=0, eta=0))
