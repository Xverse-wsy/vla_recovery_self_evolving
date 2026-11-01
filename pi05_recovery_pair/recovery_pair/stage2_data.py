"""Independent Stage2 training assets; Stage1 and matching inputs are read-only."""
import json
from collections import Counter
from pathlib import Path
import shutil

import numpy as np

from .config import Config
from .data import read_jsonl, write_jsonl
from .features import sha256
from .matching import numeric_match, image_distance, pair_mask, OBSERVATION_POLICY
from .pipeline import audit_prepared
from .schema import Window, load_windows, save_windows, stable_id

POLICY = 'stage2_native_fm_continuous_pair_v1'
SELECTED = dict(position_m=.02, rotation_deg=10., proprio_rms=.2,
                image_cosine=.002, max_candidates=50)


def validate_config(config):
    config.validate(matching_ready=True)
    if (config.horizon,config.action_dim,config.environment_action_dim,config.native_control_hz,config.flow_steps)!=(10,32,7,20,3):
        raise ValueError('Selected Pi0.5 Stage2 requires H10/D32/environment7/control20Hz/deployed K3')
    if config.to_dict()['pairing'] != SELECTED or config.fm_epochs is not None:
        raise ValueError('Stage2 needs selected numeric20 gates and an episode reuse budget, not FM epochs')
    if config.min_pair_steps != 3:
        raise ValueError('Selected Stage2 requires at least three effective steps')


def clone_record(record, output, role, *, preserve_id=False):
    data = record.to_dict()
    sid = record.sample_id if preserve_id else stable_id(record.sample_id, POLICY, role)
    dest = Path(output)/'actions'/(sid+'.npz')
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(record.action_path, dest)
    data.update(sample_id=sid, action_path=str(dest.resolve()), split='train')
    data['source'] = dict(record.source, stage2_training_role=role,
        stage2_input_id=record.sample_id, stage2_input_action_sha256=sha256(record.action_path))
    return Window.from_dict(data)


def check_pair(pair, by_id, config):
    negative = by_id[pair['negative_id']]
    if negative.controller != 'VLA' or negative.training_quality != 'negative':
        raise ValueError('Only VLA negative queries are eligible')
    if not 1 <= len(pair['references']) <= 3 or pair.get('match_inherited') is not False:
        raise ValueError('Independent start matches with one to three references required')
    for ref in pair['references']:
        positive = by_id[ref['sample_id']]
        if positive.training_quality != 'positive':
            raise ValueError('Non-positive reference')
        _, reason = numeric_match(negative, positive, config.pairing)
        images, visual_reason = image_distance(negative, positive)
        if reason or visual_reason or any(x > config.pairing.image_cosine for x in images.values()):
            raise ValueError('Saved reference no longer passes start-state matching')
        mask, reason = pair_mask(positive, negative, config)
        if reason or not np.array_equal(mask, np.asarray(ref['mask'], bool)):
            raise ValueError('Saved reference mask no longer passes action checks')
        # Pair records MUST never contain synthetic terminal repeats.
        for record in (positive, negative):
            p = record.payload()
            if 'real_action_mask' in p and not np.array_equal(p['valid_mask'], p['real_action_mask']):
                raise ValueError('FM-only synthetic repeats entered pair comparison')


def prepare(stage1_prepared, matching, config, output):
    validate_config(config)
    source, matched, out = map(lambda p: Path(p).resolve(), (stage1_prepared, matching, output))
    if any(out == p or p in out.parents for p in (source, matched)):
        raise ValueError('Use an independent Stage2 output outside the inputs')
    fm, old_pairs, stage1_sha = audit_prepared(source)
    info = json.loads((source/'report.json').read_text())
    report = json.loads((matched/'report.json').read_text())
    if old_pairs or info.get('split_policy') != 'all_train_v1' or info.get('training_start_stride') != 1:
        raise ValueError('Expected dense all-train Stage1 FM positives')
    if report.get('state') != 'complete' or report.get('source_stage1_sha256') != stage1_sha:
        raise ValueError('Matching did not use this immutable Stage1 prepared pool')
    if report.get('pairing') != SELECTED or report.get('observation_policy') != OBSERVATION_POLICY:
        raise ValueError('Use the selected numeric20 policy-input-only matching result')
    files = ['report.json', 'pairs.jsonl', 'negative_queries.jsonl', 'positive_references.jsonl']
    input_hashes = {f: sha256(matched/f) for f in files}
    queries = {r.sample_id:r for r in load_windows(matched/'negative_queries.jsonl')}
    references = {r.sample_id:r for r in load_windows(matched/'positive_references.jsonl')}
    source_fm = {r.sample_id:r for r in fm}
    pairs = read_jsonl(matched/'pairs.jsonl')
    out.mkdir(parents=True, exist_ok=False)
    records = [clone_record(r, out, 'fm') for r in fm]
    fm_ids = [r.sample_id for r in records]
    added = set()
    for pair in pairs:
        neg = Window.from_dict(pair['negative_start'])
        if neg.sample_id not in queries or neg.episode_id != queries[neg.sample_id].episode_id:
            raise ValueError('Negative start is absent from the source query inventory')
        starts = [neg]
        for ref in pair['references']:
            pos = Window.from_dict(pair['positive_starts'][ref['sample_id']])
            original_id = pos.source.get('stage2_original_id')
            if pos.sample_id not in references or original_id not in source_fm:
                raise ValueError('Positive reference is outside the admitted Stage1 scopes')
            original = source_fm[original_id]
            real = original.payload().get('real_action_mask', original.payload()['valid_mask'])
            pp = pos.payload()
            if (pos.image_refs != original.image_refs or pos.proprio != original.proprio or
                    pos.instruction != original.instruction or not np.array_equal(pp['valid_mask'],real) or
                    not np.array_equal(pp['actions'][real],original.payload()['actions'][real])):
                raise ValueError('Reference differs from its own real Stage1 observation/actions')
            starts.append(pos)
        # Verify saved start metadata/actions against the source inventories, then masks.
        for record in starts:
            origin = queries.get(record.sample_id) or references.get(record.sample_id)
            a, b = record.to_dict(), origin.to_dict()
            a.pop('visual_features'); b.pop('visual_features')
            if a != b:
                raise ValueError('Saved pair start differs from its matching index')
        check_pair(pair, {r.sample_id:r for r in starts}, config)
        for record in starts:
            if record.sample_id not in added:
                role = 'negative' if record.training_quality == 'negative' else 'reference'
                records.append(clone_record(record, out, role, preserve_id=True))
                added.add(record.sample_id)
    by_id = {r.sample_id:r for r in records}
    frames = set()
    for p in pairs:
        neg = by_id[p['negative_id']]
        union = np.logical_or.reduce([np.asarray(r['mask'],bool).any(-1) for r in p['references']])
        frames.update((neg.episode_id,neg.start_step+int(h)) for h in np.flatnonzero(union))
    summary = dict(state='complete', training_policy=POLICY, observation_policy=OBSERVATION_POLICY,
        source_stage1_prepared=str(source), source_stage1_sha256=stage1_sha,
        source_matching=str(matched), source_matching_sha256=input_hashes,
        fm_windows=len(fm_ids), fm_categories=dict(Counter(r.bank_label.get('source_category') for r in records[:len(fm)])),
        negative_windows=len(pairs), paired_episodes=len({p['negative_episode_id'] for p in pairs}),
        unique_effective_negative_action_frames=len(frames), all_train=True,
        pair_tail_padding=False, matching_redone=False, training_started=False)
    if len(pairs)!=report['paired_negative_windows'] or len(frames)!=report['unique_effective_negative_action_frames']:
        raise ValueError('Prepared counts differ from completed matching')
    save_windows(out/'windows.jsonl',records)
    write_jsonl(out/'pairs.jsonl',pairs)
    (out/'fm_ids.json').write_text(json.dumps(fm_ids))
    (out/'config.json').write_text(json.dumps(config.to_dict(),indent=2))
    (out/'report.json').write_text(json.dumps(summary,indent=2))
    checksums = {p.name:sha256(p) for p in out.glob('*.json*')}
    checksums['payloads'] = {r.sample_id:sha256(r.action_path) for r in records}
    (out/'checksums.json').write_text(json.dumps(checksums,indent=2))
    if any(sha256(matched/f)!=h for f,h in input_hashes.items()):
        raise ValueError('Matching inputs changed while preparing')
    _,_,after = audit_prepared(source)
    if after != stage1_sha:
        raise ValueError('Stage1 prepared changed')
    return summary


def audit(path, config=None):
    path = Path(path)
    stored = Config.load(path/'config.json')
    validate_config(stored)
    if config is not None:
        validate_config(config)
        for key in ('horizon','action_dim','environment_action_dim','native_control_hz','flow_steps',
                    'pairing','min_pair_steps','shared_prefix_tolerance','identical_action_tolerance'):
            if getattr(stored,key)!=getattr(config,key):
                raise ValueError('Stage2 prepared configuration mismatch: '+key)
    checksums = json.loads((path/'checksums.json').read_text())
    for name,digest in checksums.items():
        if name!='payloads' and sha256(path/name)!=digest:
            raise ValueError('Stage2 artifact changed: '+name)
    report = json.loads((path/'report.json').read_text())
    if report.get('training_policy')!=POLICY:
        raise ValueError('Not an independent Stage2 training artifact')
    records = load_windows(path/'windows.jsonl')
    by_id = {r.sample_id:r for r in records}
    if len(by_id)!=len(records) or set(by_id)!=set(checksums['payloads']):
        raise ValueError('Stage2 payload inventory differs')
    for r in records:
        r.payload()
        if r.split!='train' or sha256(r.action_path)!=checksums['payloads'][r.sample_id]:
            raise ValueError('Stage2 payload changed or split differs')
    if len({r.spec.identity for r in records})!=1:
        raise ValueError('Mixed action specifications')
    if any((r.spec.horizon,r.spec.model_dim,r.spec.control_hz)!=(stored.horizon,stored.action_dim,stored.native_control_hz) for r in records):
        raise ValueError('Stage2 data/model shape or control frequency differs')
    fm_ids = json.loads((path/'fm_ids.json').read_text())
    expected = {r.sample_id for r in records if r.source.get('stage2_training_role')=='fm'}
    if len(fm_ids)!=len(set(fm_ids)) or set(fm_ids)!=expected:
        raise ValueError('Stage2 FM inventory differs')
    if any(by_id[s].training_quality!='positive' for s in fm_ids):
        raise ValueError('Negative actions entered ordinary FM')
    pairs = read_jsonl(path/'pairs.jsonl')
    if len({p['negative_id'] for p in pairs})!=len(pairs):
        raise ValueError('Duplicate negative starts would overweight windows')
    for pair in pairs:
        check_pair(pair,by_id,stored)
    return records,pairs,fm_ids,sha256(path/'checksums.json')
