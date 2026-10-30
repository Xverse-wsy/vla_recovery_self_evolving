"""Existing segment libraries FIRST; state matching is a later independent stage.

bank_class supplies the user-selected four-way training classification.
Optional local_quality evidence remains separate from these existing labels.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from .features import sha256
from .schema import stable_id, save_windows
from .splits import split_windows, audit_split

BANKS=('D_vla_positive','D_vla_negative','D_code_positive','D_code_negative')
SELECTION_BANKS={
    'vla_progress_candidate':'D_vla_positive',
    'vla_wcm_warning':'D_vla_negative',
    'capx_completed':'D_code_positive',
    'capx_not_completed':'D_code_negative',
}


def write_lines(path, records):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records))


def selection_labels(root, provenance):
    """Use source selection.json as authoritative membership; verify metadata agrees."""
    root=Path(root)
    path=root/'selection.json'
    selection=json.loads(path.read_text())
    by_id={int(p['episode_index']):p for p in provenance}
    if len(by_id)!=len(provenance):
        raise ValueError('Duplicate source segment episode IDs')
    result={}
    selection_sha=sha256(path)
    provenance_sha=sha256(root/'meta/provenance.jsonl')
    for category,bank in SELECTION_BANKS.items():
        if category not in selection:
            raise ValueError(f'Missing existing selection category: {category}')
        for episode in selection[category]:
            eid=int(episode)
            if eid in result or eid not in by_id:
                raise ValueError('Selection IDs overlap or refer to a missing source segment')
            p=by_id[eid]
            if category.startswith('vla_'):
                if p['owner']!='pi05' or p['category']!=category:
                    raise ValueError('Existing VLA selection disagrees with segment provenance')
            else:
                completed=category=='capx_completed'
                if p['owner'] not in ('capx','capx_converted') or p['category']!='capx_program' or bool(p['task_completed_in_segment'])!=completed:
                    raise ValueError('Existing CAP selection disagrees with completion provenance')
            result[eid]=dict(bank_class=bank,bank_label=dict(
                source_category=category,label_scope='existing_segment',
                basis='WCM' if category.startswith('vla_') else 'LIBERO_program_completion',
                selection_path=str(path.resolve()),selection_sha256=selection_sha,
                provenance_path=str((root/'meta/provenance.jsonl').resolve()),
                provenance_sha256=provenance_sha,
                source_episode_index=eid,local_actions_verified=False))
    missing=set(by_id)-set(result)
    if any(by_id[e]['category']!='vla_boundary_uncertain' for e in missing):
        raise ValueError('Source selection does not cover all classified segments')
    return result


def split_manifest(manifest, config):
    entries=deepcopy(manifest)
    ids=[e['segment_id'] for e in entries]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate segment IDs')
    units=[]
    for e in entries:
        bank=e.get('bank_class')
        if bank is not None and bank not in BANKS:
            raise ValueError('Unknown existing bank class')
        if bank and (bank.startswith('D_vla_') != (e['controller']=='VLA')):
            raise ValueError('Existing bank/controller mismatch')
        units.append(SimpleNamespace(sample_id=e['segment_id'],episode_id=e['episode_id'],
            initial_group_id=e.get('initial_group_id') or 'unknown:'+e['task_id'],
            task_id=e['task_id'],uncertain_initialization=not e.get('initialization_verified',False),
            split='unassigned'))
    split_windows(units,config.split_fractions,config.seed,config.conservative_task_split)
    for e,u in zip(entries,units):
        e['initial_group_id'],e['split']=u.initial_group_id,u.split
    return entries


def create_libraries(manifest, config, output):
    """Inventory without encoder, calibration, normalization, action copies or matching."""
    config.validate()
    out=Path(output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    entries=split_manifest(manifest,config)
    write_lines(out/'split_manifest.jsonl',entries)
    for bank in BANKS:
        subset=[e for e in entries if e.get('bank_class')==bank]
        write_lines(out/'segments'/f'{bank}.jsonl',subset)
        for split in ('train','validation','test'):
            write_lines(out/'splits'/split/f'{bank}.jsonl',[e for e in subset if e['split']==split])
    write_lines(out/'unclassified_segments.jsonl',[e for e in entries if not e.get('bank_class')])
    counts={bank:dict(segments=sum(e.get('bank_class')==bank for e in entries),
        action_steps=sum(e['length'] for e in entries if e.get('bank_class')==bank)) for bank in BANKS}
    report=dict(state='libraries_complete',segments=len(entries),
        parent_episodes=len({e['episode_id'] for e in entries}),tasks=len({e['task_id'] for e in entries}),
        banks=counts,split_counts=dict(Counter(e['split'] for e in entries)),
        unclassified_segments=sum(not e.get('bank_class') for e in entries),
        classification='existing selection labels preserved',matching_performed=False,
        negative_behavior_automatically_confirmed=False,
        training_label_policy='reuse_existing_four_segment_classes',
        library_id=stable_id([(e['segment_id'],e.get('bank_class'),e['split'],e.get('bank_label')) for e in entries]))
    (out/'report.json').write_text(json.dumps(report,indent=2))
    (out/'config.json').write_text(json.dumps(config.to_dict(),indent=2))
    hashes={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file()}
    (out/'checksums.json').write_text(json.dumps(hashes,indent=2))
    return report


def audit_libraries(path):
    path=Path(path)
    hashes=json.loads((path/'checksums.json').read_text())
    for name,digest in hashes.items():
        if sha256(path/name)!=digest:
            raise ValueError(f'Segment library changed: {name}')
    entries=[json.loads(x) for x in (path/'split_manifest.jsonl').read_text().splitlines() if x.strip()]
    units=[SimpleNamespace(episode_id=e['episode_id'],initial_group_id=e['initial_group_id'],split=e['split']) for e in entries]
    audit_split(units)
    by_id={e['segment_id']:e for e in entries}
    if len(by_id)!=len(entries):
        raise ValueError('Duplicate library segment IDs')
    seen=set()
    for bank in BANKS:
        rows=[json.loads(x) for x in (path/'segments'/f'{bank}.jsonl').read_text().splitlines() if x.strip()]
        for e in rows:
            if e['segment_id'] in seen or e!=by_id[e['segment_id']] or e['bank_class']!=bank:
                raise ValueError('Invalid existing bank partition')
            seen.add(e['segment_id'])
    if seen!={e['segment_id'] for e in entries if e.get('bank_class')}:
        raise ValueError('Missing classified library segments')
    return entries


def write_window_libraries(records, output):
    """Always export existing categories before state/visual matching is available."""
    out=Path(output)
    def bank_of(r):
        if r.bank_class:
            return r.bank_class
        # Explicitly verified custom/synthetic manifests may have no old category.
        if r.local_quality in ('positive','negative') and not r.source.get('state_only'):
            return f"D_{'vla' if r.controller=='VLA' else 'code'}_{r.local_quality}"
        return None
    for bank in BANKS:
        save_windows(out/'banks'/f'{bank}.jsonl',[r for r in records if bank_of(r)==bank])
    for split in ('train','validation','test'):
        subset=[r for r in records if r.split==split]
        for bank in BANKS:
            save_windows(out/'banks'/split/f'{bank}.jsonl',[r for r in subset if bank_of(r)==bank])
        save_windows(out/'banks'/split/'S_vla_all.jsonl',[r for r in subset if r.controller=='VLA'])
        save_windows(out/'banks'/split/'B_positive.jsonl',[r for r in subset if bank_of(r)=='D_vla_positive'])
        save_windows(out/'banks'/split/'B_negative.jsonl',[r for r in subset if bank_of(r)=='D_vla_negative'])
    report=dict(windows=len(records),
        bank_windows={bank:sum(bank_of(r)==bank for r in records) for bank in BANKS},
        state_only=sum(bool(r.source.get('state_only')) for r in records),
        locally_verified_windows=dict(Counter(r.local_quality for r in records)),
        training_classes=dict(Counter(f'{r.controller}/{r.training_quality}' for r in records)),
        effective_pair_error_elements=sum(int(r.pair_error_mask().sum()) for r in records
                                         if r.training_quality=='negative'),
        trusted_error_elements=sum(int(r.payload()['trusted_error_mask'].sum()) for r in records),
        matching_performed=False)
    (out/'window_libraries.json').write_text(json.dumps(report,indent=2))
    return report
