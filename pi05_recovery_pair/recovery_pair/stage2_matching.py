"""Independent dense VLA-negative matching with expert references; never train.

Stage1's frozen admission and FM artifact are read-only inputs. Expert roles stay
explicit. Every negative observation is queried independently; no suffix admission
or inherited neighbor match exists in this module.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .config import Config
from .data import QuantileAssets, quaternion_from_rotvec, read_jsonl, write_jsonl
from .features import sha256
from .matching import OBSERVATION_POLICY, numeric_match, image_distance, pair_mask, instruction_key
from .pipeline import audit_prepared
from .schema import Window, load_windows, save_windows, stable_id

POLICY = 'stage2_dense_negative_expert_reference_v1'


def positive_references(records, output):
    """Keep exactly the Stage1 positive scope; remove FM-only repeated tail masks."""
    out = Path(output)
    (out/'actions').mkdir(parents=True, exist_ok=True)
    result = []
    for original in records:
        if original.training_quality != 'positive' or original.source.get('state_only'):
            raise ValueError('Expected positive reference records only')
        payload = original.payload()
        real = payload.get('real_action_mask', payload['valid_mask']).astype(bool)
        if not real.any() or (real & ~payload['valid_mask']).any():
            raise ValueError('Invalid real-action reference mask')
        actions = np.where(real, payload['actions'], 0)
        sid = stable_id(original.sample_id, POLICY, 'reference')
        dest = out/'actions'/(sid+'.npz')
        np.savez_compressed(dest, actions=actions, valid_mask=real,
                            trusted_error_mask=np.zeros_like(real))
        d = original.to_dict()
        d.update(sample_id=sid, action_path=str(dest), split='train')
        d['source'] = dict(original.source, stage2_policy=POLICY, stage2_role='positive_reference',
            stage2_original_id=original.sample_id, stage2_original_split=original.split,
            stage2_original_action_path=original.action_path,
            stage2_original_action_sha256=sha256(original.action_path),
            stage2_synthetic_tail_excluded=True,
            stage2_expert_reference_authorized=original.controller=='expert')
        result.append(Window.from_dict(d))
    if not result or len({r.sample_id for r in result}) != len(result):
        raise ValueError('Nonempty unique positive reference pool required')
    return result


def dense_negative_queries(records, output):
    """Expand only existing VLA-negative label/event scopes, retaining real tails."""
    groups = defaultdict(list)
    for r in records:
        if r.controller=='VLA' and r.training_quality=='negative' and not r.source.get('state_only'):
            groups[(Path(r.action_path).parent.parent/'input_manifest.jsonl',r.source['segment_id'])].append(r)
    if not groups:
        raise ValueError('No VLA-negative segments found')
    out = Path(output)
    (out/'actions').mkdir(parents=True,exist_ok=True)
    manifest_cache, norm_cache, result, scopes = {}, {}, [], []
    for (manifest,segment), anchors in groups.items():
        if manifest not in manifest_cache:
            manifest_cache[manifest] = (sha256(manifest),{m['segment_id']:m for m in read_jsonl(manifest)})
        manifest_sha, entries = manifest_cache[manifest]
        entry = entries[segment]
        cuts = sorted({0,entry['length']}|{int(a[k]) for a in entry.get('annotations',[]) for k in ('start','end')})
        by_boundary = defaultdict(list)
        for a in anchors:
            row = a.source['segment_row']
            boundary = next((lo,hi) for lo,hi in zip(cuts,cuts[1:]) if lo<=row<hi)
            by_boundary[boundary].append(a)
        table = pq.read_table(entry['parquet'],columns=['state','actions'])
        parquet_sha = sha256(entry['parquet'])
        states = np.asarray(table['state'].to_pylist(),np.float32)
        raw = np.asarray(table['actions'].to_pylist(),np.float32)
        if states.shape!=(entry['length'],8) or raw.shape!=(entry['length'],7):
            raise ValueError('Negative source length/state/action mismatch')
        for (lo,hi), scope_anchors in sorted(by_boundary.items()):
            anchor = min(scope_anchors,key=lambda r:r.source['segment_row'])
            if any((r.event_id,r.bank_class,r.local_quality,r.spec.identity,r.instruction)!=
                   (anchor.event_id,anchor.bank_class,anchor.local_quality,anchor.spec.identity,anchor.instruction)
                   for r in scope_anchors):
                raise ValueError('Negative scope crosses an event or local label')
            if anchor.controller!=entry['controller'] or anchor.episode_id!=entry['episode_id']:
                raise ValueError('Negative manifest identity mismatch')
            norm = anchor.source['normalization_asset']
            if norm not in norm_cache: norm_cache[norm]=QuantileAssets(norm)
            asset = norm_cache[norm]
            if asset.sha!=anchor.spec.normalization_sha256:
                raise ValueError('Negative normalization changed')
            normalized = asset.normalize('actions',raw)
            for a in scope_anchors:
                x,y = a.source['action_range']
                if not lo<=x<y<=hi or not np.allclose(a.payload()['actions'][:y-x,:7],normalized[x:y],atol=2e-6,rtol=0):
                    raise ValueError('Negative anchor differs from original actions')
            scopes.append(dict(segment_id=segment,episode_id=anchor.episode_id,event_id=anchor.event_id,
                row_range=[lo,hi],queries=hi-lo,original_split=anchor.split,
                source_manifest=str(manifest),source_manifest_sha256=manifest_sha))
            for row in range(lo,hi):
                n = min(anchor.spec.horizon,hi-row)
                actions = np.zeros((anchor.spec.horizon,anchor.spec.model_dim),np.float32)
                valid = np.zeros_like(actions,bool)
                actions[:n,:7],valid[:n,:7] = normalized[row:row+n],True
                sid = stable_id(segment,lo,hi,row,POLICY,'negative')
                dest = out/'actions'/(sid+'.npz')
                trusted = valid.copy() if anchor.local_quality=='negative' else np.zeros_like(valid)
                np.savez_compressed(dest,actions=actions,valid_mask=valid,trusted_error_mask=trusted)
                d = anchor.to_dict()
                d.update(sample_id=sid,action_path=str(dest),start_step=entry['parent_start_step']+row,
                    start_time_s=(entry['parent_start_step']+row)/anchor.spec.control_hz,split='train',
                    proprio=states[row].tolist(),standardized_proprio=asset.normalize('state',states[row]).tolist(),
                    eef_position_m=states[row,:3].tolist(),eef_quaternion_xyzw=quaternion_from_rotvec(states[row,3:6]),
                    image_refs={v:dict(parquet=str(Path(entry['parquet']).resolve()),column=c,row=row)
                                for v,c in [('front','image'),('wrist','wrist_image')]},visual_features={},
                    phase=None,target_object=None,gripper_state=None,held_object=None,contact_state=None,
                    progress=None,object_poses={})
                d['source'] = dict(anchor.source,segment_row=row,action_range=[row,row+n],
                    stage2_policy=POLICY,stage2_role='negative_query',stage2_original_split=anchor.split,
                    stage2_scope_range=[lo,hi],stage2_source_manifest=str(manifest),
                    stage2_source_manifest_sha256=manifest_sha,stage2_source_parquet_sha256=parquet_sha,
                    stage2_start_stride=1,stage2_no_tail_padding=True,stage2_match_inherited=False)
                result.append(Window.from_dict(d))
        if sha256(manifest)!=manifest_sha:
            raise ValueError('Negative source manifest changed')
        if sha256(entry['parquet'])!=parquet_sha:
            raise ValueError('Negative source Parquet changed')
    if len({r.sample_id for r in result})!=len(result):
        raise ValueError('Duplicate negative query starts')
    return result,scopes


def numerical_candidates(negative, references, config):
    candidates=[]
    for positive in references:
        distances,reason=numeric_match(negative,positive,config.pairing)
        if reason is None:
            rank=sum(distances[k]/getattr(config.pairing,k) for k in ('position_m','rotation_deg','proprio_rms'))
            candidates.append((rank,positive.sample_id,distances))
    return [dict(positive_id=sid,distances=d) for _,sid,d in sorted(candidates)[:config.pairing.max_candidates]]


def candidate_index(negatives, positives, config):
    groups=defaultdict(list)
    for r in positives: groups[instruction_key(r)].append(r)
    return {n.sample_id:numerical_candidates(n,groups[instruction_key(n)],config) for n in negatives}


def evaluate(negatives, positives, candidates, config):
    """Every query needs its own start match. No future observations are read."""
    by_id={r.sample_id:r for r in positives}
    rows,pairs=[],[]
    for neg in negatives:
        visual,usable,rejections=[],[],Counter()
        for candidate in candidates[neg.sample_id]:
            pos=by_id[candidate['positive_id']]
            images,reason=image_distance(neg,pos)
            if reason: raise ValueError('Missing candidate features: '+reason)
            if any(d>config.pairing.image_cosine for d in images.values()): continue
            ref=dict(sample_id=pos.sample_id,episode_id=pos.episode_id,controller=pos.controller,
                source_category=pos.bank_label.get('source_category'),
                distances=dict(candidate['distances'],images=images),
                thresholds=config.to_dict()['pairing'],encoder_sha256=neg.visual_features['encoder_sha256'])
            visual.append(ref)
            if pos.episode_id==neg.episode_id and pos.start_step==neg.start_step:
                rejections['same_start_opposing_labels']+=1
                continue
            mask,reason=pair_mask(pos,neg,config)
            if reason:
                rejections[reason]+=1
                continue
            usable.append(dict(ref,mask=mask.tolist(),effective_steps=int(mask.any(-1).sum()),
                               effective_elements=int(mask.sum())))
        usable.sort(key=lambda x:(x['episode_id']==neg.episode_id,max(x['distances']['images'].values()),x['sample_id']))
        chosen,episodes=[],set()
        for ref in usable:
            if ref['episode_id'] not in episodes:
                chosen.append(ref);episodes.add(ref['episode_id'])
            if len(chosen)==3: break
        if len(chosen)<3:
            ids={r['sample_id'] for r in chosen}
            chosen.extend(r for r in usable if r['sample_id'] not in ids)
            chosen=chosen[:3]
        row=dict(negative_id=neg.sample_id,episode_id=neg.episode_id,segment_id=neg.source['segment_id'],
            event_id=neg.event_id,start_step=neg.start_step,segment_row=neg.source['segment_row'],
            instruction=neg.instruction,valid_action_steps=int(neg.payload()['valid_mask'].any(-1).sum()),
            numerical_references=len(candidates[neg.sample_id]),visual_references=len(visual),
            training_references=len(chosen),action_rejections=dict(rejections),
            selected_positive_ids=[r['sample_id'] for r in chosen],match_inherited=False)
        rows.append(row)
        if chosen:
            pairs.append(dict(pair_id=stable_id(neg.sample_id,[r['sample_id'] for r in chosen],POLICY),
                negative_id=neg.sample_id,negative_episode_id=neg.episode_id,event_id=neg.event_id,
                references=chosen,negative_start=neg.to_dict(),
                positive_starts={r['sample_id']:by_id[r['sample_id']].to_dict() for r in chosen},
                observation_policy=OBSERVATION_POLICY,matching_policy=POLICY,
                scope='one_matched_observation_and_its_action_chunk_only',match_inherited=False,
                label_basis=neg.label_basis,error_mask_basis=neg.label_basis if neg.local_quality=='unknown' else 'verified_local_actions'))
    return pairs,rows


def summarize(negatives, positives, pairs, rows):
    by_id={r.sample_id:r for r in negatives}
    paired_ids={p['negative_id'] for p in pairs}
    raw_frames,effective_frames=set(),set()
    effective_slots=raw_slots=0
    refs=Counter()
    for p in pairs:
        n=by_id[p['negative_id']]
        valid=n.payload()['valid_mask'].any(-1)
        raw_slots+=int(valid.sum())
        raw_frames.update((n.episode_id,n.start_step+int(h)) for h in np.flatnonzero(valid))
        union=np.logical_or.reduce([np.asarray(r['mask'],bool).any(-1) for r in p['references']])
        effective_slots+=int(union.sum())
        effective_frames.update((n.episode_id,n.start_step+int(h)) for h in np.flatnonzero(union))
        refs.update(r['controller'] for r in p['references'])
    return dict(negative_query_windows=len(negatives),
        negative_query_segments=len({r.source['segment_id'] for r in negatives}),
        negative_query_episodes=len({r.episode_id for r in negatives}),
        negative_query_unique_observation_starts=len({(r.episode_id,r.start_step) for r in negatives}),
        negative_query_valid_action_slots=sum(int(r.payload()['valid_mask'].any(-1).sum()) for r in negatives),
        negative_valid_steps_distribution=dict(Counter(int(r.payload()['valid_mask'].any(-1).sum()) for r in negatives)),
        positive_reference_windows=len(positives),positive_reference_controllers=dict(Counter(r.controller for r in positives)),
        positive_reference_categories=dict(Counter(r.bank_label.get('source_category') for r in positives)),
        numerical_match_windows=sum(r['numerical_references']>0 for r in rows),
        visual_match_windows=sum(r['visual_references']>0 for r in rows),
        paired_negative_windows=len(pairs),paired_negative_unique_observation_starts=len({(r.episode_id,r.start_step) for r in negatives if r.sample_id in paired_ids}),
        paired_negative_segments=len({r.source['segment_id'] for r in negatives if r.sample_id in paired_ids}),
        paired_negative_episodes=len({r.episode_id for r in negatives if r.sample_id in paired_ids}),
        selected_positive_references=sum(len(p['references']) for p in pairs),
        selected_positive_reference_controllers=dict(refs),
        paired_windows_with_expert_reference=sum(any(r['controller']=='expert' for r in p['references']) for p in pairs),
        raw_action_slots_in_paired_windows=raw_slots,unique_raw_action_frames_in_paired_windows=len(raw_frames),
        effective_action_slots_union_per_window=effective_slots,unique_effective_negative_action_frames=len(effective_frames),
        action_rejections=dict(sum((Counter(r['action_rejections']) for r in rows),Counter())))


def run(positive_prepared, negative_windows, checkpoint, config, output, *, device='cuda:0', memory_gib=8., batch_size=2, encoder_factory=None, expected_expert_episodes=40):
    config.validate(matching_ready=True)
    out=Path(output).resolve()
    sources=[Path(positive_prepared).resolve(),Path(negative_windows).resolve()]
    if any(out==s or s in out.parents for s in sources):
        raise ValueError('Stage2 output must be outside Stage1 inputs')
    out.mkdir(parents=True,exist_ok=False)
    status=out/'status.json'
    def update(state,**extra):
        status.write_text(json.dumps(dict(state=state,**extra),indent=2))
        print(json.dumps(dict(state=state,**extra)),flush=True)
    try:
        update('auditing_stage1_inputs')
        prepared,pairs,digest=audit_prepared(positive_prepared)
        if pairs: raise ValueError('Expected Stage1 positive-only prepared data')
        info=json.loads((Path(positive_prepared)/'report.json').read_text())
        if info.get('training_start_stride')!=1 or info.get('expert_episodes')!=expected_expert_episodes or info.get('split_policy')!='all_train_v1':
            raise ValueError('Use the dense all-train Stage1 pool including expert40')
        if any((r.spec.horizon,r.spec.model_dim,r.spec.control_hz)!=(config.horizon,config.action_dim,config.native_control_hz) for r in prepared):
            raise ValueError('Stage2 action dimensions/frequency differ from Stage1')
        original_sha=sha256(negative_windows)
        originals=load_windows(negative_windows)
        positives=positive_references(prepared,out/'references')
        negatives,scopes=dense_negative_queries(originals,out/'queries')
        write_jsonl(out/'negative_scopes.jsonl',scopes)
        save_windows(out/'positive_references.jsonl',positives)
        save_windows(out/'negative_queries.jsonl',negatives)
        update('numerical_candidates',positive_windows=len(positives),negative_windows=len(negatives),negative_segments=len(scopes))
        candidates=candidate_index(negatives,positives,config)
        write_jsonl(out/'numerical_candidates.jsonl',[dict(negative_id=n,candidates=v) for n,v in candidates.items()])
        wanted={n for n,v in candidates.items() if v}|{r['positive_id'] for v in candidates.values() for r in v}
        selected=[r for r in positives+negatives if r.sample_id in wanted]
        feature_report=None
        if selected:
            update('encoding_candidate_rgb',windows=len(selected),max_candidates=config.pairing.max_candidates)
            if encoder_factory is None:
                from .pi05_vision import Pi05Vision
                encoder_factory=Pi05Vision
            encoder=encoder_factory(checkpoint,device,memory_gib)
            feature_report=encoder.extract(selected,out/'feature_cache',batch_size)
        update('independent_visual_and_action_pairing')
        pairs,rows=evaluate(negatives,positives,candidates,config)
        report=dict(state='complete',matching_policy=POLICY,observation_policy=OBSERVATION_POLICY,
            all_train=True,negative_start_stride=1,reference_start_stride=1,
            expert_references_enabled=True,stage1_admission_unchanged=True,
            negative_suffix_inheritance=False,negative_tail_padding=False,synthetic_positive_tail_excluded=True,
            pairing=config.to_dict()['pairing'],source_stage1_prepared=str(sources[0]),
            source_stage1_sha256=digest,source_negative_windows=str(sources[1]),source_negative_windows_sha256=original_sha,
            feature_report=feature_report,training_started=False,formal_training_prepared=False,
            **summarize(negatives,positives,pairs,rows))
        save_windows(out/'windows_with_features.jsonl',positives+negatives)
        write_jsonl(out/'pairs.jsonl',pairs)
        write_jsonl(out/'negative_window_results.jsonl',rows)
        save_windows(out/'matched_negative_windows.jsonl',[r for r in negatives if r.sample_id in {p['negative_id'] for p in pairs}])
        (out/'config.json').write_text(json.dumps(config.to_dict(),indent=2))
        if sha256(negative_windows)!=original_sha: raise ValueError('Original negative input changed')
        audit_prepared(positive_prepared)
        (out/'report.json').write_text(json.dumps(report,indent=2))
        update('complete',paired_windows=report['paired_negative_windows'],paired_segments=report['paired_negative_segments'])
        return report
    except Exception as exc:
        update('failed',error=str(exc))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('positive-prepared','negative-windows','checkpoint','config','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--max-cuda-memory-gb',type=float,default=8.)
    p.add_argument('--batch-size',type=int,default=2)
    a=p.parse_args()
    print(json.dumps(run(a.positive_prepared,a.negative_windows,a.checkpoint,Config.load(a.config),a.output,
        device=a.device,memory_gib=a.max_cuda_memory_gb,batch_size=a.batch_size),indent=2))


if __name__=='__main__': main()
