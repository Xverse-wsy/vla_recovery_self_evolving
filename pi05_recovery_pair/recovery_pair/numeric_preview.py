"""Read-only numerical candidate counts before any image filtering or admission."""
from collections import Counter, defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path
import numpy as np

from .features import sha256
from .matching import numeric_match, instruction_key, OBSERVATION_POLICY
from .schema import load_windows
from .splits import audit_split


def quantiles(values):
    if not values:
        return None
    return dict(zip(('min','p25','median','p75','p90','p95','max'),
                    map(float,np.quantile(values,[0,.25,.5,.75,.9,.95,1]))))


def preview_numeric(windows, config, output, proprio_rms=None):
    config.validate()
    source=Path(windows).expanduser().resolve()
    before=sha256(source)
    records=load_windows(source)
    audit_split(records)
    if len({r.sample_id for r in records})!=len(records):
        raise ValueError('Duplicate sample IDs')
    references=[r for r in records if r.controller=='VLA' and r.training_quality=='positive']
    queries=[r for r in records if r.controller=='code' and r.training_quality=='positive']
    rms=config.coverage.proprio_rms if proprio_rms is None else proprio_rms
    if rms is not None and (not math.isfinite(rms) or rms<=0):
        raise ValueError('RMS threshold must be finite and positive')
    # Infinity is only an INTERNAL diagnostic bypass of the unset RMS gate.
    # Never export it as a calibrated config, accepted training pool or match.
    thresholds=replace(config.coverage,proprio_rms=float('inf') if rms is None else rms)
    groups=defaultdict(list)
    for r in references:
        groups[(r.split,instruction_key(r))].append(r)
    out=Path(output).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=False)
    statuses=out/'status.json'
    statuses.write_text(json.dumps({'state':'counting_numeric_candidates'}))
    try:
        rows=[]
        rejected=Counter()
        per_instruction={}
        per_segment={}
        per_split={}
        best_rms=[]
        minimum_rms=[]
        compared=passing=retained=0
        def tally(container,key,matched):
            item=container.setdefault(key,dict(total_windows=0,matched_windows=0))
            item['total_windows']+=1
            item['matched_windows']+=int(matched)
        for q in queries:
            candidates=[]
            refs=groups.get((q.split,instruction_key(q)),[])
            compared+=len(refs)
            for ref in refs:
                distances,reason=numeric_match(q,ref,thresholds)
                if reason:
                    rejected[reason]+=1
                    continue
                score=sum(distances[k]/getattr(thresholds,k)
                          for k in ('position_m','rotation_deg','proprio_rms'))
                candidates.append(dict(sample_id=ref.sample_id,episode_id=ref.episode_id,
                    segment_id=ref.source.get('segment_id',ref.sample_id),start_step=ref.start_step,
                    distances=distances,numeric_rank_score=score))
            candidates.sort(key=lambda c:(c['numeric_rank_score'],c['sample_id']))
            matched=bool(candidates)
            passing+=len(candidates)
            kept=candidates[:config.coverage.max_candidates]
            retained+=len(kept)
            if matched:
                best_rms.append(candidates[0]['distances']['proprio_rms'])
                minimum_rms.append(min(c['distances']['proprio_rms'] for c in candidates))
            row=dict(sample_id=q.sample_id,episode_id=q.episode_id,
                segment_id=q.source.get('segment_id',q.sample_id),split=q.split,
                instruction=instruction_key(q),start_step=q.start_step,
                matching_reference_count=len(refs),numeric_candidate_count=len(candidates),
                matched_numeric_only=matched,candidates=kept)
            rows.append(row)
            tally(per_instruction,row['instruction'],matched)
            tally(per_segment,row['segment_id'],matched)
            tally(per_split,q.split,matched)
        matched_rows=[r for r in rows if r['matched_numeric_only']]
        report=dict(state='complete',preview_only=True,observation_policy=OBSERVATION_POLICY,
            input_windows=str(source),input_sha256=before,
            cap_success_windows=len(queries),vla_success_reference_windows=len(references),
            matched_cap_windows=len(matched_rows),
            matched_window_rate=len(matched_rows)/len(queries) if queries else None,
            cap_success_segments=len(per_segment),
            matched_cap_segments=sum(v['matched_windows']>0 for v in per_segment.values()),
            cap_success_episodes=len({r.episode_id for r in queries}),
            matched_cap_episodes=len({r['episode_id'] for r in matched_rows}),
            public_instructions=len(per_instruction),
            matched_instructions=sum(v['matched_windows']>0 for v in per_instruction.values()),
            comparisons_same_instruction_and_split=compared,
            all_numeric_pairs_passing=passing,retained_candidates=retained,
            thresholds=dict(position_m=thresholds.position_m,rotation_deg=thresholds.rotation_deg,
                proprio_rms=rms,max_candidates=thresholds.max_candidates),
            proprio_gate='not_applied_uncalibrated' if rms is None else 'applied',
            numeric_ranking='sum(distance/threshold); uncalibrated RMS has zero ranking weight',
            best_geometric_reference_rms=quantiles(best_rms),
            minimum_rms_per_matched_query=quantiles(minimum_rms),
            per_instruction=per_instruction,per_segment=per_segment,per_split=per_split,
            comparison_rejections=dict(rejected),
            queries_without_same_instruction_reference=sum(r['matching_reference_count']==0 for r in rows),
            image_filter_applied=False,training_started=False,accepted_pool_modified=False)
        (out/'numeric_candidates.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        (out/'config.json').write_text(json.dumps(config.to_dict(),indent=2))
        if sha256(source)!=before:
            raise RuntimeError('Input window index changed while counting')
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
        statuses.write_text(json.dumps({'state':'complete','matched_windows':len(matched_rows),'total_windows':len(queries)}))
        return report
    except Exception as exc:
        statuses.write_text(json.dumps({'state':'failed','error':str(exc)}))
        raise
