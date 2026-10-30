"""Portable pi05-vision/RMS pilot on numerical candidates; no training admission."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

from .features import read_start_image,sha256
from .matching import image_distance,instruction_key,OBSERVATION_POLICY
from .numeric_preview import quantiles
from .schema import load_windows,save_windows,stable_id
from .pi05_vision import Pi05Vision


def threshold_grid(rows):
    result=[]
    for rms in (.05,.1,.15,.2,.3,.5,1.,None):
        for image in (.0005,.001,.002,.005,.01,.02,.05,.1,.2,None):
            accepted=[r for r in rows if (rms is None or r['proprio_rms']<=rms) and
                      (image is None or r['image_worst']<=image)]
            result.append(dict(proprio_rms=rms,image_cosine=image,candidate_pairs=len(accepted),
                cap_windows=len({r['query_id'] for r in accepted}),
                cap_segments=len({r['query_segment'] for r in accepted})))
    return result


def review_sheets(rows,records,out,limit=36):
    ordered=sorted(rows,key=lambda r:(r['image_worst'],r['proprio_rms'],r['pair_id']))
    selected=[ordered[int(i)] for i in np.linspace(0,len(ordered)-1,min(limit,len(ordered)),dtype=int)] if ordered else []
    review=[]
    sheets=[]
    for page in range((len(selected)+5)//6):
        group=selected[page*6:page*6+6]
        canvas=Image.new('RGB',(896,60+len(group)*280),'white')
        draw=ImageDraw.Draw(canvas)
        for col,title in enumerate(('CAP front','VLA front','CAP wrist','VLA wrist')):
            draw.text((col*224+5,8),title,fill='black')
        for offset,row in enumerate(group):
            number=page*6+offset+1
            y=45+offset*280
            q,r=records[row['query_id']],records[row['reference_id']]
            draw.text((4,y),f"#{number} RMS={row['proprio_rms']:.4f} front={row['image_front']:.5f} wrist={row['image_wrist']:.5f}",fill='black')
            draw.text((4,y+13),instruction_key(q)[:125],fill='black')
            for col,(record,view) in enumerate(((q,'front'),(r,'front'),(q,'wrist'),(r,'wrist'))):
                canvas.paste(Image.fromarray(read_start_image(record.image_refs[view])).resize((224,224)),(col*224,y+28))
            draw.text((4,y+254),f"CAP step {q.start_step} gripper={q.proprio[-2:]} | VLA step {r.start_step} gripper={r.proprio[-2:]}",fill='black')
            review.append(dict(review_index=number,pair_id=row['pair_id'],query_id=q.sample_id,
                reference_id=r.sample_id,compatible=None,reviewer=None,evidence=None,
                sheet=f'review_{page+1:02d}.png',row=offset+1,
                allowed_evidence='starting front/wrist RGB and policy proprio only'))
        name=f'review_{page+1:02d}.png'
        canvas.save(out/name)
        sheets.append(name)
    return review,sheets


def trial(windows,numeric_preview,checkpoint,output,device='cpu',max_cuda_memory_gb=8.,batch_size=2,review_count=36):
    out=Path(output).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=False)
    status=out/'status.json'
    try:
        status.write_text(json.dumps({'state':'loading_verified_vision_encoder'}))
        source=Path(windows).resolve()
        preview=Path(numeric_preview).resolve()
        info=json.loads((preview/'report.json').read_text())
        if sha256(source)!=info['input_sha256'] or info.get('observation_policy')!=OBSERVATION_POLICY:
            raise ValueError('Numeric preview source or observation policy mismatch')
        candidate_rows=[json.loads(s) for s in (preview/'numeric_candidates.jsonl').read_text().splitlines()]
        pairs=[(q,c) for q in candidate_rows for c in q['candidates']]
        ids={q['sample_id'] for q,c in pairs}|{c['sample_id'] for q,c in pairs}
        records={r.sample_id:r for r in load_windows(source) if r.sample_id in ids}
        if not pairs or len(records)!=len(ids):
            raise ValueError('Need valid numerical candidates')
        encoder=Pi05Vision(checkpoint,device,max_cuda_memory_gb)
        status.write_text(json.dumps({'state':'extracting_frozen_rgb_features','windows':len(records)}))
        feature_report=encoder.extract(list(records.values()),out/'feature_cache',batch_size)
        save_windows(out/'windows_with_features.jsonl',records.values())
        rows=[]
        for q,c in pairs:
            a,b=records[q['sample_id']],records[c['sample_id']]
            if (a.controller!='code' or b.controller!='VLA' or a.training_quality!='positive' or
                b.training_quality!='positive' or a.split!=b.split or instruction_key(a)!=instruction_key(b)):
                raise ValueError('Candidate is outside CAP-success to VLA-success policy')
            distances,reason=image_distance(a,b)
            if reason:
                raise ValueError(reason)
            delta=np.asarray(a.standardized_proprio)-b.standardized_proprio
            rows.append(dict(pair_id=stable_id(a.sample_id,b.sample_id),query_id=a.sample_id,
                reference_id=b.sample_id,query_segment=q['segment_id'],split=a.split,
                instruction=instruction_key(a),**c['distances'],
                standardized_position_rms=float(np.sqrt(np.mean(delta[:3]**2))),
                standardized_rotation_rms=float(np.sqrt(np.mean(delta[3:6]**2))),
                standardized_gripper_rms=float(np.sqrt(np.mean(delta[6:]**2))),
                image_front=distances['front'],image_wrist=distances['wrist'],
                image_worst=max(distances.values())))
        (out/'joint_distances.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        grid=threshold_grid(rows)
        (out/'threshold_grid.json').write_text(json.dumps(grid,indent=2))
        review,sheets=review_sheets(rows,records,out,review_count)
        (out/'review_template.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in review))
        report=dict(state='complete',pilot_only=True,observation_policy=OBSERVATION_POLICY,
            numeric_candidates=len(rows),cap_windows=len({r['query_id'] for r in rows}),
            encoded_windows=len(records),feature_report=feature_report,
            distributions={k:quantiles([r[k] for r in rows]) for k in
                ('proprio_rms','image_front','image_wrist','image_worst','standardized_gripper_rms')},
            threshold_grid=grid,review_pairs=len(review),review_sheets=sheets,
            calibration_status='awaiting_visible_input_compatibility_reviews',
            selected_thresholds=None,training_started=False,accepted_pool_modified=False,
            inputs=dict(windows=str(source),numeric_preview=str(preview),input_sha256=info['input_sha256']))
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        status.write_text(json.dumps({'state':'complete','candidates':len(rows),'review_pairs':len(review)}))
        return report
    except Exception as exc:
        status.write_text(json.dumps({'state':'failed','error':str(exc)}))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('windows','numeric-preview','checkpoint','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--max-cuda-memory-gb',type=float,default=8.)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--review-count',type=int,default=36)
    a=p.parse_args()
    print(json.dumps(trial(a.windows,a.numeric_preview,a.checkpoint,a.output,a.device,
        a.max_cuda_memory_gb,a.batch_size,a.review_count),indent=2))


if __name__=='__main__':
    main()
