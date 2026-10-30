from dataclasses import replace
import json

from recovery_pair.numeric_preview import preview_numeric
from recovery_pair.schema import save_windows


def test_numeric_preview_counts_only_cap_successes_without_images_and_reports_uncalibrated_rms(window,config,tmp_path):
    ref=window('vla')
    matched=window('cap_match','code',position=.02)
    rms_far=window('cap_rms_far','code',position=.01)
    rms_far.standardized_proprio[-1]=10.
    outside=window('cap_outside','code',position=.04)
    failure=window('cap_failure','code','negative')
    for r in (ref,matched,rms_far,outside,failure):
        r.visual_features={}
        r.image_refs={}
        r.source['critical_state_verified']=False
        r.phase=r.target_object=r.held_object=r.contact_state=None
    windows=tmp_path/'windows.jsonl'
    save_windows(windows,[ref,matched,rms_far,outside,failure])
    original=windows.read_bytes()
    uncalibrated=replace(config,coverage=replace(config.coverage,proprio_rms=None,image_cosine=None))
    result=preview_numeric(windows,uncalibrated,tmp_path/'coarse')
    assert result['cap_success_windows']==3 and result['matched_cap_windows']==2
    assert result['matched_cap_segments']==2
    assert result['proprio_gate']=='not_applied_uncalibrated'
    assert not result['image_filter_applied'] and not result['accepted_pool_modified']
    assert windows.read_bytes()==original
    strict=preview_numeric(windows,uncalibrated,tmp_path/'with_rms',proprio_rms=.2)
    assert strict['matched_cap_windows']==1 and strict['proprio_gate']=='applied'


def test_numeric_preview_limits_candidates_and_respects_public_instruction_and_split(window,config,tmp_path):
    refs=[window('r'+str(i)) for i in range(60)]
    other=window('other_instruction','code')
    other.instruction='open drawer'
    other_split=window('other_split','code',split='test')
    query=window('cap','code')
    path=tmp_path/'windows.jsonl'
    save_windows(path,refs+[query,other,other_split])
    report=preview_numeric(path,config,tmp_path/'preview')
    assert report['matched_cap_windows']==1
    assert report['all_numeric_pairs_passing']==60 and report['retained_candidates']==50
    assert report['queries_without_same_instruction_reference']==2
    rows=[json.loads(s) for s in (tmp_path/'preview/numeric_candidates.jsonl').read_text().splitlines()]
    assert len(next(r for r in rows if r['sample_id']=='cap')['candidates'])==50
