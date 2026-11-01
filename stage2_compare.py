"""Compare fewshot/Stage1/Stage2 on exactly the same LIBERO initial states."""
import argparse
import json
from pathlib import Path
from stage2_eval_jobs import jobs, summarize, save, SUITES


def compare(folders, output, allow_subset=False):
    loaded={}
    for label,folder in folders.items():
        folder=Path(folder)
        status=json.loads((folder/'status.json').read_text())
        if status['state']!='complete':raise ValueError('Evaluation is incomplete: '+label)
        run=json.loads((folder/'run.json').read_text())
        if run['label']!=label:raise ValueError('Evaluation label mismatch')
        rows=[json.loads(p.read_text()) for p in (folder/'episodes').glob('*/*/*.json')]
        if any(r['state']!='complete' for r in rows):raise ValueError('Incomplete episode')
        by_id={(r['suite'],r['task_id'],r['trial_id']):r for r in rows}
        if len(by_id)!=len(rows) or set(by_id)!=set(jobs(run['suites'],run['tasks'],run['trials'])):
            raise ValueError('Episode inventory is incomplete or duplicated')
        if not allow_subset and (run['scope']!='full2000' or len(rows)!=2000):
            raise ValueError('Formal comparison requires the full 2000 episodes; --allow-subset is smoke only')
        loaded[label]=(run,by_id,summarize(rows))
    if set(loaded)!={'fewshot','stage1','stage2'}:raise ValueError('Three named models required')
    baseline=loaded['fewshot']
    for label,(run,rows,summary) in loaded.items():
        for key in ('suites','tasks','trials','seed','max_steps','flow_steps','action_chunk','norm_sha256','benchmark','wcm_enabled','capx_enabled'):
            if run[key]!=baseline[0][key]:raise ValueError('Incompatible evaluation setting: '+key)
        if set(rows)!=set(baseline[1]):raise ValueError('Different evaluation episodes')
        for key,row in rows.items():
            if row['initial_state_sha256']!=baseline[1][key]['initial_state_sha256']:
                raise ValueError('Different initial state: '+str(key))
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    result={label:summary for label,(_,_,summary) in loaded.items()}
    save(out/'comparison.json',result)
    lines=['| Model | Episodes | Successes | Success rate |','|---|---:|---:|---:|']
    for label,summary in result.items():
        lines.append(f"| {label} | {summary['episodes']} | {summary['successes']} | {summary['success_rate']:.2%} |")
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('fewshot','stage1','stage2','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--allow-subset',action='store_true')
    a=p.parse_args()
    print(json.dumps(compare({k:getattr(a,k) for k in ('fewshot','stage1','stage2')},a.output,a.allow_subset),indent=2))


if __name__=='__main__':main()
