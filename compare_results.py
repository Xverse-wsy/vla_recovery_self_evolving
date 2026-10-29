"""Summarize matched completed evaluations; refuse mixed settings or states."""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fewshot',type=Path,required=True)
    ap.add_argument('--a60',type=Path,required=True)
    ap.add_argument('--b80',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    results={}
    identities=[]
    shared=[]
    for label in ('fewshot','a60','b80'):
        folder=getattr(args,label)
        status=json.loads((folder/'status.json').read_text())
        config=json.loads((folder/'run.json').read_text())
        assert status['state']=='complete',label
        assert config['label']==label
        shared.append({k:v for k,v in config.items() if k not in ('label','checkpoint','checkpoint_sha256')})
        records=[json.loads(p.read_text()) for p in sorted((folder/'episodes').glob('*/*/*.json'))]
        assert all(r['state']=='complete' for r in records)
        ids=[(r['suite'],r['task_id'],r['trial_id'],r['initial_state_sha256'],r['step_limit']) for r in records]
        assert len(ids)==len(set(ids))
        identities.append(ids)
        assert len(records)==len(config['suites'])*len(config['tasks'])*config['trials']
        results[label]=status['summary']
    assert shared[0]==shared[1]==shared[2], 'Inference/evaluation settings differ'
    assert identities[0]==identities[1]==identities[2], 'Evaluation initial states/budgets differ'
    report=dict(state='complete',matched_evaluation=True,configuration=shared[0],models=results)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['| Model | Spatial | Object | Goal | LIBERO-10 | Overall |',
           '|---|---:|---:|---:|---:|---:|']
    for label,result in results.items():
        cells=[f"{100*result['suites'][s]['success_rate']:.2f}%" for s in
               ['libero_spatial','libero_object','libero_goal','libero_10'] if s in result['suites']]
        if len(cells)==4:
            lines.append('| '+label+' | '+' | '.join(cells)+f" | {100*result['success_rate']:.2f}% |")
    (args.output/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
