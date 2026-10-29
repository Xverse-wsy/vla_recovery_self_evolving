"""Reject partial validation and ensure all three evals used identical episodes."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, type=Path)
    args = ap.parse_args()
    run = args.run
    reports = dict(state='complete', training={}, evaluations={}, data={})
    datasets = []
    for label in ['a60','b80']:
        data = json.loads((run/'data'/f'{label}_data.json').read_text())
        assert data['state'] == 'complete' and data['matches_native_lerobot']
        datasets.append(data)
        status = json.loads((run/label/'status.json').read_text())
        assert status['state'] == 'complete' and status['exact_epochs_verified']
        roundtrip = json.loads((run/label/'roundtrip_report.json').read_text())
        assert roundtrip['state'] == 'complete'
        initial = json.loads((run/label/'initial_weights.json').read_text())
        assert all(a['model_digest'] != b['model_digest'] for a,b in zip(initial,roundtrip['final'])), 'No actual parameter update'
        reports['data'][label] = {k:v for k,v in data.items() if k!='dataset'} | {
            'episodes':data['dataset']['episodes'],'frames':data['dataset']['frames']}
        reports['training'][label] = status
    assert datasets[0]['dataset']['episodes_index'] == datasets[1]['dataset']['episodes_index'][:60]
    assert datasets[0]['original_norm_sha256'] == datasets[1]['original_norm_sha256']
    sampler = json.loads((run/'data/sampler16_validation.json').read_text())
    assert sampler['state']=='complete' and sampler['exact_global_mean_gradients']
    reports['sampler_16_ranks'] = sampler
    initial_a = json.loads((run/'a60/initial_weights.json').read_text())
    initial_b = json.loads((run/'b80/initial_weights.json').read_text())
    assert [r['model_digest'] for r in initial_a] == [r['model_digest'] for r in initial_b]
    assert all(r['optimizer_steps']==0 for r in initial_a+initial_b)
    identities = []
    for label in ['fewshot','a60','b80']:
        status = json.loads((run/f'eval_{label}/status.json').read_text())
        assert status['state']=='complete' and status['summary']['episodes']==4
        records = [json.loads(p.read_text()) for p in sorted((run/f'eval_{label}/episodes').glob('*/*/*.json'))]
        assert all(r['state']=='complete' and r['steps']>0 for r in records)
        assert all(p.with_suffix('.mp4').stat().st_size>0 for p in (run/f'eval_{label}/episodes').glob('*/*/*.json'))
        identities.append([(r['suite'],r['task_id'],r['trial_id'],r['initial_state_sha256'],r['step_limit']) for r in records])
        reports['evaluations'][label] = status
    assert identities[0] == identities[1] == identities[2]
    reports.update(same_original_initial_weights=True, checkpoint_resume_reproduces_training=True,
                   same_evaluation_states_and_budgets=True, formal_training_started=False,
                   scope='two isolated 2-epoch/15-sample training checks and 12 real LIBERO episodes')
    (run/'validation_report.json').write_text(json.dumps(reports,indent=2)+'\n')
    print('ALL_VALIDATIONS_PASSED',run,flush=True)


if __name__=='__main__':
    main()
