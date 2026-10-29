"""CPU integration check of plans, stage ordering, handoff and failure/resume."""
import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from eval_jobs import SUITES, save, summarize
from experiment import build_plan, execute


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    report = dict(state='complete', gpu_used=False, formal_training_started=False,
                  simulated_training_backend=True)
    with tempfile.TemporaryDirectory() as temporary:
        os.environ.update(PI05_EXPERIMENT_OUTPUT=temporary, PI05_GPUS='1,3',
                          PI05_EVAL_GPUS='1,3', PI05_MICRO_BATCH='2', PI05_ACCUMULATION='25',
                          PI05_GLOBAL_BATCH='', PI05_TRAIN_NUM_NODES='1', PI05_RAY_ADDRESS='')
        formal = build_plan()
        assert [r['max_updates'] for r in formal['training'].values()] == [4950, 26700]
        os.environ.update(PI05_GPUS=','.join(map(str, range(16))), PI05_ACCUMULATION='4')
        sixteen = build_plan()
        assert all(r['global_batch'] == 128 and r['epochs'] == 50 for r in sixteen['training'].values())
        assert [r['max_updates'] for r in sixteen['training'].values()] == [3900, 20900]
        report['planned_updates_global_batch_128'] = {'a60': 3900, 'b80': 20900}
        os.environ.update(PI05_GPUS='1,3', PI05_ACCUMULATION='25')
        # Real paths with new directory names must feed training and eval alike.
        keys = {'PI05_AUTHOR40_DIR': 'author40', 'PI05_AUTONOMOUS20_DIR': 'autonomous20',
                'PI05_RECOVERY20_DIR': 'recovery20', 'PI05_BASE_CHECKPOINT': 'base_checkpoint',
                'PI05_NORM_STATS': 'norm_stats'}
        saved = {key: os.environ.get(key) for key in keys}
        from evaluate import training_overlap
        expected_overlap = training_overlap(Path(os.environ['PI05_ROOT']), 'b80')
        for key, name in keys.items():
            alias = Path(temporary) / ('alias_' + name)
            alias.symlink_to(formal['paths'][name])
            os.environ[key] = str(alias)
        aliased = build_plan()
        assert all(aliased['training'][name]['dataset'] == formal['training'][name]['dataset'] for name in ['a60', 'b80'])
        assert training_overlap(Path('/unused'), 'b80') == expected_overlap
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        # Exercise the actual Bash config/preview entrypoint with 16 planned ranks.
        preview = Path(temporary) / 'preview_output'
        config = Path(temporary) / 'preview.env'
        values = dict(PI05_ROOT=os.environ['PI05_ROOT'], PI05_TRAIN_PYTHON=os.environ['PI05_TRAIN_PYTHON'],
                      PI05_EVAL_PYTHON=os.environ['PI05_EVAL_PYTHON'], PI05_EXPERIMENT_OUTPUT=str(preview),
                      PI05_GPUS=','.join(map(str, range(16))), PI05_MICRO_BATCH='2', PI05_ACCUMULATION='4',
                      PI05_GLOBAL_BATCH='', PI05_EVAL_GPUS='1,3', PI05_TRAIN_NUM_NODES='1')
        config.write_text('\n'.join(f'export {key}={shlex.quote(value)}' for key, value in values.items())+'\n')
        result = subprocess.run(['bash', str(Path(__file__).resolve().parent / 'run_experiment.sh'), 'plan'],
                                env=dict(os.environ, PI05_EXPERIMENT_CONFIG=str(config)), capture_output=True, text=True, check=True)
        assert '3900' in result.stdout and '20900' in result.stdout and not preview.exists()
        report.update(shared_path_overrides_verified=True, shell_plan_preview_verified=True)
        plan = build_plan(smoke=True)
        events, fail_once = [], [False]

        def value(command, flag):
            return command[command.index(flag) + 1]

        def backend(command, env):
            if command[1].endswith('run_train.sh'):
                label = value(command, '--experiment')
                request = current['training'][label]
                events.append('train_' + label)
                assert env['PI05_GPUS'] == request['gpus']
                folder = Path(request['output'])
                checkpoint = Path(request['checkpoint'])
                for name, data in [('dataset_manifest.json', request['dataset']),
                                   ('initial_weights.json', [{'optimizer_steps': 0, 'model_digest': 'original-base'}])]:
                    save(folder / name, data)
                # JSON is a YAML subset; this is read by the real final-status guard.
                save(folder / 'resolved_config.yaml', {'actor': {'model': {'model_path': request['initial_checkpoint']}}})
                status = {key: request[key] for key in ['epochs', 'global_batch', 'micro_batch', 'max_updates']}
                status.update(state='complete', exact_epochs_verified=True, checkpoint=str(checkpoint))
                if label == 'b80' and fail_once[0]:
                    fail_once[0] = False
                    save(checkpoint.parent / 'global_step_2/COMPLETE.json', {'step': 2})
                    save(folder / 'status.json', status | {'state': 'failed'})
                    raise RuntimeError('simulated interruption')
                if '--resume' in command:
                    assert value(command, '--resume').endswith('global_step_2')
                save(checkpoint / 'COMPLETE.json', {'step': request['max_updates']})
                (checkpoint / 'actor').mkdir(exist_ok=True)
                save(folder / 'status.json', status)
            elif command[1].endswith('run_eval_parallel.sh'):
                label = value(command, '--label')
                events.append('eval_' + label)
                checkpoint = value(command, '--checkpoint')
                expected = current['paths']['base_checkpoint'] if label == 'fewshot' else str(Path(current['training'][label]['checkpoint']) / 'actor')
                assert checkpoint == expected, 'Wrong model was sent to evaluation'
                folder = Path(value(command, '--output'))
                if '--resume' in command:
                    assert (folder / 'status.json').is_file()
                    return
                save(folder / 'parallel.json', {})
                save(folder / 'run.json', dict(label=label, checkpoint=checkpoint,
                     checkpoint_sha256=label, suites=SUITES, tasks=[0], trials=1, seed=0))
                records = []
                for suite in SUITES:
                    record = dict(state='complete', suite=suite, task_id=0, trial_id=0,
                                  success=True, steps=5, step_limit=240,
                                  initial_state_sha256=suite, rollout_training_init_overlap=False)
                    records.append(record)
                    save(folder / 'episodes' / suite / 'task_00/trial_00.json', record)
                save(folder / 'status.json', dict(state='complete', summary=summarize(records)))
            else:
                events.append('compare')
                subprocess.run(command, env=env, check=True, stdout=subprocess.DEVNULL)

        current = plan
        execute(current, run_command=backend)
        assert events == ['train_a60', 'train_b80', 'eval_fewshot', 'eval_a60', 'eval_b80', 'compare']
        execute(current, resume=True, run_command=backend)
        assert events.count('train_a60') == events.count('train_b80') == 1
        comparison = json.loads((Path(current['output']) / 'comparison/comparison.json').read_text())
        assert comparison['matched_evaluation'] and len(comparison['models']) == 3
        changed = copy.deepcopy(current)
        changed['training']['b80']['global_batch'] += 1
        try:
            execute(changed, run_command=backend)
            raise AssertionError('Changed plan accepted')
        except AssertionError as exc:
            assert 'changed' in str(exc)
        # Formal resume uses a complete intermediate checkpoint; no real model runs.
        current = json.loads(json.dumps(formal).replace(temporary, temporary + '/interrupted'))
        fail_once[0] = True
        before = len(events)
        try:
            execute(current, stage='train', run_command=backend)
            raise AssertionError('Failure did not stop the pipeline')
        except RuntimeError as exc:
            assert 'simulated interruption' in str(exc)
        assert events[before:] == ['train_a60', 'train_b80']
        execute(current, stage='train', resume=True, run_command=backend)
        assert events[before:] == ['train_a60', 'train_b80', 'train_b80']
        report.update(stage_order_verified=True, correct_model_handoff=True,
                      completed_training_not_repeated=True, partial_training_resumed=True,
                      failure_stops_later_stages=True, changed_plan_rejected=True,
                      three_model_comparison_verified=True)
    if args.output:
        save(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
