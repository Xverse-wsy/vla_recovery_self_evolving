"""Prepare a NEW A60-recipe Stage1 run from existing, unchanged strict admission.

No conversion, matching, simulator, optimizer or training is run here.
The historical prepared/config files remain immutable and must not be resumed.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

import convert_selected_success as core


def aligned_config(pair_code, previous):
    from recovery_pair.config import Config
    value = core.read(pair_code / 'configs/libero_pi05_positive_fm.json')
    if value.get('training_recipe') != 'stage1_a60_sft_v1':
        raise ValueError('Synchronize the A60-aligned training code/config first')
    # These fields define DATA admission, not the training recipe.
    for key in ('coverage', 'pairing', 'split_fractions', 'conservative_task_split',
                'cap_success_admission', 'matching_policy', 'negative_query_policy',
                'state_stride', 'native_control_hz'):
        value[key] = previous[key]
    value['split_fractions'] = tuple(value['split_fractions'])
    from recovery_pair.config import MatchThresholds
    for key in ('coverage', 'pairing'):
        value[key] = MatchThresholds(**value[key])
    return Config(**value).validate()


def compare_training_data(previous, current):
    from recovery_pair.schema import load_windows
    old = {r.sample_id: r for r in load_windows(previous / 'training_pool.jsonl')}
    new = {r.sample_id: r for r in load_windows(current / 'training_pool.jsonl')}
    if old.keys() != new.keys():
        raise ValueError('A60 alignment changed the admitted training starts')
    for sid, a in old.items():
        b = new[sid]
        for key in ('image_refs', 'proprio', 'instruction', 'start_step', 'controller', 'split', 'bank_label'):
            if getattr(a, key) != getattr(b, key):
                raise ValueError('A60 alignment changed observations/provenance: '+key)
        for key in ('fm_scope_range', 'fm_real_steps', 'fm_repeated_steps', 'fm_tail_mode'):
            if a.source.get(key) != b.source.get(key):
                raise ValueError('A60 alignment changed segment/tail rules: '+key)
        pa, pb = a.payload(), b.payload()
        if pa.keys() != pb.keys() or any(not np.array_equal(pa[k], pb[k]) for k in pa):
            raise ValueError('A60 alignment changed recorded actions or masks')
    return dict(windows_checked=len(new), identical_start_ids=True,
        identical_observations=True, identical_actions_and_masks=True,
        identical_admission_and_tail_rules=True)


def align(root, source, pair_code, output=None):
    from recovery_pair.a60_recipe import validate_training_settings
    from recovery_pair.pipeline import audit_prepared
    from recovery_pair.positive_fm import prepare_positive_fm, audit_source_alignment
    root, source, pair_code = map(lambda p: Path(p).resolve(), (root, source, pair_code))
    out = Path(output).resolve() if output else source / 'stage1_a60_sft_v1'
    if out.exists():
        raise FileExistsError('Keep completed assets immutable; use a NEW --output: '+str(out))
    previous = core.read(source / 'algorithm.json')
    config = aligned_config(pair_code, previous)
    recipe = validate_training_settings(config)
    protected = [source / p for p in ['algorithm.json', 'prepared/checksums.json',
        'matching/admission/config.json', 'matching/admission/positive_pool.jsonl',
        'matching/admission/cap_success_suffixes.jsonl']]
    protected_hashes = {str(p): core.sha(p) for p in protected}
    out.mkdir(parents=True)
    core.save(out / 'status.json', dict(state='preparing_A60_recipe', training_started=False))
    core.save(out / 'algorithm.json', config.to_dict())
    report = prepare_positive_fm(source / 'matching/admission', config, out / 'prepared',
        all_train=True, expert_dataset=root / 'dataset_subsets/rlinf_pi05_libero_fewshot40_20261021',
        expected_expert_episodes=40)
    unchanged = compare_training_data(source / 'prepared', out / 'prepared')
    norm = core.read(source / 'sources.json')['norm_stats']
    records, pairs, digest = audit_prepared(out / 'prepared', config)
    source_audit = audit_source_alignment(records, norm)
    for p, digest_before in protected_hashes.items():
        if core.sha(Path(p)) != digest_before:
            raise ValueError('Historical asset changed: '+p)
    core.save(out / 'prepared/source_alignment_audit.json', source_audit)
    core.save(out / 'recipe.json', recipe)
    result = dict(state='complete', training_started=False, source=str(source),
        run=str(out), training_recipe=config.training_recipe, recipe=recipe,
        positive_windows=report['positive_windows'], unique_real_frames=report['unique_real_frames'],
        training_budget=report['training_budget'], data_unchanged=unchanged,
        source_alignment=source_audit, protected_sha256=protected_hashes, prepared_sha256=digest,
        cap_success_segments=report['admitted_cap_success_segments'],
        all_train=True, negative_windows=0, pairs=len(pairs))
    core.save(out / 'report.json', result)
    (out / 'README.md').write_text(
        '# Active Stage1 training: A60 recipe\n\n'
        'Use algorithm.json and prepared/ in THIS directory. The parent directory\n'
        'may contain a historical frozen-VLM training config; do not train or resume it.\n\n'
        'Full model including VLM; native train-time image preprocessing; native FM\n'
        'mean over H=10 and model_dim=32. Same original few-shot initialization/norm.\n'
        'AdamW lr=5e-6, betas=(0.9,0.95), eps=1e-8, weight_decay=1e-10, clip=1;\n'
        'warmup100 then native openpi_cosine to5e-7; Euler K=5 for inference.\n'
        'Global batch128, default microbatch2 (adjust for memory without changing global batch).\n'
        'All dense positives once per epoch, 50 complete epochs, no validation or pair loss.\n\n'
        f'Current data: {report["positive_windows"]} starts, {report["unique_real_frames"]} unique real actions, '
        f'{report["training_budget"]["expected_updates"]} total updates. No training has started.\n'
        'Strict matching and source actions are unchanged; VLA-progress tails first remove10\n'
        'real actions/starting frames, then require a full real H10 horizon. Terminal expert,\n'
        'autonomous-success and admitted CAP-success tails may repeat the final action.\n\n'
        'Run from the COMPLETE adaptation repo pi05_recovery_pair/ directory. Set PAIR_DATA_ROOT,\n'
        'PAIR_CHECKPOINT, PAIR_NORM_STATS, PAIR_RUN to this directory, PAIR_PYTHON,\n'
        'CUDA_VISIBLE_DEVICES, NPROC_PER_NODE and an explicit PAIR_TRAIN_MEMORY_GIB.\n'
        'Then bash scripts/run_positive_fm_pipeline.sh train. For a bounded smoke set\n'
        'PAIR_MAX_UPDATES=2; leave it unset for the formal50 epochs. Never resume a legacy run.\n\n'
        'On another machine, rebuild indexes from the local datasets. Run the scheme1 banks,\n'
        'match, prepare, then align-training stages; do not reuse A800 absolute paths.\n')
    core.save(out / 'status.json', dict(state='complete', training_started=False,
        training_recipe=config.training_recipe, positive_windows=report['positive_windows']))
    core.save(source / 'active_training.json', dict(training_recipe=config.training_recipe,
        run_relative_path=str(out.relative_to(source)) if out.is_relative_to(source) else str(out),
        report_sha256=core.sha(out / 'report.json'), training_started=False))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'source', 'pair-code'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    sys.path.insert(0, str(args.pair_code.resolve()))
    print(json.dumps(align(args.root, args.source, args.pair_code, args.output), indent=2))


if __name__ == '__main__':
    main()
