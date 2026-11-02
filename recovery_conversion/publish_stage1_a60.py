"""Publish the corrected, versioned Stage1 recipe without re-uploading trajectories."""
import argparse
import datetime
import json
from pathlib import Path
import re

import convert_selected_success as core
from upload_scheme1 import archive, REPO, PREFIX


TRAINING_FILES = [
    'AGENTS.md', 'config/base.yaml', 'pi05_recovery_pair/AGENTS.md', 'pi05_recovery_pair/README_STAGE1.md',
    'pi05_recovery_pair/README.md', 'pi05_recovery_pair/RUN_POSITIVE_FM.md',
    'pi05_recovery_pair/DATA_CONTRACT.md',
    'pi05_recovery_pair/configs/libero_pi05_positive_fm.json',
    'pi05_recovery_pair/recovery_pair/a60_recipe.py',
    'pi05_recovery_pair/recovery_pair/a60_data_validation.py',
    'pi05_recovery_pair/recovery_pair/config.py', 'pi05_recovery_pair/recovery_pair/engine.py',
    'pi05_recovery_pair/recovery_pair/openpi_backend.py', 'pi05_recovery_pair/recovery_pair/trainer.py',
    'pi05_recovery_pair/recovery_pair/fm_validation.py',
    'pi05_recovery_pair/scripts/positive_fm.env.example',
    'pi05_recovery_pair/scripts/run_positive_fm_pipeline.sh',
    'pi05_recovery_pair/tests/test_a60_recipe.py', 'pi05_recovery_pair/tests/test_pipeline_script.py',
]


def scan(paths):
    secret = re.compile(r'hf_[A-Za-z0-9]{25,}|sk-(?:proj-)?[A-Za-z0-9_-]{32,}')
    for p in paths:
        if p.is_file() and p.suffix in ('.py', '.md', '.json', '.jsonl', '.txt', '.sh', '.yaml'):
            if secret.search(p.read_text()):
                raise ValueError('Possible private credential: '+p.name)


def publish(root, pair_code):
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    source = root / 'dataset_subsets/pi05_recovery_scheme1_stage1_20261102'
    active = source / 'stage1_a60_sft_v1'
    report = core.read(active / 'report.json')
    assert report['state'] == 'complete' and not report['training_started']
    assert report['data_unchanged']['identical_actions_and_masks']
    assert report['training_recipe'] == 'stage1_a60_sft_v1'
    gpu = core.read(active / 'gpu_validation_report.json')
    assert gpu['state'] == 'complete' and gpu['full_model_sft']
    assert gpu['checkpoint_resume_verified'] and gpu['uninterrupted_equivalence_verified']
    assert gpu['prepared_sha256'] == report['prepared_sha256']
    inputs = core.read(active / 'a60_data_parity.json')
    assert inputs['state'] == 'complete' and len(inputs['probes']) >= 5
    assert all(p['native_preprocessed_observation_equal'] and p['state_fp32_equal'] for p in inputs['probes'])
    package = pair_code.parent
    code_paths = [package / rel for rel in TRAINING_FILES]
    for p in code_paths:
        assert p.is_file(), p
    scan(code_paths)
    scan(active.rglob('*'))
    scan(core.HERE.rglob('*'))
    native_bundle = source / 'stage1_a60_sft_v1.tar.gz'
    patch_bundle = source / 'stage1_training_patch_a60_sft_v1.tar.gz'
    rebuild_bundle = source / 'rebuild_scripts_a60_sft_v1.tar.gz'
    archive(source, native_bundle, ['stage1_a60_sft_v1', 'active_training.json'])
    archive(package, patch_bundle, TRAINING_FILES)
    archive(core.HERE, rebuild_bundle, ['build_scheme1.py', 'align_stage1_a60.py',
        'publish_stage1_a60.py', 'convert_selected_success.py', 'convert_fixed_failures.py',
        'run_scheme1_build.sh', 'upload_scheme1.py', 'README_SCHEME1_STAGE1.md', 'tasks.json',
        'historical_engine', 'segmentation_engine'])
    text = ('# Scheme1 recovery40: active A60-aligned Stage1\n\n'
        '**For training use stage1_a60_sft_v1.tar.gz.** Its algorithm.json enables full-model\n'
        'SFT including VLM, native train preprocessing, H10x32 FM mean, A60 AdamW/warmup/cosine,\n'
        'global batch128 and50 complete epochs. Original trajectory folders and strict admission\n'
        'are unchanged. The old stage1_libraries.tar.gz contains a historical frozen-VLM config;\n'
        'do not train/resume that recipe.\n\n'
        'Migration: clone the COMPLETE adaptation repo, extract stage1_training_patch_a60_sft_v1.tar.gz\n'
        'into its ROOT (preserve config/ and pi05_recovery_pair/). Extract rebuild_scripts_a60_sft_v1.tar.gz\n'
        'into a separate tools directory. Follow README_SCHEME1_STAGE1.md to rebuild local indexes;\n'
        'A800 absolute paths are not portable. No formal training has started.\n\n'
        'A800 GPU0 validation:4 bounded full-model FM updates, test global batch2/microbatch1;\n'
        'resume and continuous final weights/metrics are exactly equal; peak allocated32.42GiB.\n'
        'Production batch128/microbatch2 unchanged. Native-input checks and GPU report included.\n'
        'No formal50-epoch training, H100/16-GPU run or LIBERO evaluation was executed.\n\n'
        f'Current positives: {report["positive_windows"]} dense starts, {report["unique_real_frames"]} unique real actions, '
        f'{report["cap_success_segments"]} admitted CAP segments. '
        f'Budget: {report["training_budget"]["expected_updates"]} updates /50 epochs.\n\n'
        'Four original/full/segmented recovery folders remain separate; original and converted\n'
        'success labels remain distinct. VLA progress trims10 physical tail actions/frames before\n'
        'H10 windowing. Matching remains1cm/5deg/RMS0.10/per-finger5mm/both-image0.002,\n'
        'earliest passing start through the SAME actually successful program, no privileged input.\n')
    readme = source / 'README_ACTIVE_STAGE1.md'
    readme.write_text(text)
    files = {p.name: p for p in [native_bundle, patch_bundle, rebuild_bundle, source / 'active_training.json']}
    files['stage1_a60_report.json'] = active / 'report.json'
    files['README.md'] = readme
    api = HfApi()
    assert api.repo_info(REPO, repo_type='dataset').id == REPO
    commit = api.create_commit(repo_id=REPO, repo_type='dataset',
        commit_message='Align active Stage1 training recipe with A60; preserve trajectories and admission',
        operations=[CommitOperationAdd(path_in_repo=PREFIX+'/'+name, path_or_fileobj=str(p))
                    for name, p in files.items()])
    remote = {p.path: p for p in api.get_paths_info(REPO,
        [PREFIX+'/'+name for name in files], repo_type='dataset', revision=commit.oid)}
    entries = []
    for name, p in files.items():
        f = remote[PREFIX+'/'+name]
        assert f.size == p.stat().st_size
        digest = core.sha(p)
        if f.lfs:
            sha = f.lfs.get('sha256') if isinstance(f.lfs, dict) else f.lfs.sha256
            assert sha == digest
        else:
            downloaded = hf_hub_download(REPO, PREFIX+'/'+name, repo_type='dataset', revision=commit.oid)
            assert core.sha(Path(downloaded)) == digest
        entries.append(dict(path=PREFIX+'/'+name, size=f.size, sha256=digest))
    result = dict(state='verified', remote_verified=True, training_recipe='stage1_a60_sft_v1',
        repo=REPO, revision=commit.oid, files=entries,
        updated_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
    job = root / 'hf_upload_jobs/pi05_recovery_scheme1_20261102'
    core.save(job / 'stage1_a60_sft_v1.uploaded.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--pair-code', required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(publish(args.root.resolve(), args.pair_code.resolve()), indent=2))


if __name__ == '__main__':
    main()
