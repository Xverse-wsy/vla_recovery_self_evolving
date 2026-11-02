"""Package scheme-1 original recovery episodes; no conversion, replay or training.

Success quotas: LIBERO-10/Goal/Spatial/Object = 3/5/10/2. The user-confirmed
failure selection is immutable (10/8/2/0). Requires the original 400-report batch.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics

SUCCESS_QUOTAS = dict(libero_10=3, libero_goal=5, libero_spatial=10, libero_object=2)
FAILURE_QUOTAS = dict(libero_10=10, libero_goal=8, libero_spatial=2, libero_object=0)
POLICY = 'scheme1_original_recovery_raw_v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def task_key(row):
    return row['suite'], ' '.join(row['instruction'].split())


def inventory(results):
    rows = []
    paths = sorted(results.glob('*/task_*/trial_*/report.json'))
    if len(paths) != 400:
        raise ValueError(f'Expected exactly the specified 400 reports; found {len(paths)}')
    for p in paths:
        r = json.loads(p.read_text())
        if r.get('used_capx') is not True or r.get('success') is not True:
            continue
        owners = r.get('steps_by_owner') or {}
        cap_steps = int(owners.get('capx', 0))
        if cap_steps <= 0:
            raise ValueError(f'Successful intervention without CAP actions: {p}')
        rows.append(dict(source_episode=str(p.parent.resolve()),
            source_report_sha256=digest(p), suite=r['suite'],
            evaluation_task_id=r['task_id'], trial_id=r['trial_id'],
            instruction=r['instruction'], original_success=True,
            original_steps=r['steps'], vla_steps=int(owners.get('pi05', 0)),
            cap_steps=cap_steps, handoff_step=(r.get('handoff') or {}).get('step'),
            original_trajectory_complete=(r.get('trajectory') or {}).get('complete'),
            trajectory_exists=(p.parent/'trajectory/manifest.json').is_file(),
            conversion_status='not_checked_or_executed_in_this_raw_selection',
            label_scope='original_episode_outcome_only'))
    return rows


def select_success(rows, failures, seed=42):
    if len({r['source_episode'] for r in rows}) != len(rows):
        raise ValueError('Duplicate success episodes')
    if any(r.get('original_success') is not True for r in rows):
        raise ValueError('Success pool contains non-success episode')
    if {r['source_episode'] for r in rows} & {r['source_episode'] for r in failures}:
        raise ValueError('Success/failure source overlap')
    groups = defaultdict(list)
    for r in rows:
        groups[task_key(r)].append(r)
    tie = lambda value: hashlib.sha256(f'{seed}|{value}'.encode()).hexdigest()
    for group in groups.values():
        group.sort(key=lambda r: (not r.get('trajectory_exists', False),
            r.get('original_trajectory_complete') is not True, tie(r['source_episode'])))
    fixed = Counter(task_key(r) for r in failures)
    counts, picked = Counter(), []
    for suite, quota in SUCCESS_QUOTAS.items():
        keys = [k for k in groups if k[0] == suite]
        if sum(len(groups[k]) for k in keys) < quota:
            raise ValueError(f'Not enough original successes for {suite}')
        for _ in range(quota):
            eligible = [k for k in keys if counts[k] < len(groups[k])]
            # Cover distinct successful tasks first; fill tasks absent from the
            # failed selection first. Subsequent repeats balance joint counts.
            key = min(eligible, key=lambda k: (counts[k] > 0,
                fixed[k] + counts[k], counts[k], tie(k)))
            row = dict(groups[key][counts[key]])
            counts[key] += 1
            row.update(selection_rank=len(picked)+1,
                       within_task_selection_round=counts[key])
            picked.append(row)
    return picked


def check_original(row, results, success):
    source = Path(row['source_episode']).resolve()
    if results not in source.parents:
        raise ValueError(f'Episode outside specified 400-report batch: {source}')
    report_path = source/'report.json'
    if digest(report_path) != row['source_report_sha256']:
        raise ValueError(f'Original report changed: {source}')
    report = json.loads(report_path.read_text())
    if report.get('success') is not success or report.get('used_capx') is not True:
        raise ValueError(f'Original outcome mismatch: {source}')
    if task_key(report) != task_key(row):
        raise ValueError(f'Task mismatch: {source}')
    return source


def snapshot(source, destination, inventory_file, package_root, mode):
    """Copy metadata; hardlink large immutable files by default, never symlink."""
    destination.mkdir(parents=True, exist_ok=False)
    files = total_bytes = linked = 0
    for directory, dirs, names in os.walk(source, followlinks=False):
        src_dir = Path(directory)
        if any((src_dir/name).is_symlink() for name in dirs + names):
            raise ValueError(f'Refusing symlink inside original episode: {src_dir}')
        relative = src_dir.relative_to(source)
        target_dir = destination/relative
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in sorted(names):
            src, dst = src_dir/name, target_dir/name
            before = src.stat()
            if mode == 'hardlink' and before.st_size >= 1024 * 1024:
                # No silent fallback that could unexpectedly copy tens of GB.
                os.link(src, dst)
                after = dst.stat()
                if (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size):
                    raise ValueError(f'Hardlink integrity failure: {src}')
                method, checksum = 'hardlink_same_inode', None
                linked += 1
            else:
                shutil.copy2(src, dst)
                checksum = digest(src)
                if digest(dst) != checksum:
                    raise ValueError(f'Copy integrity failure: {src}')
                method = 'copy_sha256'
            if src.stat().st_mtime_ns != before.st_mtime_ns or src.stat().st_size != before.st_size:
                raise ValueError(f'Original file changed during snapshot: {src}')
            inventory_file.write(json.dumps(dict(path=str(dst.relative_to(package_root)),
                original=str(src), bytes=before.st_size, method=method, sha256=checksum))+'\n')
            files += 1
            total_bytes += before.st_size
    return dict(files=files, logical_bytes=total_bytes, hardlinked_files=linked)


def audit_trajectory(directory, row):
    root = directory/'trajectory'
    manifest = json.loads((root/'manifest.json').read_text())
    if not (root/'initial.npz').is_file() or not (root/'model.xml').is_file():
        raise ValueError(f'Missing initial state/model: {directory}')
    last, verified_bytes, count = 0, 0, 0
    for shard in manifest['shards']:
        path = (root/shard['file']).resolve()
        if root.resolve() not in path.parents:
            raise ValueError('Shard path escapes trajectory directory')
        if shard['first_step'] != last+1 or shard['last_step'] < shard['first_step']:
            raise ValueError(f'Non-contiguous original shard ranges: {directory}')
        if digest(path) != shard['sha256']:
            raise ValueError(f'Original shard checksum mismatch: {path}')
        last = shard['last_step']
        count += 1
        verified_bytes += path.stat().st_size
    if last != manifest['steps'] or manifest['observations'] != last+1:
        raise ValueError(f'Saved trajectory length inconsistent: {directory}')
    if manifest['complete'] is True and last != row['original_steps']:
        raise ValueError(f'Complete recording differs from executed steps: {directory}')
    return dict(complete=manifest['complete'], saved_steps=last,
        original_executed_steps=row['original_steps'], shards_sha256_verified=count,
        verified_shard_bytes=verified_bytes)


def summarize(rows):
    def length(key):
        values = [r[key] for r in rows]
        return dict(total=sum(values), min=min(values), mean=statistics.mean(values),
                    median=statistics.median(values), max=max(values))
    return dict(episodes=len(rows), by_suite=dict(Counter(r['suite'] for r in rows)),
        tasks=len({task_key(r) for r in rows}), total_steps=length('original_steps'),
        vla_steps=length('vla_steps'), cap_steps=length('cap_steps'),
        incomplete_recordings=sum(r.get('original_trajectory_complete') is not True for r in rows),
        zero_cap_action_episodes=sum(r['cap_steps'] == 0 for r in rows))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def prepare(results, failure_selection, output, seed=42, mode='hardlink'):
    results, failure_selection, output = (Path(p).resolve() for p in (results, failure_selection, output))
    saved = json.loads(failure_selection.read_text())
    confirmation = failure_selection.parent/'CURRENT_SELECTION.json'
    marker = json.loads(confirmation.read_text())
    failure_sha = digest(failure_selection)
    if marker.get('status') != 'user_selected' or marker.get('selection_manifest_sha256') != failure_sha:
        raise ValueError('Requires identical user-confirmed failure20 selection')
    failures = saved['selected']
    # Python <3.10 treats a missing Counter key differently from an explicit 0.
    expected_failures = {suite:n for suite,n in FAILURE_QUOTAS.items() if n}
    if len(failures) != 20 or dict(Counter(r['suite'] for r in failures)) != expected_failures:
        raise ValueError('User-confirmed failure20 quotas differ')
    candidates = inventory(results)
    successes = select_success(candidates, failures, seed)
    all_rows = successes + failures
    if len({r['source_episode'] for r in all_rows}) != 40:
        raise ValueError('Expected 40 distinct original episodes')
    for rows, success in [(successes, True), (failures, False)]:
        for row in rows:
            check_original(row, results, success)
    staging = output.with_name(output.name+'.building')
    if output.exists() or staging.exists():
        raise FileExistsError('Use a new output directory; do not overwrite an earlier selection')
    staging.mkdir(parents=True)
    write_json(staging/'status.json', dict(state='building', policy=POLICY))
    try:
        storage, trajectory_audits = [], []
        with (staging/'files.jsonl').open('w') as file_index:
            for name, rows in [('recovery_success20_raw', successes), ('recovery_failure20_raw', failures)]:
                group = staging/name
                group.mkdir()
                indexed = []
                for index, row in enumerate(rows):
                    episode_name = f"episode_{index:03d}_{row['suite']}_task{row['evaluation_task_id']:02d}_trial{row['trial_id']:02d}"
                    relative = Path(name)/'episodes'/episode_name
                    storage.append(snapshot(Path(row['source_episode']), staging/relative,
                                            file_index, staging, mode))
                    audit = audit_trajectory(staging/relative, row)
                    trajectory_audits.append(dict(snapshot_relative_path=str(relative), **audit))
                    indexed.append(dict(row, dataset_episode_index=index,
                        snapshot_relative_path=str(relative), recording_audit=audit,
                        episode_outcome='success' if name.startswith('recovery_success') else 'failure'))
                    print(json.dumps(dict(progress=name, completed=index+1,
                        total=20, suite=row['suite'], verified_shards=audit['shards_sha256_verified'])), flush=True)
                if name.startswith('recovery_failure'):
                    shutil.copy2(failure_selection, group/'selection.json')
                    shutil.copy2(confirmation, group/'CURRENT_SELECTION.json')
                else:
                    write_json(group/'selection.json', dict(policy=POLICY, selected=rows,
                        label_scope='original_episode_outcome_only', summary=summarize(rows)))
                with (group/'episodes.jsonl').open('w') as f:
                    for row in indexed:
                        f.write(json.dumps(row, ensure_ascii=False)+'\n')
                fields = ['dataset_episode_index', 'suite', 'evaluation_task_id', 'trial_id',
                          'instruction', 'original_steps', 'vla_steps', 'cap_steps',
                          'original_trajectory_complete', 'snapshot_relative_path', 'source_episode']
                with (group/'episodes.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
                    writer.writeheader(); writer.writerows(indexed)
        for row in all_rows:
            check_original(row, results, row['original_success'])
        if digest(failure_selection) != failure_sha:
            raise ValueError('Original confirmed failure selection changed')
        write_json(staging/'trajectory_integrity.json', trajectory_audits)
        report = dict(state='complete', policy=POLICY, seed=seed, script_sha256=digest(Path(__file__)),
            scope='specified_400_episodes_only',
            source_results=str(results), original_failure_selection=str(failure_selection),
            original_failure_selection_sha256=failure_sha, original_failure_selection_unchanged=True,
            success20=summarize(successes), failure20=summarize(failures), combined40=summarize(all_rows),
            original_success_pool=summarize(candidates), raw_only=True,
            conversion_executed=False, replay_executed=False, segmentation_executed=False,
            train_ready=False, files=sum(x['files'] for x in storage),
            logical_bytes=sum(x['logical_bytes'] for x in storage),
            hardlinked_files=sum(x['hardlinked_files'] for x in storage), storage_mode=mode,
            original_shards_sha256_verified=sum(x['shards_sha256_verified'] for x in trajectory_audits),
            saved_raw_steps=sum(x['saved_steps'] for x in trajectory_audits),
            data_integrity='all trajectory shards checked against original SHA256; hardlink inode/size; copied-file SHA256')
        write_json(staging/'summary.json', report)
        (staging/'README.md').write_text('# 方案1：恢复原始成功20＋失败20\n\n'
            '成功20：LIBERO-10 3、Goal 5、Spatial 10、Object 2。失败20保持已确认的10、8、2、0。'
            '分别保存在recovery_success20_raw/和recovery_failure20_raw/，不混合原始episode。\n\n'
            '仅原始轨迹和来源整理；没有OSC转换、回放确认或切分，不能当作可直接训练的数据集。'
            '成功标签只表示原始episode最终成功，不表示其中每个CAP程序或每个动作成功。'
            'CAP总步数包含多次尝试。失败标签只依据原始最终失败；保留两条CAP启动失败/记录不完整的Goal案例。\n\n'
            'episodes.jsonl/csv中的snapshot_relative_path相对于本包根目录。原始报告及轨迹manifest内的'
            '旧绝对路径原样保留作溯源；访问已整理文件应使用本包相对路径。\n\n'
            '大文件默认使用硬链接，原目录删除不会使本包文件消失；请勿原位修改任一份硬链接数据。'
            '迁移本目录时复制实际内容，不能只复制本机绝对路径清单。files.jsonl记录逐文件来源、大小和校验方式。\n')
        write_json(staging/'status.json', dict(state='complete', policy=POLICY, episodes=40, raw_only=True))
        staging.rename(output)
        return report
    except Exception as error:
        write_json(staging/'status.json', dict(state='failed', error=str(error)))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', required=True, type=Path)
    p.add_argument('--failure-selection', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--storage', choices=['hardlink', 'copy'], default='hardlink')
    a = p.parse_args()
    print(json.dumps(prepare(a.results, a.failure_selection, a.output, a.seed, a.storage), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
