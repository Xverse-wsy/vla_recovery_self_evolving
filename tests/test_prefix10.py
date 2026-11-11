import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from prefix_training.build import materialize
from prefix_training.dataset import PrefixBank
from prefix_training.spec import records, selection
from settings import load_config
from subtask_training.fetch import sha256
from subtask_training.training_data import audit_recipe, smoke_indices


@pytest.fixture
def sources(tmp_path):
    cache = tmp_path / 'sources'
    spec = selection()
    groups = {}
    buf = io.BytesIO()
    Image.new('RGB', (256, 256), (20, 60, 100)).save(buf, format='PNG')
    packed = {'bytes': buf.getvalue(), 'path': None}
    image_type = pa.struct([('bytes', pa.binary()), ('path', pa.string())])
    hashes = {}
    for i, r in enumerate(spec['segments']):
        n = r['action_count'] + 5
        actions = np.arange(n * 7, dtype=np.float32).reshape(n, 7) + i * 10000
        table = pa.table(dict(
            image=pa.array([packed] * n, type=image_type), wrist_image=pa.array([packed] * n, type=image_type),
            state=np.zeros((n, 8), np.float32).tolist(), actions=actions.tolist(),
            timestamp=[j / 10 for j in range(n)], frame_index=list(range(n)),
            episode_index=[r['source_segment_episode_index']] * n, index=list(range(n)),
            task_index=[r['canonical_task_index']] * n))
        path = cache / r['source_parquet']
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        hashes[r['source_parquet']] = sha256(path)
        group = r['source_parquet'].split('/data/')[0]
        groups.setdefault(group, []).append(dict(
            episode_index=r['source_segment_episode_index'], category='vla_progress_candidate', owner='pi05',
            instruction=r['task_instruction'], suite=r['suite'], task_id=r['task_id'], trial_id=r['trial_id'],
            parent_episode_index=r['source_parent_episode_index'],
            start=r['source_parent_rows_after_trim10'][0], end=r['source_parent_rows_after_trim10'][1],
            untrimmed_parent_range=r['untrimmed_parent_range'],
            discarded_tail_actions_and_start_frames=10, task_completed_in_segment=False))
    for group, prov in groups.items():
        tasks = [dict(task_index=r['canonical_task_index'], task=r['task_instruction'])
                 for r in spec['segments'] if r['source_parquet'].startswith(group + '/')]
        for name, value in [('provenance.jsonl', prov), ('tasks.jsonl', tasks), ('info.json', {'fps': 10})]:
            path = cache / group / 'meta' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value) if name.endswith('.json') else ''.join(json.dumps(v) + '\n' for v in value))
            hashes[path.relative_to(cache).as_posix()] = sha256(path)
    return cache, hashes


@pytest.fixture
def bank(sources, tmp_path):
    cache, hashes = sources
    return materialize(cache, tmp_path / 'built', hashes), cache


def test_builder_removes_only_five_more_rows_and_preserves_original_data(bank):
    root, cache = bank
    ds = PrefixBank(root)
    assert len(ds) == 755
    assert sum(m['action_count'] for m in ds.segments.values()) == 845
    for m in ds.segments.values():
        source = pq.read_table(cache / m['source_parquet'])
        got = pq.read_table(root / m['parquet'])
        assert source.num_rows - got.num_rows == 5
        for col in ('image', 'wrist_image', 'state', 'actions', 'timestamp'):
            assert got[col].equals(source[col].slice(0, got.num_rows))


def test_last_window_is_real_and_no_second_trim_occurs(bank):
    root, _ = bank
    ds = PrefixBank(root)
    for sid, m in ds.segments.items():
        indices = [i for i, w in enumerate(ds.windows) if w['segment_id'] == sid]
        assert len(indices) == m['action_count'] - 9
        last = ds.windows[indices[-1]]
        assert last['start'] == m['action_count'] - 10
        expected = np.asarray(ds.numeric(m['parquet'])['actions'].slice(last['start'], 10).to_pylist(), np.float32)
        np.testing.assert_array_equal(ds[indices[-1]]['actions'], expected)
        assert ds[indices[-1]]['prompt'] == m['task_instruction']


def test_padding_or_incomplete_prefix_window_is_rejected(bank):
    root, _ = bank
    ds = PrefixBank(root)
    w = ds.windows[-1]
    w['start'] += 1
    with pytest.raises(ValueError, match='Invalid window'):
        ds[len(ds) - 1]
    w.update(tail_policy='repeat_last')
    with pytest.raises(ValueError):
        ds[len(ds) - 1]


def test_attempt_to_materialize_from_already_trim15_source_is_rejected(sources, tmp_path):
    cache, hashes = sources
    r = selection()['segments'][0]
    path = cache / r['source_parquet']
    cropped = pq.read_table(path).slice(0, r['action_count'])
    pq.write_table(cropped, path)
    hashes[r['source_parquet']] = sha256(path)
    with pytest.raises(ValueError, match='double trim'):
        materialize(cache, tmp_path / 'double_trim', hashes)


def test_wrong_old_trim10_dataset_is_rejected(bank):
    root, _ = bank
    report = json.loads((root / 'build_state.json').read_text())
    report['unique_action_rows'] = 895
    (root / 'build_state.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Wrong prefix dataset'):
        PrefixBank(root)


def test_a60_recipe_and_smoke_cover_every_source(monkeypatch):
    monkeypatch.setenv('PI05_ROOT', '/tmp/prefix10_recipe')
    cfg, base = load_config('a60_prefix10'), load_config('a60')
    assert audit_recipe(cfg, base)['data_scope_is_only_recipe_difference']
    assert cfg.runner.max_epochs == 50 and cfg.actor.global_batch_size == 100
    assert cfg.actor.model.openpi.action_horizon == 10
    indices = smoke_indices(10619, 9864, 6632)
    assert len(indices) == 15 and {0, 6631, 6632, 9863, 9864, 10618} <= set(indices)
    cfg.actor.model.num_steps = 3
    with pytest.raises(ValueError, match='differs from A60'):
        audit_recipe(cfg, base)
