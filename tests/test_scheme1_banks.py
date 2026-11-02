import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parents[1] / 'recovery_conversion'
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('scheme1_build', HERE / 'build_scheme1.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_trim_removes_actual_ten_actions_and_observation_rows_before_windowing():
    span = dict(start=0, end=100, category='vla_progress_candidate', owner='pi05', turn=None)
    result = m.trim_progress_span(span)
    assert span['end'] == 100
    assert result['end'] == 90
    assert result['discarded_parent_range'] == [90, 100]
    # H10 windows can end at action89; they never read discarded actions90..99.
    starts = list(range(result['start'], result['end'] - 10 + 1))
    assert starts[-1] == 80 and starts[-1] + 10 == 90


def test_warning_and_code_are_not_trimmed_and_short_progress_becomes_empty():
    for category in ['vla_wcm_warning', 'capx_program']:
        span = dict(start=20, end=50, category=category)
        assert m.trim_progress_span(span) == span
    result = m.trim_progress_span(dict(start=0, end=7, category='vla_progress_candidate'))
    assert result['end'] == 0 and result['discarded_tail_actions_and_start_frames'] == 7


def test_pre_action_observation_alignment_uses_initial_then_previous_post_state(tmp_path):
    trajectory = tmp_path / 'trajectory'
    trajectory.mkdir()
    np.savez_compressed(trajectory / 'initial.npz', **{'policy/state': np.zeros(8), 'raw/test': np.array(0)})
    shard = trajectory / 'shard.npz'
    np.savez_compressed(shard, **{'policy/state': np.array([np.ones(8), np.ones(8) * 2]),
        'raw/test': np.array([1, 2]), 'transition/action': np.array([np.ones(8) * 10, np.ones(8) * 20])})
    (trajectory / 'manifest.json').write_text(json.dumps(dict(steps=2, shards=[dict(file='shard.npz', sha256=m.core.sha(shard))])))
    records = list(m.trajectory_rows(trajectory))
    assert len(records) == 2
    assert records[0][0]['raw/test'] == 0 and records[0][1][0] == 10
    assert records[1][0]['raw/test'] == 1 and records[1][1][0] == 20


def test_strict_config_is_separate_all_train_and_does_not_modify_original():
    original = HERE.parent / 'pi05_recovery_pair/configs/libero_pi05_positive_fm.json'
    before = original.read_bytes()
    config = m.algorithm(HERE.parent / 'pi05_recovery_pair')
    assert config['coverage'] == dict(position_m=.01, rotation_deg=5., proprio_rms=.1, image_cosine=.002, max_candidates=50)
    assert config['cap_success_admission'] == 'segment_suffix'
    assert config['split_fractions'] == [1., 0., 0.]
    assert config['pair_batch'] == 0 and config['fm_epochs'] == 50
    assert config['training_recipe'] == 'stage1_a60_sft_v1'
    assert config['learning_rate'] == 5e-6 and config['flow_steps'] == 5
    assert config['fm_batch'] == 128 and config['fm_micro_batch'] == 2
    assert original.read_bytes() == before


def test_a60_alignment_preserves_existing_strict_data_gates():
    sys.path.insert(0, str(HERE.parent / 'pi05_recovery_pair'))
    from align_stage1_a60 import aligned_config
    previous = m.algorithm(HERE.parent / 'pi05_recovery_pair')
    previous.update(learning_rate=1e-5, flow_steps=3, fm_batch=64)
    previous.pop('training_recipe')
    result = json.loads(json.dumps(aligned_config(HERE.parent / 'pi05_recovery_pair', previous).to_dict()))
    for key in ['coverage','pairing','state_stride','cap_success_admission','split_fractions']:
        assert result[key] == previous[key]
    assert result['training_recipe'] == 'stage1_a60_sft_v1'
    assert result['learning_rate'] == 5e-6 and result['flow_steps'] == 5
    assert result['fm_batch'] == 128
