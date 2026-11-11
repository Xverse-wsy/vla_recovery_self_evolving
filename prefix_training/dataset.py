"""The original A60 reader plus fixed, already-cropped, no-padding prefixes."""
import hashlib
import json
from pathlib import Path

from direct_data import DirectEpisodes
from subtask_training.dataset import SubtaskDataset
from .spec import ACTION_ROWS, SEGMENTS, TOTAL_ACTION_ROWS, TOTAL_WINDOWS, WINDOWS, selection, selection_sha


class PrefixBank(SubtaskDataset):
    def __init__(self, root):
        super().__init__(root)
        report = json.loads((self.root / 'build_state.json').read_text())
        if (report.get('policy') != 'vla_prefix10_tail15_v1'
                or report.get('selection_sha256') != selection_sha()
                or (report['segments'], report['unique_action_rows'], len(self)) != (SEGMENTS, ACTION_ROWS, WINDOWS)
                or report['horizon'] != 10 or report['total_original_trim'] != 15
                or report['additional_source_trim'] != 5 or report['padded_action_positions'] != 0):
            raise ValueError('Wrong prefix dataset: expected fixed10, physical total trim15, 845 rows/755 windows')
        if hashlib.sha256((self.root / 'meta/selection.json').read_bytes()).hexdigest() != selection_sha():
            raise ValueError('Prefix selection checksum mismatch')
        expected = {r['segment_id']: r for r in selection()['segments']}
        if expected.keys() != self.segments.keys():
            raise ValueError('Wrong prefix task/episode selection')
        for sid, ref in expected.items():
            meta = self.segments[sid]
            if any(meta.get(k) != v for k, v in ref.items()):
                raise ValueError('Prefix provenance differs from fixed selection')
            if (meta['padding_policy'] != 'full_real_only' or meta['endpoint_has_recorded_task_success']
                    or meta['controller'] != 'VLA'):
                raise ValueError('VLA prefixes must never repeat terminal actions')
            table = self.numeric(meta['parquet'])
            if table.num_rows != ref['action_count']:
                raise ValueError('Prefix must be physically cropped exactly once before loading')
            path = self.root / meta['parquet']
            if hashlib.sha256(path.read_bytes()).hexdigest() != report['file_sha256'][meta['parquet']]:
                raise ValueError('Prefix Parquet checksum mismatch')
            ws = [w for w in self.windows if w['segment_id'] == sid]
            if [w['start'] for w in ws] != list(range(ref['action_count'] - 9)):
                raise ValueError('Prefix starts must be exactly 0..n-10, inclusive')
            for w in ws:
                start = ref['retained_parent_action_rows'][0] + w['start']
                if (w['horizon'] != 10 or w['real_steps'] != 10 or w['repeated_steps'] != 0
                        or w['tail_policy'] != 'full_real_only'
                        or w['parent_action_rows_half_open'] != [start, start + 10]):
                    raise ValueError('Invalid or padded prefix window')


class PrefixEpisodes:
    def __init__(self, sources, horizon=10):
        if horizon != 10:
            raise ValueError('Fixed prefix bank requires H10')
        additions = [s for s in sources if s.get('kind') == 'vla_prefix10_tail15']
        baseline = [s for s in sources if s.get('kind') != 'vla_prefix10_tail15']
        if (len(additions) != 1 or len(baseline) != 2
                or {s['name']: int(s['episodes']) for s in baseline} != {'author40': 40, 'autonomous20': 20}
                or int(additions[0]['episodes']) != 10):
            raise ValueError('Expected original expert40 + autonomous20 + prefix10')
        self.base = DirectEpisodes(baseline, horizon=horizon)
        if len(self.base) != 9864 or len(self.base.episodes) != 60 or len(self.base.tasks) != 40:
            raise ValueError('Wrong original A60 data: expected 9864 windows and all 40 tasks')
        self.root = Path(additions[0]['path']).expanduser().resolve()
        self.clips = PrefixBank(self.root)
        auto_tasks = {r['task'] for r in self.base.episodes if r['source'] == 'autonomous20'}
        if any(m['task_instruction'] in auto_tasks for m in self.clips.segments.values()):
            raise ValueError('Selected prefix tasks must not overlap autonomous20 tasks')
        self.tasks = dict(self.base.tasks)
        self.episodes = list(self.base.episodes)
        self.name = additions[0]['name']
        for m in self.clips.segments.values():
            if m['task_instruction'] not in self.tasks:
                raise ValueError('Prefix task absent from original expert40')
            self.episodes.append(dict(source=self.name, original_episode_id=m['dataset_episode_index'],
                path=str(self.root / m['parquet']), length=m['action_count'], task=m['task_instruction'],
                task_index=self.tasks[m['task_instruction']], segment_id=m['segment_id'],
                training_windows=m['trainable_window_count'], padding_policy='full_real_only'))
        files = [self.root / m['parquet'] for m in self.clips.segments.values()] + [
            self.root / p for p in ('windows.jsonl', 'build_state.json', 'meta/provenance.jsonl', 'meta/selection.json')]
        self.snapshots = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
        self.identity = {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in files if p.suffix != '.parquet'}

    def __len__(self):
        return len(self.base) + len(self.clips)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < len(self.base):
            return self.base[index]
        item = self.clips[index - len(self.base)]
        return dict(item, task=item['prompt'], task_index=self.tasks[item['prompt']])

    def summary(self):
        base = self.base.summary()
        counts = dict(base['source_episodes']); counts[self.name] = 10
        frames = dict(base['source_frames']); frames[self.name] = 845
        return dict(base, episodes=70, frames=len(self), tasks=len(self.tasks),
            source_episodes=counts, source_frames=frames, episodes_index=[
                {k: v for k, v in m.items() if k not in ('state', 'actions', 'row_group_ends')}
                for m in self.episodes], training_windows=TOTAL_WINDOWS,
            unique_action_rows=TOTAL_ACTION_ROWS, prefix_windows=755,
            prefix_index_sha256=self.identity, dataset_policy='a60_plus_prefix10_tail15_v1')

    def assert_unchanged(self):
        self.base.assert_unchanged()
        for path, expected in self.snapshots.items():
            p = Path(path)
            if (p.stat().st_size, p.stat().st_mtime_ns) != expected:
                raise ValueError('Prefix source changed during training: ' + path)
