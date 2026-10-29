"""Read selected existing Parquets in place; no merged dataset or HF downloads."""
from bisect import bisect_right
from collections import Counter, OrderedDict
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


class DirectEpisodes:
    def __init__(self, sources, horizon=10, cache_row_groups=128):
        self.horizon = int(horizon)
        self.cache_limit = cache_row_groups
        self.cache = OrderedDict()
        self.episodes = []
        self.ends = []
        self.tasks = {}
        self.snapshots = {}
        self.source_counts = Counter()
        self.source_frames = Counter()
        for source in sources:
            root = Path(source['path']).expanduser().resolve()
            info = json.loads((root / 'meta/info.json').read_text())
            tasks = {int(x['task_index']): x['task'] for x in
                     map(json.loads, (root / 'meta/tasks.jsonl').read_text().splitlines())}
            if source.get('episodes_file'):
                ids = json.loads((root / source['episodes_file']).read_text())
            else:
                ids = [int(x['episode_index']) for x in
                       map(json.loads, (root / 'meta/episodes.jsonl').read_text().splitlines())]
            if len(ids) != source['episodes'] or len(ids) != len(set(ids)):
                raise ValueError(f'Wrong episode selection for {source["name"]}: {len(ids)}')
            for eid in ids:
                path = root / info['data_path'].format(episode_chunk=eid // info['chunks_size'], episode_index=eid)
                # Read numerical columns once. Images remain in their original row groups.
                table = pq.read_table(path, columns=['state', 'actions', 'task_index', 'episode_index', 'frame_index'])
                n = len(table)
                state = np.array(table['state'].to_pylist(), dtype=np.float32)
                actions = np.array(table['actions'].to_pylist(), dtype=np.float32)
                tids = np.asarray(table['task_index'])
                if state.shape != (n, 8) or actions.shape != (n, 7) or n < 1:
                    raise ValueError(f'Invalid shapes: {path}')
                if not np.isfinite(state).all() or not np.isfinite(actions).all():
                    raise ValueError(f'Non-finite state/actions: {path}')
                if not np.all(np.asarray(table['episode_index']) == eid):
                    raise ValueError(f'Episode identity mismatch: {path}')
                np.testing.assert_array_equal(np.asarray(table['frame_index']), np.arange(n))
                if len(set(tids.tolist())) != 1:
                    raise ValueError(f'Task changes inside episode: {path}')
                task = tasks[int(tids[0])]
                if task not in self.tasks:
                    self.tasks[task] = len(self.tasks)
                metadata = pq.read_metadata(path)
                groups = np.cumsum([metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)]).tolist()
                rec = dict(source=source['name'], original_episode_id=eid, path=str(path),
                           length=n, task=task, task_index=self.tasks[task],
                           state=state, actions=actions, row_group_ends=groups)
                self.episodes.append(rec)
                self.ends.append((self.ends[-1] if self.ends else 0) + n)
                st = path.stat()
                self.snapshots[str(path)] = (st.st_size, st.st_mtime_ns)
                self.source_counts[source['name']] += 1
                self.source_frames[source['name']] += n

    def __len__(self):
        return self.ends[-1]

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ep = bisect_right(self.ends, index)
        local = index - (self.ends[ep - 1] if ep else 0)
        return self.episodes[ep], local

    def _images(self, rec, local):
        group = bisect_right(rec['row_group_ends'], local)
        key = (rec['path'], group)
        if key not in self.cache:
            self.cache[key] = pq.ParquetFile(rec['path']).read_row_group(group, columns=['image', 'wrist_image'])
            while len(self.cache) > self.cache_limit:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        table = self.cache[key]
        row = local - (rec['row_group_ends'][group - 1] if group else 0)
        result = {}
        for key in ['image', 'wrist_image']:
            packed = table[key][row].as_py()
            if not packed.get('bytes'):
                raise ValueError('This reader requires embedded images; will not resolve external image paths')
            with Image.open(io.BytesIO(packed['bytes'])) as im:
                result[key] = np.array(im.convert('RGB'), dtype=np.uint8)
            if result[key].shape != (256, 256, 3):
                raise ValueError(f'Unexpected image shape: {rec["path"]}')
        return result

    def __getitem__(self, index):
        rec, local = self.locate(index)
        # Match LeRobot: terminal windows repeat the final action of THIS episode.
        positions = np.minimum(local + np.arange(self.horizon), rec['length'] - 1)
        return dict(self._images(rec, local), state=rec['state'][local].copy(),
                    actions=rec['actions'][positions].copy(), prompt=rec['task'],
                    task=rec['task'], task_index=rec['task_index'])

    def summary(self):
        return dict(episodes=len(self.episodes), frames=len(self), tasks=len(self.tasks),
                    source_episodes=dict(self.source_counts), source_frames=dict(self.source_frames),
                    action_horizon=self.horizon, image_rotation_applied=False,
                    dataset_files_copied=False,
                    episodes_index=[{k: v for k, v in r.items() if k not in ('state', 'actions', 'row_group_ends')}
                                    for r in self.episodes])

    def assert_unchanged(self):
        for file, expected in self.snapshots.items():
            st = Path(file).stat()
            assert (st.st_size, st.st_mtime_ns) == expected, f'Source modified: {file}'

    def __getstate__(self):
        state = self.__dict__.copy()
        state['cache'] = OrderedDict()
        return state


def build_transformed(cfg, sources=None):
    """Use the installed RLinf/OpenPI transforms and ORIGINAL normalization."""
    import dataclasses
    from omegaconf import OmegaConf
    from openpi.training.data_loader import transform_dataset
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    raw = DirectEpisodes(sources or OmegaConf.to_container(cfg.data.sources, resolve=True),
                         horizon=int(cfg.actor.model.openpi.action_horizon))
    config = get_openpi_config('pi05_libero', model_path=cfg.actor.model.model_path,
                              batch_size=cfg.actor.micro_batch_size,
                              repo_id=cfg.data.sources[0].path,
                              data_kwargs=OmegaConf.to_container(cfg.actor.model.openpi_data, resolve=True))
    data_config = config.data.create(config.assets_dirs, config.model)
    # No extra delta transform: the source actions are already controller commands.
    return raw, transform_dataset(raw, data_config), data_config
