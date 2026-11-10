"""A60 unchanged plus independently indexed, reviewed subtask observations."""
import hashlib
import json
from pathlib import Path

from direct_data import DirectEpisodes
from .dataset import SubtaskDataset


def smoke_indices(samples,baseline_samples,expert_samples):
    """Keep explicit source boundaries before filling a bounded 15-sample probe."""
    required=[0,expert_samples-1,expert_samples,baseline_samples-1,baseline_samples,samples-1]
    chosen=list(dict.fromkeys(required))
    for i in range(14):
        value=i*(samples-1)//13
        if value not in chosen:chosen.append(value)
        if len(chosen)==15:break
    if len(chosen)!=15 or any(i<0 or i>=samples for i in chosen):
        raise ValueError('Invalid mixed-source smoke selection')
    return sorted(chosen)


class MixedEpisodes:
    def __init__(self,sources,horizon=10):
        baseline=[s for s in sources if s.get('kind')!='reviewed_subtasks']
        additions=[s for s in sources if s.get('kind')=='reviewed_subtasks']
        if len(additions)!=1 or len(baseline)!=2:
            raise ValueError('Expected expert40, autonomous20, and one reviewed subtask source')
        self.base=DirectEpisodes(baseline,horizon=horizon)
        source=additions[0]
        self.root=Path(source['path']).expanduser().resolve()
        self.clips=SubtaskDataset(self.root)
        report=json.loads((self.root/'build_state.json').read_text())
        if (len(self.clips.segments)!=int(source['episodes']) or len(self.clips.segments)!=22
                or len(self.clips)!=1194 or report['unique_action_rows']!=1311
                or report['review_manifest_sha256']!=source['review_manifest_sha256']
                or report['tail_policy']!='success_endpoint_repeat' or report['horizon']!=horizon):
            raise ValueError('Wrong subtask dataset: expected the ORIGINAL reviewed22 with success-only repeats')
        manifest_bytes=(self.root/'meta/review_manifest.json').read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest()!=source['review_manifest_sha256']:
            raise ValueError('Original22 reviewed manifest checksum mismatch')
        original={r['segment_id']:r for r in json.loads(manifest_bytes)}
        if original.keys()!=self.clips.segments.keys():
            raise ValueError('Selected clips differ from original22 review')
        for sid,meta in self.clips.segments.items():
            ref=original[sid]
            if any(meta[k]!=ref[k] for k in ('task_instruction','action_rows_half_open','parent_parquet','candidate_id','endpoint_has_recorded_task_success')):
                raise ValueError('Subtask provenance/instruction differs from reviewed selection')
            path=self.root/meta['parquet']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=report['file_sha256'][meta['parquet']]:
                raise ValueError('Subtask Parquet checksum mismatch')
            starts=[w['start'] for w in self.clips.windows if w['segment_id']==sid]
            expected=meta['action_count'] if meta['padding_policy']=='repeat_last' else max(0,meta['action_count']-horizon+1)
            if starts!=list(range(expected)):
                raise ValueError('Subtask index is not exact dense source-aware sampling')
        if {s['name']:int(s['episodes']) for s in baseline}!={'author40':40,'autonomous20':20}:
            raise ValueError('Expected original expert40 and autonomous-success20')
        self.episodes=list(self.base.episodes)
        self.tasks=dict(self.base.tasks)
        self.snapshots={}
        for meta in self.clips.segments.values():
            task=meta['task_instruction']
            if task not in self.tasks:self.tasks[task]=len(self.tasks)
            self.episodes.append(dict(source=source['name'],original_episode_id=meta['dataset_episode_index'],
                path=str(self.root/meta['parquet']),length=meta['action_count'],task=task,
                task_index=self.tasks[task],segment_id=meta['segment_id'],
                training_windows=meta['trainable_window_count'],padding_policy=meta['padding_policy']))
        for path in [self.root/r['parquet'] for r in self.clips.segments.values()]+[
                self.root/p for p in ['windows.jsonl','build_state.json','meta/provenance.jsonl','meta/review_manifest.json']]:
            st=path.stat();self.snapshots[str(path)]=(st.st_size,st.st_mtime_ns)
        self.identity={p:hashlib.sha256((self.root/p).read_bytes()).hexdigest() for p in
                       ['windows.jsonl','build_state.json','meta/provenance.jsonl','meta/review_manifest.json']}
        self.name=source['name']

    def __len__(self):
        return len(self.base)+len(self.clips)

    def __getitem__(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        if index<len(self.base):return self.base[index]
        item=self.clips[index-len(self.base)]
        return dict(item,task=item['prompt'],task_index=self.tasks[item['prompt']])

    def summary(self):
        summary=self.base.summary()
        counts=dict(summary['source_episodes']);counts[self.name]=len(self.clips.segments)
        frames=dict(summary['source_frames']);frames[self.name]=1311
        return dict(summary,episodes=len(self.episodes),frames=len(self),tasks=len(self.tasks),
            source_episodes=counts,source_frames=frames,episodes_index=summary['episodes_index']+self.episodes[len(self.base.episodes):],
            training_windows=len(self),unique_action_rows=sum(frames.values()),
            subtask_windows=len(self.clips),subtask_index_sha256=self.identity,
            dataset_policy='a60_plus_reviewed22_parent_instruction_v1')

    def assert_unchanged(self):
        self.base.assert_unchanged()
        for file,expected in self.snapshots.items():
            st=Path(file).stat()
            if (st.st_size,st.st_mtime_ns)!=expected:raise ValueError('Source modified: '+file)


def audit_recipe(cfg,a60):
    """Compare every model/optimizer/runtime field; permit data and logger identity only."""
    from omegaconf import OmegaConf
    left=OmegaConf.to_container(cfg,resolve=True)
    right=OmegaConf.to_container(a60,resolve=True)
    for value in (left,right):
        value['data'].pop('sources')
        value['runner']['logger'].pop('experiment_name')
    if left!=right:
        raise ValueError('New experiment recipe differs from A60 outside data/name')
    return dict(state='passed',model_optimizer_scheduler_runtime_exactly_equal_to_a60=True,
                data_scope_is_only_recipe_difference=True,epochs=int(cfg.runner.max_epochs),
                default_global_batch=int(cfg.actor.global_batch_size),flow_steps=int(cfg.actor.model.num_steps))
