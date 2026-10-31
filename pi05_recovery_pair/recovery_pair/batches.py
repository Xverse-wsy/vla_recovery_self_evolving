"""Full positive epochs, plus the legacy episode-balanced pair sampler."""
from collections import defaultdict
import math
import random
from .schema import stable_id


class PositiveEpochSampler:
    """One shuffled visit per positive window per epoch; keep the partial batch.

    A global plan is identical on every rank. The engine shards its actual size,
    including tails smaller than world_size; empty ranks contribute zero grads.
    """
    policy = 'positive_epoch_permutation_v1'

    def __init__(self, records, pairs, config, split='train'):
        config.validate()
        if config.fm_epochs is None or pairs or config.pair_batch or config.eta:
            raise ValueError('Epoch sampler requires positive-only FM and explicit fm_epochs')
        self.config = config
        if any(r.split != 'train' for r in records) or split != 'train':
            raise ValueError('Full positive epochs require an all-train pool')
        selected = [r for r in records if r.split == split]
        if not selected or any(r.training_quality != 'positive' for r in selected):
            raise ValueError('Epoch sampler requires a nonempty positive-only pool')
        self.ids = sorted(r.sample_id for r in selected)
        if len(set(self.ids)) != len(self.ids):
            raise ValueError('Duplicate epoch sample IDs')
        self.inventory = stable_id(self.ids)
        self.visits = {}
        self.updates = self.epochs_completed = self.position = self.samples_seen = 0
        self.batches_per_epoch = math.ceil(len(self.ids)/config.fm_batch)
        self.nominal_updates = self.batches_per_epoch*config.fm_epochs
        self._set_order()

    def _set_order(self):
        self.order = self.ids.copy() if not self.done else []
        random.Random(self.config.seed+self.epochs_completed).shuffle(self.order)

    @property
    def done(self):
        return self.epochs_completed == self.config.fm_epochs

    def draw(self):
        if self.done:
            raise StopIteration
        epoch = self.epochs_completed+1
        batch_in_epoch = self.position//self.config.fm_batch+1
        fm = self.order[self.position:self.position+self.config.fm_batch]
        self.position += len(fm)
        self.samples_seen += len(fm)
        self.updates += 1
        complete = self.position == len(self.ids)
        if complete:
            self.epochs_completed += 1
            self.position = 0
            self._set_order()
        return dict(fm_ids=fm,pairs=[],episode_visits={},update=self.updates,
            epoch=epoch,batch_in_epoch=batch_in_epoch,epoch_complete=complete,
            epochs_completed=self.epochs_completed,position_in_epoch=self.position,
            samples_seen=self.samples_seen)

    def state_dict(self):
        return dict(policy=self.policy,inventory=self.inventory,seed=self.config.seed,
            fm_batch=self.config.fm_batch,fm_epochs=self.config.fm_epochs,
            updates=self.updates,epochs_completed=self.epochs_completed,
            position=self.position,samples_seen=self.samples_seen)

    def load_state_dict(self,state):
        expected=self.state_dict()
        for k in ('policy','inventory','seed','fm_batch','fm_epochs'):
            if state.get(k)!=expected[k]:
                raise ValueError('Epoch checkpoint sampling identity differs: '+k)
        for k in ('updates','epochs_completed','position','samples_seen'):
            if type(state.get(k)) is not int or state[k]<0:
                raise ValueError('Invalid epoch checkpoint counters')
        epoch,pos=state['epochs_completed'],state['position']
        if (epoch>self.config.fm_epochs or not 0<=pos<len(self.ids) or
                (epoch==self.config.fm_epochs and pos) or pos%self.config.fm_batch or
                state['updates']!=epoch*self.batches_per_epoch+pos//self.config.fm_batch or
                state['samples_seen']!=epoch*len(self.ids)+pos):
            raise ValueError('Inconsistent epoch checkpoint cursor')
        self.updates,self.epochs_completed,self.position,self.samples_seen=(
            state[k] for k in ('updates','epochs_completed','position','samples_seen'))
        self._set_order()

    def budget_report(self):
        return dict(sampling_policy=self.policy,target_epochs=self.config.fm_epochs,
            windows_per_epoch=len(self.ids),batches_per_epoch=self.batches_per_epoch,
            expected_updates=self.nominal_updates,expected_samples=len(self.ids)*self.config.fm_epochs,
            tail_batch_size=(len(self.ids)-1)%self.config.fm_batch+1,
            source_quota_applied=False,drop_last=False,pad_last=False)


def training_sampler(records,pairs,config,*,mode,split='train'):
    if config.fm_epochs is not None:
        if mode!='fm_only':
            raise ValueError('Full positive epochs require --mode fm_only')
        return PositiveEpochSampler(records,pairs,config,split)
    return EpisodeEventSampler(records,pairs,config,split)


class EpisodeEventSampler:
    def __init__(self, records, pairs, config, split='train'):
        self.config = config
        self.rng = random.Random(config.seed)
        self.records = {r.sample_id: r for r in records}
        self.positive = {owner: [r.sample_id for r in records if r.split == split and
                                r.training_quality == 'positive' and r.controller == owner]
                         for owner in ['VLA', 'code', 'expert']}
        # Expert provenance stays separate; preserve the existing code quota and
        # draw remaining slots uniformly from the combined expert/VLA window pool.
        self.non_code = self.positive['VLA'] + self.positive['expert']
        self.by_episode = defaultdict(lambda: defaultdict(list))
        for pair in pairs:
            if pair['split'] == split:
                negative = self.records.get(pair['negative_id'])
                if negative is None or negative.controller != 'VLA' or negative.training_quality != 'negative':
                    raise ValueError('Pair sampler accepts VLA failure queries only')
                self.by_episode[pair['negative_episode_id']][pair['event_id']].append(pair)
        self.visits = {episode: 0 for episode in self.by_episode}
        self.updates = 0
        batch = min(config.pair_batch, len(self.visits))
        self.nominal_updates = (math.ceil(len(self.visits) * config.max_episode_reuses / batch)
                                if batch else config.fm_only_updates)
        if not self.non_code:
            raise ValueError('Need VLA or expert positive windows')
        if self.positive['expert'] and (pairs or config.pair_batch or config.eta):
            raise ValueError('Expert FM pool is supported only with the positive-only configuration')

    @property
    def done(self):
        if self.visits and self.config.pair_batch:
            return all(n >= self.config.max_episode_reuses for n in self.visits.values())
        return self.updates >= self.config.fm_only_updates

    def draw(self):
        if self.done:
            raise StopIteration
        eligible = sorted(e for e, n in self.visits.items() if n < self.config.max_episode_reuses)
        self.rng.shuffle(eligible)
        # Random ties, balanced reuse counts; no episode is visited 51 times.
        selected = sorted(eligible, key=lambda e: self.visits[e])[:self.config.pair_batch]
        references, picked = [], []
        for episode in selected:
            event = self.rng.choice(sorted(self.by_episode[episode]))
            pair = self.rng.choice(self.by_episode[episode][event])
            positive = self.rng.choice(pair['references'])
            picked.append(dict(pair_id=pair['pair_id'], negative_id=pair['negative_id'],
                               positive_id=positive['sample_id'], mask=positive['mask'],
                               episode_id=episode, event_id=event))
            references.append(positive['sample_id'])
            self.visits[episode] += 1
        ordinary = self.config.fm_batch - len(references)
        n_code = round(ordinary * (1 - self.config.vla_fraction)) if self.positive['code'] else 0
        fm = [self.rng.choice(self.non_code) for _ in range(ordinary - n_code)]
        fm += [self.rng.choice(self.positive['code']) for _ in range(n_code)]
        fm += references
        self.rng.shuffle(fm)
        self.updates += 1
        return dict(fm_ids=fm, pairs=picked, episode_visits=dict(self.visits), update=self.updates)

    def state_dict(self):
        return dict(visits=self.visits, updates=self.updates, rng=self.rng.getstate())

    def load_state_dict(self, state):
        if state['visits'].keys() != self.visits.keys():
            raise ValueError('Paired episode inventory changed on resume')
        if any(n > self.config.max_episode_reuses for n in state['visits'].values()):
            raise ValueError('Checkpoint exceeds episode reuse budget')
        self.visits = dict(state['visits'])
        self.updates = state['updates']
        self.rng.setstate(state['rng'])
