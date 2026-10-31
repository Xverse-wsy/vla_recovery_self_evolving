"""One negative episode gets one vote per global update, irrespective of windows."""
from collections import defaultdict
import math
import random


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
