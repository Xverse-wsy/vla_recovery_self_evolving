"""Episode -> event -> negative start -> reference, independent of Stage1."""
from collections import defaultdict
import math
import random
from .schema import stable_id


class Stage2Sampler:
    def __init__(self, records, pairs, fm_ids, config):
        self.config = config
        config.validate()
        self.rng = random.Random(config.seed)
        by_id = {r.sample_id:r for r in records}
        self.non_code = sorted(s for s in fm_ids if by_id[s].controller in ('VLA','expert'))
        self.code = sorted(s for s in fm_ids if by_id[s].controller=='code')
        if not self.non_code:
            raise ValueError('Need ordinary VLA/expert FM positives')
        self.by_episode = defaultdict(lambda:defaultdict(list))
        for p in pairs:
            neg = by_id[p['negative_id']]
            if neg.controller!='VLA' or neg.training_quality!='negative':
                raise ValueError('Only VLA failures enter Stage2 pairing')
            self.by_episode[neg.episode_id][neg.event_id].append(p)
        self.visits = {e:0 for e in sorted(self.by_episode)}
        self.updates = 0
        self.inventory = stable_id(sorted(fm_ids), pairs)
        batch = min(config.pair_batch,len(self.visits))
        self.nominal_updates = math.ceil(len(self.visits)*config.max_episode_reuses/batch) if batch else config.fm_only_updates

    @property
    def done(self):
        if self.visits and self.config.pair_batch:
            return all(v==self.config.max_episode_reuses for v in self.visits.values())
        return self.updates>=self.config.fm_only_updates

    def draw(self):
        if self.done:
            raise StopIteration
        eligible = [e for e in sorted(self.visits) if self.visits[e]<self.config.max_episode_reuses]
        self.rng.shuffle(eligible)
        episodes = sorted(eligible,key=lambda e:self.visits[e])[:self.config.pair_batch]
        chosen,refs = [],[]
        for e in episodes:
            event = self.rng.choice(sorted(self.by_episode[e]))
            pair = self.rng.choice(self.by_episode[e][event])
            ref = self.rng.choice(pair['references'])
            chosen.append(dict(pair_id=pair['pair_id'],negative_id=pair['negative_id'],
                positive_id=ref['sample_id'],mask=ref['mask'],episode_id=e,event_id=event))
            refs.append(ref['sample_id'])
            self.visits[e]+=1
        ordinary = self.config.fm_batch-len(refs)
        n_code = round(ordinary*(1-self.config.vla_fraction)) if self.code else 0
        fm = [self.rng.choice(self.non_code) for _ in range(ordinary-n_code)]
        fm += [self.rng.choice(self.code) for _ in range(n_code)] + refs
        self.rng.shuffle(fm)
        self.updates+=1
        return dict(fm_ids=fm,pairs=chosen,episode_visits=dict(self.visits),update=self.updates)

    def state_dict(self):
        return dict(policy='stage2_episode_balanced_v1',inventory=self.inventory,visits=dict(self.visits),
                    updates=self.updates,rng=self.rng.getstate())

    def load_state_dict(self,state):
        if (state.get('policy')!='stage2_episode_balanced_v1' or state.get('inventory')!=self.inventory or
                state['visits'].keys()!=self.visits.keys()):
            raise ValueError('Stage2 sampler inventory changed')
        if any(type(v) is not int or not 0<=v<=self.config.max_episode_reuses for v in state['visits'].values()):
            raise ValueError('Invalid Stage2 episode reuse counters')
        batch = min(self.config.pair_batch,len(self.visits))
        expected = math.ceil(sum(state['visits'].values())/batch) if batch else state['updates']
        if state['updates']!=expected:
            raise ValueError('Stage2 update and reuse counters differ')
        self.visits,self.updates = dict(state['visits']),state['updates']
        self.rng.setstate(state['rng'])
