"""Connected grouping BEFORE any reference index or pair is constructed."""
from collections import defaultdict
import random
import math


def split_windows(records, fractions=(0.8, 0.1, 0.1), seed=42, conservative_task=True):
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        parent[b] = a

    seen = {}
    uncertain_tasks = {r.task_id for r in records if r.uncertain_initialization}
    for i, r in enumerate(records):
        keys = [('initial', r.initial_group_id), ('episode', r.episode_id)]
        if conservative_task or r.task_id in uncertain_tasks:
            keys.append(('task', r.task_id))
        for key in keys:
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(i)
    ordered = sorted(groups.values(), key=lambda ids: min(records[i].sample_id for i in ids))
    random.Random(seed).shuffle(ordered)
    # Assign indivisible components; no window-level random split or match leakage.
    targets = [len(ordered) * f for f in fractions]
    choices = [j for j, f in enumerate(fractions) if f > 0]
    minimum = 1 if len(ordered) >= len(choices) else 0
    quotas = [max(minimum, math.floor(targets[j])) if j in choices else 0 for j in range(3)]
    while sum(quotas) < len(ordered):
        j = max(choices, key=lambda k: targets[k] - quotas[k])
        quotas[j] += 1
    while sum(quotas) > len(ordered):
        j = max((k for k in choices if quotas[k] > minimum), key=lambda k: quotas[k]-targets[k])
        quotas[j] -= 1
    offset = 0
    for j, quota in enumerate(quotas):
        for group in ordered[offset:offset+quota]:
            for i in group:
                records[i].split = ['train','validation','test'][j]
        offset += quota
    audit_split(records)
    return records


def audit_split(records):
    for attr in ['initial_group_id', 'episode_id']:
        seen = defaultdict(set)
        for r in records:
            seen[getattr(r, attr)].add(r.split)
        if any(len(v) != 1 for v in seen.values()):
            raise ValueError(f'{attr} leaks across splits')
    if any(r.split not in ('train', 'validation', 'test') for r in records):
        raise ValueError('All records must be split before indexing')
