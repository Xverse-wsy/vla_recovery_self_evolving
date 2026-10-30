"""CAP successes match a frozen VLA-success bank before positive-pool admission."""
from collections import defaultdict
from dataclasses import asdict
import math
import numpy as np

from .splits import audit_split
from .schema import stable_id

OBSERVATION_POLICY = 'policy_inputs_only_v1'


def instruction_key(record):
    """Public task text is the only task identity used for state retrieval."""
    return ' '.join(record.instruction.split())


def proprio_quaternion(rotvec):
    """Orientation from the SAME raw state[3:6] provided to the policy."""
    angle = float(np.linalg.norm(rotvec))
    scale = math.sin(angle / 2) / angle if angle > 1e-12 else .5
    return np.r_[rotvec * scale, math.cos(angle / 2)]


def rotation_degrees(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return math.degrees(2 * math.acos(float(np.clip(abs(a @ b), 0, 1))))


def pair_mask(positive, negative, config):
    p, n = positive.payload(), negative.payload()
    if positive.spec.identity != negative.spec.identity:
        raise ValueError('Action representation/units/frame/frequency/normalization/H mismatch')
    dims = np.asarray(negative.spec.valid_dimensions, bool)[None]
    valid = p['valid_mask'] & n['valid_mask'] & dims
    mask = valid & negative.pair_error_mask(n)
    # Remove only the shared leading prefix, not arbitrary later similar steps.
    for h in range(valid.shape[0]):
        compared = valid[h]
        if not compared.any() or np.max(np.abs(p['actions'][h, compared] - n['actions'][h, compared])) > config.shared_prefix_tolerance:
            break
        mask[h] = False
    if np.count_nonzero(mask.any(-1)) < config.min_pair_steps:
        return None, 'insufficient_trusted_overlap'
    if np.mean((p['actions'][mask] - n['actions'][mask]) ** 2) <= config.identical_action_tolerance ** 2:
        return None, 'nearly_identical_actions'
    return mask, None


def numeric_match(query, reference, thresholds):
    task = instruction_key(query)
    if query.split != reference.split or not task or task != instruction_key(reference):
        return None, 'split_or_instruction'
    if query.spec.identity != reference.spec.identity:
        return None, 'action_spec'
    # Input whitelist: public instruction and the policy's raw/normalized state.
    # Do not read legacy EEF sidecars, phase, object/contact/held state, progress,
    # simulator object poses or manually verified flags, even when present.
    raw_q, raw_r = np.asarray(query.proprio, float), np.asarray(reference.proprio, float)
    q, r = np.asarray(query.standardized_proprio), np.asarray(reference.standardized_proprio)
    if raw_q.shape != (8,) or raw_r.shape != (8,) or q.shape != (8,) or r.shape != (8,):
        return None, 'proprio_shape'
    if not all(np.isfinite(x).all() for x in (raw_q, raw_r, q, r)):
        return None, 'invalid_proprio'
    distance = dict(
        position_m=float(np.linalg.norm(raw_q[:3] - raw_r[:3])),
        rotation_deg=rotation_degrees(proprio_quaternion(raw_q[3:6]), proprio_quaternion(raw_r[3:6])),
        proprio_rms=float(np.sqrt(np.mean((q - r) ** 2))))
    for key, value in distance.items():
        bound = getattr(thresholds, key)
        if bound is None or value > bound:
            return None, 'threshold_' + key
    return distance, None


def image_distance(query, reference):
    qa, ra = query.visual_features, reference.visual_features
    if not qa or not ra or qa.get('encoder_sha256') != ra.get('encoder_sha256'):
        return None, 'missing_or_different_frozen_encoder'
    if qa.get('calibration_id') != ra.get('calibration_id'):
        return None, 'visual_preprocessing_mismatch'
    views = qa.get('views', {})
    if not all(v in views and v in ra.get('views', {}) for v in ('front', 'wrist')):
        return None, 'missing_views'
    distances = {}
    for view in ('front', 'wrist'):
        feature = views[view]
        a, b = np.asarray(feature, float), np.asarray(ra['views'][view], float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return None, 'invalid_visual_feature'
        distances[view] = float(1 - np.clip((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
    return distances, None


class FrozenMatcher:
    def __init__(self, records, config):
        audit_split(records)
        config.validate(matching_ready=True)
        self.config = config
        for record in records:
            record.validate()
        self.records = {r.sample_id: r for r in records}
        if len(self.records) != len(records):
            raise ValueError('Duplicate sample IDs')
        self.vla_states = tuple(r for r in records if r.controller == 'VLA')
        self.vla_success = tuple(r for r in self.vla_states if r.training_quality == 'positive')
        self.frozen_state_ids = tuple(sorted(r.sample_id for r in self.vla_success))
        self.coverage_bank_id = stable_id(self.frozen_state_ids)
        self.by_task = defaultdict(tuple)
        groups = defaultdict(list)
        for r in self.vla_success:
            groups[(r.split, instruction_key(r))].append(r)
        self.by_task = {k: tuple(v) for k, v in groups.items()}
        self.coverage = {}
        self.success_matches = []
        self.success_segments = []
        self.positive = list(self.vla_success)
        self.negative = []
        self.rejections = []
        # Stage 1: ONLY code-success queries against the ORIGINAL VLA-success bank.
        # Accepted code successes never become references for this admission step.
        direct = {}
        code_positive = [r for r in records if r.controller == 'code' and r.training_quality == 'positive']
        for r in code_positive:
            candidates = self.match(r, self.by_task.get((r.split, instruction_key(r)), ()), config.coverage)
            if not candidates:
                continue
            direct[r.sample_id] = candidates[0]
            self.success_matches.append(dict(sample_id=r.sample_id, episode_id=r.episode_id,
                split=r.split, code_start=r.to_dict(), vla_reference=candidates[0],
                vla_start=self.records[candidates[0]['sample_id']].to_dict(),
                candidate_count=len(candidates),
                label_basis=r.label_basis, bank_label=r.bank_label,
                frozen_bank_id=self.coverage_bank_id, observation_policy=OBSERVATION_POLICY))
        if config.cap_success_admission == 'segment_suffix':
            self._admit_success_suffixes(code_positive, direct)
        else:
            for r in code_positive:
                if r.sample_id in direct:
                    self.coverage[r.sample_id] = dict(reference=direct[r.sample_id],
                        frozen_bank_id=self.coverage_bank_id, admission_kind='direct_window_match')
                    self.positive.append(r)
        for r in code_positive:
            if r.sample_id not in self.coverage:
                self.rejections.append(dict(sample_id=r.sample_id,
                    reason='code_success_outside_matched_suffix' if config.cap_success_admission == 'segment_suffix'
                           else 'code_success_no_vla_success_match'))
        # Stage 2: keep both failure classes as labeled queries for strict pairing.
        self.negative = [r for r in records if r.training_quality == 'negative']

    def _admit_success_suffixes(self, code_positive, direct):
        groups = defaultdict(list)
        for r in code_positive:
            # Missing original segment identity must never join distinct programs.
            groups[(r.episode_id, r.source.get('segment_id') or r.sample_id)].append(r)
        for (episode, segment), windows in groups.items():
            anchors = [r for r in windows if r.sample_id in direct]
            if not anchors:
                continue
            if len({(r.split, instruction_key(r), r.spec.identity) for r in windows}) != 1:
                raise ValueError('Success segment contains inconsistent split/instruction/action spec')
            anchor = min(anchors, key=lambda r: (r.start_step, r.sample_id))
            suffix = sorted((r for r in windows if r.start_step >= anchor.start_step),
                            key=lambda r: (r.start_step, r.sample_id))
            def action_range(r):
                row = int(r.source.get('segment_row', r.start_step))
                lo, hi = r.source.get('action_range', [row, row + r.spec.horizon])
                if lo != row or hi <= lo or hi - lo > r.spec.horizon:
                    raise ValueError('Invalid success window action range')
                return int(lo), int(hi)
            start, end = action_range(anchor)
            origin = anchor.start_step - start
            for r in suffix:
                lo, hi = action_range(r)
                if r.start_step - lo != origin or lo > end:
                    raise ValueError('Success suffix windows do not cover a contiguous original action range')
                end = max(end, hi)
            entry = dict(episode_id=episode, segment_id=segment, split=anchor.split,
                anchor_id=anchor.sample_id, anchor_start_step=anchor.start_step,
                segment_row_range=[start, end], parent_step_range=[anchor.start_step, origin + end],
                action_steps=end-start, admitted_window_ids=[r.sample_id for r in suffix],
                direct_match_window_ids=[r.sample_id for r in suffix if r.sample_id in direct],
                anchor_vla_reference=direct[anchor.sample_id], source=anchor.source,
                image_refs=anchor.image_refs, instruction=anchor.instruction,
                admission_kind='successful_segment_suffix', frozen_bank_id=self.coverage_bank_id,
                observation_policy=OBSERVATION_POLICY)
            self.success_segments.append(entry)
            for r in suffix:
                self.coverage[r.sample_id] = dict(
                    reference=direct.get(r.sample_id, direct[anchor.sample_id]),
                    reference_applies_to='window' if r.sample_id in direct else 'suffix_anchor',
                    directly_matched=r.sample_id in direct, suffix_anchor_id=anchor.sample_id,
                    suffix_anchor_reference=direct[anchor.sample_id],
                    segment_id=segment, segment_row_range=[start,end],
                    admission_kind='successful_segment_suffix', frozen_bank_id=self.coverage_bank_id)
                self.positive.append(r)

    def match(self, query, references, thresholds):
        numeric = []
        for ref in references:
            if ref.sample_id == query.sample_id:
                continue
            d, reason = numeric_match(query, ref, thresholds)
            if reason is None:
                score = sum(d[k] / getattr(thresholds, k) for k in ['position_m', 'rotation_deg', 'proprio_rms'])
                numeric.append((score, ref.sample_id, ref, d))
        numeric.sort(key=lambda x: (x[0], x[1]))
        result = []
        for _, _, ref, d in numeric[:thresholds.max_candidates]:
            images, reason = image_distance(query, ref)
            if reason or max(images.values()) > thresholds.image_cosine:
                continue
            result.append(dict(sample_id=ref.sample_id, episode_id=ref.episode_id,
                               distances=dict(d, images=images), thresholds=asdict(thresholds),
                               encoder_sha256=query.visual_features['encoder_sha256']))
        return sorted(result, key=lambda x: (max(x['distances']['images'].values()), x['sample_id']))

    def build_pairs(self):
        output = []
        positive_by_task = defaultdict(list)
        for pos in self.positive:
            positive_by_task[(pos.split,instruction_key(pos))].append(pos)
        for neg in self.negative:
            candidates = self.match(neg, positive_by_task.get((neg.split,instruction_key(neg)),()), self.config.pairing)
            # Different full episodes first. At most one reference per episode on pass1.
            candidates.sort(key=lambda x: (x['episode_id'] == neg.episode_id,
                                           max(x['distances']['images'].values()), x['sample_id']))
            usable = []
            for candidate in candidates:
                pos = self.records[candidate['sample_id']]
                if pos.episode_id == neg.episode_id and pos.start_step == neg.start_step:
                    continue  # Opposing labels on the SAME starting action are contradictory.
                mask, reason = pair_mask(pos, neg, self.config)
                if reason:
                    continue
                usable.append(dict(candidate, mask=mask.tolist(), effective_steps=int(mask.any(-1).sum()),
                                   effective_elements=int(mask.sum())))
            chosen, episodes = [], set()
            for candidate in usable:
                if candidate['episode_id'] not in episodes:
                    chosen.append(candidate)
                    episodes.add(candidate['episode_id'])
                if len(chosen) == 3:
                    break
            if len(chosen) < 3:
                ids = {x['sample_id'] for x in chosen}
                chosen.extend(x for x in usable if x['sample_id'] not in ids)
                chosen = chosen[:3]
            if chosen:
                output.append(dict(pair_id=stable_id(neg.sample_id, [x['sample_id'] for x in chosen]),
                    negative_id=neg.sample_id, negative_episode_id=neg.episode_id,
                    event_id=neg.event_id, split=neg.split, references=chosen,
                    negative_evidence=neg.quality_evidence,
                    negative_label_basis=neg.label_basis, negative_bank_label=neg.bank_label,
                    error_mask_basis=neg.label_basis if neg.local_quality=='unknown' else 'verified_local_actions',
                    negative_start=neg.to_dict(),
                    positive_starts={x['sample_id']: self.records[x['sample_id']].to_dict() for x in chosen},
                    positive_coverage={x['sample_id']: self.coverage.get(x['sample_id']) for x in chosen},
                    coverage=self.coverage.get(neg.sample_id), frozen_bank_id=self.coverage_bank_id,
                    observation_policy=OBSERVATION_POLICY))
            else:
                self.rejections.append(dict(sample_id=neg.sample_id, reason='no_reliable_positive_reference'))
        return output
