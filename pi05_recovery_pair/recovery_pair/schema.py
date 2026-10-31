from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import numpy as np


def stable_id(*values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()[:24]


@dataclass(frozen=True)
class ActionSpec:
    representation: str
    fields: tuple[str, ...]
    units: tuple[str, ...]
    frame: str
    control_hz: int
    normalization_sha256: str
    horizon: int
    model_dim: int
    valid_dimensions: tuple[bool, ...]

    def validate(self):
        if not self.normalization_sha256 or len(self.fields) != self.model_dim or len(self.units) != self.model_dim:
            raise ValueError("Action fields, units and normalization identity are required")
        if len(self.valid_dimensions) != self.model_dim or not any(self.valid_dimensions):
            raise ValueError("Invalid comparable dimension mask")
        if self.horizon < 1 or self.control_hz < 1:
            raise ValueError("Invalid action horizon/frequency")
        return self

    @property
    def identity(self):
        return stable_id(asdict(self))


@dataclass
class Window:
    sample_id: str
    task_id: str
    episode_id: str
    initial_group_id: str
    event_id: str
    controller: str
    local_quality: str
    start_step: int
    start_time_s: float
    instruction: str
    phase: str | None
    target_object: str | None
    image_refs: dict
    proprio: list
    standardized_proprio: list
    eef_position_m: list
    eef_quaternion_xyzw: list
    gripper_state: str | None
    held_object: str | None
    contact_state: str | None
    progress: float | None
    action_path: str
    spec: ActionSpec
    split: str = "unassigned"
    quality_evidence: dict = field(default_factory=dict)
    visual_features: dict = field(default_factory=dict)
    object_poses: dict = field(default_factory=dict)
    uncertain_initialization: bool = False
    source: dict = field(default_factory=dict)
    bank_class: str | None = None
    bank_label: dict = field(default_factory=dict)

    @property
    def training_quality(self):
        """User-selected experiment: reuse existing segment classes as labels.

        local_quality remains optional finer action evidence; do not forge it.
        """
        if self.source.get('state_only'):
            return 'unknown'
        if self.bank_class:
            return self.bank_class.rsplit('_', 1)[1]
        return self.local_quality

    @property
    def label_basis(self):
        if self.controller == 'expert':
            return 'original_expert_demonstration'
        return 'existing_segment_class' if self.bank_class else 'verified_local_actions'

    def pair_error_mask(self, payload=None):
        """Use the selected negative segment label, or an explicit finer mask."""
        payload = self.payload() if payload is None else payload
        if self.bank_class and self.training_quality == 'negative' and self.local_quality == 'unknown':
            return payload['valid_mask'].copy()
        return payload['trusted_error_mask'].copy()

    def validate(self):
        if self.controller not in ("VLA", "code", "expert"):
            raise ValueError("Controller must be VLA/code/expert")
        if self.controller == 'expert' and (self.bank_class != 'D_expert_positive' or
                not self.source.get('expert_training_only') or self.source.get('state_only')):
            raise ValueError('Expert demonstrations must be positive and explicitly training-only')
        if self.bank_class is not None:
            expected = {'VLA':'D_vla_', 'code':'D_code_', 'expert':'D_expert_'}[self.controller]
            if self.bank_class not in (expected+'positive', expected+'negative'):
                raise ValueError('Bank class disagrees with controller')
        if self.local_quality not in ("positive", "negative", "unknown"):
            raise ValueError("Invalid local quality")
        if self.bank_class and self.local_quality != 'unknown' and self.local_quality != self.training_quality:
            raise ValueError('Existing segment class conflicts with explicit local action label')
        if not all([self.sample_id, self.task_id, self.episode_id, self.initial_group_id, self.event_id]):
            raise ValueError("Identity/group/event fields are required")
        if self.progress is not None and not 0 <= self.progress <= 1:
            raise ValueError("Progress is a calibrated [0,1] state score, not action quality")
        if len(self.eef_position_m) != 3 or len(self.eef_quaternion_xyzw) != 4:
            raise ValueError("Missing starting end-effector pose")
        for x in [self.proprio, self.standardized_proprio, self.eef_position_m, self.eef_quaternion_xyzw]:
            if not np.isfinite(np.asarray(x, dtype=float)).all():
                raise ValueError("Nonfinite starting state")
        if len(self.proprio) != len(self.standardized_proprio):
            raise ValueError("Proprio normalization dimension mismatch")
        q = np.asarray(self.eef_quaternion_xyzw)
        if abs(np.linalg.norm(q) - 1) > 1e-4:
            raise ValueError("EEF quaternion must have unit length")
        if self.local_quality in ("positive", "negative") and not self.quality_evidence.get("verified_local_actions"):
            raise ValueError("Weak episode/WCM labels cannot establish local action quality")
        if self.local_quality == "negative" and not self.quality_evidence.get("negative_behavior_confirmed"):
            raise ValueError("Being in a bad state does not prove subsequent actions are bad")
        self.spec.validate()
        return self

    def payload(self):
        with np.load(self.action_path, allow_pickle=False) as data:
            result = {k: data[k].copy() for k in data.files}
        shape = (self.spec.horizon, self.spec.model_dim)
        for key in ("actions", "valid_mask", "trusted_error_mask"):
            if result[key].shape != shape:
                raise ValueError(f"{self.sample_id}: invalid {key} shape")
        if not np.isfinite(result['actions']).all():
            raise ValueError("Nonfinite recorded actions")
        dims = np.asarray(self.spec.valid_dimensions, bool)[None]
        if (result['valid_mask'] & ~dims).any() or (result['trusted_error_mask'] & ~result['valid_mask']).any():
            raise ValueError("Padding/trusted masks are inconsistent")
        if self.local_quality != 'negative' and result['trusted_error_mask'].any():
            raise ValueError("Only verified negative behavior gets a trusted error mask")
        return result

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        spec = dict(data['spec'])
        for key in ['fields', 'units', 'valid_dimensions']:
            spec[key] = tuple(spec[key])
        data['spec'] = ActionSpec(**spec)
        return cls(**data).validate()


def save_windows(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r.to_dict(), ensure_ascii=False) + '\n' for r in records))


def load_windows(path):
    return [Window.from_dict(json.loads(s)) for s in Path(path).read_text().splitlines() if s.strip()]
