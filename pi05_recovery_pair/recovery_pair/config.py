from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class MatchThresholds:
    position_m: float = 0.03
    rotation_deg: float = 15.0
    proprio_rms: float | None = None
    image_cosine: float | None = None
    max_candidates: int = 50


@dataclass(frozen=True)
class Config:
    horizon: int = 10
    action_dim: int = 32
    environment_action_dim: int = 7
    native_control_hz: int = 20
    flow_steps: int = 3
    integrator: str = "euler"
    learning_rate: float = 1e-5
    gradient_clip: float = 1.0
    lambda_push: float = 0.1
    eta: float = 0.05
    eta_warmup_fraction: float = 0.10
    fm_batch: int = 64
    pair_batch: int = 4
    vla_fraction: float = 0.75
    fm_micro_batch: int = 1
    pair_micro_batch: int = 1
    max_episode_reuses: int = 50
    fm_only_updates: int = 100  # Legacy replacement sampler only when fm_epochs=None.
    fm_epochs: int | None = None
    min_pair_steps: int = 3
    identical_action_tolerance: float = 1e-4
    shared_prefix_tolerance: float = 1e-4
    state_stride: int = 10
    seed: int = 42
    checkpoint_sampling: bool = True
    conservative_task_split: bool = True
    matching_policy: str = 'cap_success_to_vla_success_v1'
    cap_success_admission: str = 'matched_window'
    negative_query_policy: str = 'vla_failure_only_v1'
    split_fractions: tuple = (0.8, 0.1, 0.1)
    coverage: MatchThresholds = field(default_factory=MatchThresholds)
    pairing: MatchThresholds = field(default_factory=lambda: MatchThresholds(
        position_m=0.01, rotation_deg=5.0))

    def validate(self, *, matching_ready=False):
        if self.fm_epochs is not None:
            if type(self.fm_epochs) is not int or self.fm_epochs < 1:
                raise ValueError('fm_epochs must be a positive integer or null for legacy update sampling')
            if self.pair_batch != 0 or self.eta != 0:
                raise ValueError('Full positive epochs require pair_batch=0 and eta=0')
        if self.matching_policy != 'cap_success_to_vla_success_v1':
            raise ValueError('Unsupported matching policy')
        if self.cap_success_admission not in ('matched_window', 'segment_suffix'):
            raise ValueError('Unsupported CAP success admission policy')
        if self.negative_query_policy != 'vla_failure_only_v1':
            raise ValueError('Negative matching queries must use VLA failures only')
        for key in ('learning_rate','gradient_clip','lambda_push','eta','eta_warmup_fraction','vla_fraction',
                    'identical_action_tolerance','shared_prefix_tolerance'):
            if not math.isfinite(getattr(self,key)):
                raise ValueError(f'{key} must be finite')
        for key in ('horizon','action_dim','environment_action_dim','native_control_hz','flow_steps','fm_batch',
                    'pair_batch','fm_micro_batch','pair_micro_batch','max_episode_reuses','fm_only_updates',
                    'min_pair_steps','state_stride','seed'):
            if type(getattr(self,key)) is not int:
                raise ValueError(f'{key} must be integer')
        if not 0 < self.lambda_push < 1:
            raise ValueError("Continuous squared-distance repulsion requires 0 < lambda_push < 1")
        if self.eta < 0 or not 0 < self.eta_warmup_fraction <= 1:
            raise ValueError("Invalid pair coefficient/warmup")
        if self.learning_rate <= 0 or self.gradient_clip <= 0 or self.fm_only_updates < 1:
            raise ValueError('Invalid optimizer/FM-only budget')
        if self.integrator != "euler":
            raise ValueError("This implementation supports deployment-matched Euler only")
        if any(v < 1 for v in [self.horizon, self.flow_steps, self.action_dim, self.fm_batch,
                               self.fm_micro_batch, self.pair_micro_batch, self.max_episode_reuses,
                               self.state_stride, self.min_pair_steps]):
            raise ValueError("Dimensions, batches and budgets must be positive")
        if self.pair_batch < 0 or self.pair_batch >= self.fm_batch:
            raise ValueError("Need 0 <= pair_batch < fm_batch")
        if not 0 <= self.vla_fraction <= 1:
            raise ValueError("Invalid VLA:code ratio")
        if self.min_pair_steps > self.horizon:
            raise ValueError("min_pair_steps exceeds horizon")
        if self.environment_action_dim > self.action_dim:
            raise ValueError("Environment action dimension exceeds model dimension")
        if len(self.split_fractions) != 3 or any(x < 0 for x in self.split_fractions) or abs(sum(self.split_fractions) - 1) > 1e-8:
            raise ValueError("Train/validation/test fractions must sum to one")
        for t in [self.coverage, self.pairing]:
            if not 1 <= t.max_candidates <= 50:
                raise ValueError("At most50 numeric candidates may reach visual comparison")
            for key in ["position_m", "rotation_deg", "proprio_rms", "image_cosine"]:
                v = getattr(t, key)
                if v is not None and (not math.isfinite(v) or v <= 0):
                    raise ValueError(f"{key} must be positive")
            if matching_ready and (t.proprio_rms is None or t.image_cosine is None):
                raise ValueError("Calibrate proprio/image thresholds on TRAIN candidates before reliable matching")
        for key in ["position_m", "rotation_deg", "proprio_rms", "image_cosine"]:
            a, b = getattr(self.coverage, key), getattr(self.pairing, key)
            if a is not None and b is not None and b > a:
                raise ValueError(f"Pairing {key} cannot be wider than coverage")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if 'split_fractions' in data:
            data['split_fractions'] = tuple(data['split_fractions'])
        for key in ["coverage", "pairing"]:
            if key in data:
                thresholds = dict(data[key])
                # Read old immutable configs without restoring the removed criterion.
                thresholds.pop('progress', None)
                thresholds.pop('object_position_m', None)
                data[key] = MatchThresholds(**thresholds)
        return cls(**data).validate()
