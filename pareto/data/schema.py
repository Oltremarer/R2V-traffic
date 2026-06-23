from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from typing import Any, Dict, List, Optional

from pareto.constants import OBJECTIVE_NAMES


def _loads_json(payload: str | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload, str):
        return json.loads(payload)
    return dict(payload)


def _require_objective_keys(values: Dict[str, Any], field_name: str) -> None:
    missing = [name for name in OBJECTIVE_NAMES if name not in values]
    if missing:
        raise ValueError(f"{field_name} missing objective keys: {missing}")


def validate_preference(w: List[float]) -> None:
    if len(w) != len(OBJECTIVE_NAMES):
        raise ValueError(f"preference must have {len(OBJECTIVE_NAMES)} weights")
    if any(weight < 0 for weight in w):
        raise ValueError("preference weights must be non-negative")
    if abs(sum(w) - 1.0) > 1e-6:
        raise ValueError("preference weights must sum to 1")


@dataclass
class JsonRecord:
    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str | Dict[str, Any]):
        data = _loads_json(payload)
        valid_names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in valid_names})

    def validate(self) -> None:
        return None


@dataclass
class TrajectoryRecord(JsonRecord):
    schema_version: str
    run_id: str
    sample_id: str
    scenario: str
    roadnet: str
    traffic_file: str
    seed: int
    episode: int
    step: int
    sim_time_sec: float
    intersection_id: str
    policy_id: str
    action: Optional[int]
    action_name: Optional[str]
    phase_id: Optional[int]
    phase_duration: Optional[float]
    obs_features: List[float]
    obs_feature_names: List[str]
    llm_state: Dict[str, Any]
    incoming_state: Dict[str, Any]
    mean_speed: float
    objective_values_raw: Dict[str, float]
    objective_values_norm: Dict[str, float]
    objective_valid_mask: Dict[str, bool]
    prev_sample_id: Optional[str]
    next_sample_id: Optional[str]
    metadata: Dict[str, Any]

    def validate(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id is required")
        if len(self.obs_features) != len(self.obs_feature_names):
            raise ValueError("feature vector and feature names length mismatch")
        _require_objective_keys(self.objective_values_raw, "objective_values_raw")
        _require_objective_keys(self.objective_values_norm, "objective_values_norm")
        _require_objective_keys(self.objective_valid_mask, "objective_valid_mask")


@dataclass
class ObjectivePairLabel(JsonRecord):
    pair_id: str
    split: str
    scenario: str
    a_id: str
    b_id: str
    objective: str
    label: int
    is_tie: bool
    margin_raw: float
    margin_norm: float
    source: str
    rule_version: str
    sampling_strategy: str

    def validate(self) -> None:
        if self.objective not in OBJECTIVE_NAMES:
            raise ValueError(f"unknown objective: {self.objective}")
        if self.is_tie:
            raise ValueError("tie pairs must be filtered before serialization")
        if self.label not in (0, 1):
            raise ValueError("label must be 0 or 1")


@dataclass
class PreferencePairLabel(JsonRecord):
    pair_id: str
    split: str
    scenario: str
    a_id: str
    b_id: str
    w: List[float]
    label: int
    is_tie: bool
    source: str
    rule_utility_a: float
    rule_utility_b: float
    rule_margin: float
    sampling_strategy: str
    confidence: Optional[float] = None
    llm_model: Optional[str] = None
    prompt_hash: Optional[str] = None
    response_hash: Optional[str] = None
    raw_response_path: Optional[str] = None

    def validate(self) -> None:
        validate_preference(self.w)
        if self.is_tie:
            raise ValueError("tie pairs must be filtered before serialization")
        if self.label not in (0, 1):
            raise ValueError("label must be 0 or 1")


@dataclass
class DominancePairLabel(JsonRecord):
    pair_id: str
    split: str
    scenario: str
    a_id: str
    b_id: str
    dominates: str
    objective_margins_norm: Dict[str, float]
    min_margin_norm: float
    dominance_threshold: float
    source: str

    def validate(self) -> None:
        if self.dominates not in ("a", "b"):
            raise ValueError("dominates must be 'a' or 'b'")
        _require_objective_keys(self.objective_margins_norm, "objective_margins_norm")


@dataclass
class TransitionRecord(JsonRecord):
    schema_version: str
    run_id: str
    transition_id: str
    sample_id: str
    next_sample_id: Optional[str]
    scenario: str
    traffic_file: str
    seed: int
    episode: int
    step: int
    intersection_id: str
    obs_features: List[float]
    next_obs_features: List[float]
    action: int
    env_reward: float
    objectives_t_norm: Dict[str, float]
    objectives_tp1_norm: Dict[str, float]
    done: bool
    policy_id: str
    metadata: Dict[str, Any]

    def validate(self) -> None:
        if not self.transition_id:
            raise ValueError("transition_id is required")
        if not self.sample_id:
            raise ValueError("sample_id is required")
        if not self.done and not self.next_sample_id:
            raise ValueError("next_sample_id is required for non-terminal transitions")
        if len(self.obs_features) != len(self.next_obs_features):
            raise ValueError("obs_features and next_obs_features length mismatch")
        _require_objective_keys(self.objectives_t_norm, "objectives_t_norm")
        _require_objective_keys(self.objectives_tp1_norm, "objectives_tp1_norm")


@dataclass
class EvaluationResult(JsonRecord):
    eval_id: str
    run_id: str
    method: str
    scenario: str
    seed: int
    checkpoint_path: Optional[str]
    eval_preference: List[float]
    eval_preference_name: str
    conventional_metrics: Dict[str, float]
    objective_values_raw: Dict[str, float]
    objective_values_norm: Dict[str, float]
    utility_under_w: float
    action_histogram: Dict[str, int]
    status: str
    failure_reason: Optional[str] = None

    def validate(self) -> None:
        validate_preference(self.eval_preference)
        _require_objective_keys(self.objective_values_raw, "objective_values_raw")
        _require_objective_keys(self.objective_values_norm, "objective_values_norm")
