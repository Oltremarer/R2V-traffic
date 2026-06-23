from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


FORMAL_REWARD_ADAPTERS = {
    "film_scalar_potential",
    "weighted_proxy",
    "env_reward",
    "pressure",
    "vectorq_diagnostic_potential",
}


StageName = Literal["dry_run", "preflight", "formal"]


def is_placeholder_value(value: str | None) -> bool:
    if value is None:
        return False
    lowered = value.strip().lower()
    return any(marker in lowered for marker in ("placeholder", "todo", "tbd", "dummy"))


@dataclass(frozen=True)
class FormalExperimentSpec:
    scenario: str
    traffic_file: str
    cityflow_seed: int
    policy_seed: int
    model_seed: int
    state_encoder_id: str
    state_encoder_hash: str
    feature_norm_path: str | None
    objective_norm_path: str
    objective_normalizer_hash: str
    film_model_dir: str | None
    film_model_hash: str | None
    film_model_selection_report: str | None
    film_training_commit: str | None
    reward_adapter: str
    reward_scale: float
    reward_clip: float | None
    reward_normalization: str
    potential_gamma: float
    mix_env_reward: bool
    policy_conditioned_on_w: bool
    critic_conditioned_on_w: bool
    preference_sampling: str
    train_preferences: list[str]
    eval_preferences: list[str]
    min_action_time: int
    action_interval_source: str
    forbid_inner_step_direct_call: bool
    ppo_budget: dict[str, Any]
    eval_protocol: str
    formal_gate_decision_path: str
    approved_formal_spec: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FormalExperimentSpec":
        spec = cls(**payload)
        spec.validate()
        return spec

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.reward_adapter not in FORMAL_REWARD_ADAPTERS:
            raise ValueError(f"unknown reward_adapter: {self.reward_adapter}")
        if self.reward_adapter == "film_scalar_potential":
            if not self.film_model_dir:
                raise ValueError("film_model_dir is required for FiLMScalar-PPO")
            if not self.film_model_hash:
                raise ValueError("film_model_hash is required for FiLMScalar-PPO")
            if not self.policy_conditioned_on_w:
                raise ValueError("policy_conditioned_on_w must be true for FiLMScalar-PPO")
            if not self.critic_conditioned_on_w:
                raise ValueError("critic_conditioned_on_w must be true for FiLMScalar-PPO")
        if self.min_action_time <= 0:
            raise ValueError("min_action_time must be positive")
        if not self.forbid_inner_step_direct_call:
            raise ValueError("formal runs must forbid direct env._inner_step usage")
        if not self.state_encoder_hash:
            raise ValueError("state_encoder_hash is required")
        if not self.objective_normalizer_hash:
            raise ValueError("objective_normalizer_hash is required")
        if self.preference_sampling not in {"episode_fixed", "rollout_fixed"}:
            raise ValueError("preference_sampling must be episode_fixed or rollout_fixed")
        if not self.eval_preferences:
            raise ValueError("eval_preferences cannot be empty")

    def validate_for_stage(self, stage: StageName) -> None:
        self.validate()
        if stage == "dry_run":
            return
        if stage not in {"preflight", "formal"}:
            raise ValueError(f"unknown formal validation stage: {stage}")
        for field_name in ("state_encoder_hash", "objective_normalizer_hash"):
            value = getattr(self, field_name)
            if is_placeholder_value(value):
                raise ValueError(f"{field_name} contains placeholder value for {stage}: {value}")
        if self.reward_adapter == "film_scalar_potential" and is_placeholder_value(self.film_model_hash):
            raise ValueError(f"film_model_hash contains placeholder value for {stage}: {self.film_model_hash}")

    def spec_hash(self) -> str:
        stable = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def load_formal_experiment_spec(path: str | Path) -> FormalExperimentSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return FormalExperimentSpec.from_dict(payload)


def write_formal_experiment_spec(path: str | Path, spec: FormalExperimentSpec) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
