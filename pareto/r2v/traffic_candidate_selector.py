from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

from pareto.constants import OBJECTIVE_NAMES
from pareto.r2v.artifact_validation import (
    get_path_value,
    validate_unique_transition_ids,
    validate_weighted_transition_rows,
)
from pareto.r2v.generative_scorer import GenerativeScoreRow, load_generative_score_artifact
from pareto.r2v.traffic_artifact_schema import (
    upgrade_weighted_row_to_v2_metadata,
    validate_r2v_traffic_artifact,
)


DEFAULT_TRAFFIC_UTILITY_WEIGHTS = {
    "efficiency": 0.5,
    "safety": 0.2,
    "fairness": 0.15,
    "stability": 0.15,
}

SUPPORTED_VALUE_MODES = {"objective_delta", "frozen_target_utility"}
SUPPORTED_REPAIR_STORIES = {"none", "not_val_to_val", "not_rare_to_val"}
SUPPORTED_REPAIR_METADATA_POLICIES = {"require_metadata", "metadata_or_proxy"}
SUPPORTED_GATE_VARIANTS = {"full", "no_support", "no_ood", "no_dynamics"}
SUPPORTED_ADMISSION_MODES = {"weights_only", "weights_plus_repaired"}
REQUIRED_GATE_KEYS = ("rare", "value", "support", "safety")
_MISSING = object()


@dataclass(frozen=True)
class R2VTrafficSelectorConfig:
    candidate_model: str = "feature_density_proxy"
    value_mode: str = "objective_delta"
    rare_quantile: float = 0.8
    value_quantile: float = 0.6
    support_min_quantile: float = 0.02
    safety_min: float = -1.0
    env_reward_weight: float = 0.0
    density_neighbors: int = 1
    base_weight: float = 1.0
    admitted_weight: float | None = None
    admitted_weight_bonus: float = 2.0
    max_weight: float = 5.0
    utility_weights: dict[str, float] | None = None
    score_artifact_path: str | None = None
    score_artifact_id_key: str = "transition_id"
    score_artifact_backend: str | None = None
    repair_story: str = "none"
    repair_metadata_policy: str = "require_metadata"
    gate_variant: str = "full"
    admission_mode: str = "weights_only"
    repair_rejected_weight: float = 2.0
    source_gates_key: str = "metadata.r2v_source_gates"
    final_gates_key: str = "metadata.r2v_final_gates"
    repaired_transition_key: str = "metadata.r2v_repaired_transition"

    def validate(self) -> None:
        if self.value_mode not in SUPPORTED_VALUE_MODES:
            raise ValueError(f"value_mode must be one of {sorted(SUPPORTED_VALUE_MODES)}")
        if self.repair_story not in SUPPORTED_REPAIR_STORIES:
            raise ValueError(f"repair_story must be one of {sorted(SUPPORTED_REPAIR_STORIES)}")
        if self.repair_metadata_policy not in SUPPORTED_REPAIR_METADATA_POLICIES:
            raise ValueError(f"repair_metadata_policy must be one of {sorted(SUPPORTED_REPAIR_METADATA_POLICIES)}")
        if self.gate_variant not in SUPPORTED_GATE_VARIANTS:
            raise ValueError(f"gate_variant must be one of {sorted(SUPPORTED_GATE_VARIANTS)}")
        if self.admission_mode not in SUPPORTED_ADMISSION_MODES:
            raise ValueError(f"admission_mode must be one of {sorted(SUPPORTED_ADMISSION_MODES)}")
        if self.value_mode == "frozen_target_utility" and abs(float(self.env_reward_weight)) > 1e-12:
            raise ValueError("frozen_target_utility mode requires env_reward_weight=0.0")
        for name, value in (
            ("rare_quantile", self.rare_quantile),
            ("value_quantile", self.value_quantile),
            ("support_min_quantile", self.support_min_quantile),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.base_weight <= 0:
            raise ValueError("base_weight must be positive")
        if self.density_neighbors < 1:
            raise ValueError("density_neighbors must be at least 1")
        if self.admitted_weight is not None and self.admitted_weight <= 0:
            raise ValueError("admitted_weight must be positive when provided")
        if self.admitted_weight_bonus < 0:
            raise ValueError("admitted_weight_bonus must be non-negative")
        if self.repair_rejected_weight <= 0:
            raise ValueError("repair_rejected_weight must be positive")
        if self.max_weight < self.base_weight:
            raise ValueError("max_weight must be at least base_weight")
        if self.score_artifact_path is not None and str(self.score_artifact_path).strip() == "":
            raise ValueError("score_artifact_path must be non-empty when provided")
        if str(self.score_artifact_id_key).strip() == "":
            raise ValueError("score_artifact_id_key must be non-empty")
        if str(self.source_gates_key).strip() == "":
            raise ValueError("source_gates_key must be non-empty")
        if str(self.final_gates_key).strip() == "":
            raise ValueError("final_gates_key must be non-empty")
        if str(self.repaired_transition_key).strip() == "":
            raise ValueError("repaired_transition_key must be non-empty")
        self.resolved_utility_weights()

    def resolved_utility_weights(self) -> dict[str, float]:
        if self.utility_weights is None:
            return dict(DEFAULT_TRAFFIC_UTILITY_WEIGHTS)
        missing = [name for name in OBJECTIVE_NAMES if name not in self.utility_weights]
        if missing:
            raise ValueError(f"utility_weights missing objective keys: {missing}")
        return {name: float(self.utility_weights[name]) for name in OBJECTIVE_NAMES}

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["utility_weights"] = self.resolved_utility_weights()
        return data


def select_r2v_candidates(
    transitions: Iterable[dict[str, Any]],
    config: R2VTrafficSelectorConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = config or R2VTrafficSelectorConfig()
    cfg.validate()
    rows = [dict(row) for row in transitions]
    if not rows:
        return [], _summary([], cfg, thresholds=_empty_thresholds(cfg))

    feature_rows = [_transition_feature_vector(row) for row in rows]
    center, scale = _feature_center_scale(feature_rows)
    standardized_rows = [_standardize(features, center, scale) for features in feature_rows]
    score_source_summary, score_sources, rarity_scores, support_scores = _resolve_candidate_scores(
        rows,
        standardized_rows,
        cfg,
    )
    value_components = [_value_components(row, cfg) for row in rows]
    value_scores = [components["value_score"] for components in value_components]
    safety_scores = [_safe_float(row.get("objectives_tp1_norm", {}).get("safety", 0.0)) for row in rows]

    thresholds = {
        "rarity_min": _quantile(rarity_scores, cfg.rare_quantile),
        "value_min": _quantile(value_scores, cfg.value_quantile),
        "support_min": _quantile(support_scores, cfg.support_min_quantile),
        "safety_min": float(cfg.safety_min),
    }
    raw_candidates: list[dict[str, Any]] = []
    max_admission = 0.0
    for idx, row in enumerate(rows):
        computed_gates = {
            "rare": rarity_scores[idx] >= thresholds["rarity_min"],
            "value": value_scores[idx] >= thresholds["value_min"],
            "support": support_scores[idx] >= thresholds["support_min"],
            "safety": safety_scores[idx] >= thresholds["safety_min"],
        }
        repair_status = _repair_story_status(row, computed_gates, cfg)
        gates = repair_status["gates"]
        active_gates = _active_gate_keys(cfg.gate_variant)
        admitted = all(gates[name] for name in active_gates) and bool(repair_status["repair_story_match"])
        admission_score_components = _admission_score_components(
            rarity=rarity_scores[idx],
            value=value_scores[idx],
            support=support_scores[idx],
            safety=safety_scores[idx],
            thresholds=thresholds,
        )
        admission_score = _admission_score(
            components=admission_score_components,
            admitted=admitted,
        )
        max_admission = max(max_admission, admission_score)
        raw_candidates.append(
            {
                "schema_version": "r2v-tsc-candidate-v1",
                "candidate_model": cfg.candidate_model,
                "transition_id": row.get("transition_id"),
                "sample_id": row.get("sample_id"),
                "next_sample_id": row.get("next_sample_id"),
                "scenario": row.get("scenario"),
                "traffic_file": row.get("traffic_file"),
                "seed": row.get("seed"),
                "episode": row.get("episode"),
                "step": row.get("step"),
                "intersection_id": row.get("intersection_id"),
                "policy_id": row.get("policy_id"),
                "action": row.get("action"),
                "rarity_score": rarity_scores[idx],
                "value_score": value_scores[idx],
                "support_score": support_scores[idx],
                "safety_score": safety_scores[idx],
                "admission_score": admission_score,
                "admission_score_components": admission_score_components,
                "gates": gates,
                "admitted": admitted,
                "gate_variant": cfg.gate_variant,
                "active_gates": list(active_gates),
                "repair_story": cfg.repair_story,
                "repair_metadata_policy": cfg.repair_metadata_policy,
                "repair_story_match": repair_status["repair_story_match"],
                "source_gates": repair_status["source_gates"],
                "final_gates": repair_status["final_gates"],
                "gate_source": repair_status["gate_source"],
                "repaired_transition": score_sources[idx].get("repaired_transition"),
                "repaired_transition_source": score_sources[idx].get("repaired_transition_source"),
                "sample_weight": cfg.base_weight,
                "thresholds": dict(thresholds),
                "debug": {
                    "utility_t": _utility(row.get("objectives_t_norm", {}), cfg),
                    "utility_tp1": _utility(row.get("objectives_tp1_norm", {}), cfg),
                    "env_reward": _safe_float(row.get("env_reward", 0.0)),
                    "value_mode": cfg.value_mode,
                    "utility_delta": value_components[idx]["utility_delta"],
                    "env_reward_component": value_components[idx]["env_reward_component"],
                    "frozen_target_utility": value_components[idx]["frozen_target_utility"],
                    "feature_distance": rarity_scores[idx],
                    "feature_dim": len(feature_rows[idx]),
                    "score_source": score_sources[idx],
                    "rare_is_not_value": True,
                    "computed_gates": dict(computed_gates),
                    "repair_story": cfg.repair_story,
                    "repair_metadata_policy": cfg.repair_metadata_policy,
                    "gate_variant": cfg.gate_variant,
                    "active_gates": list(active_gates),
                    "repair_story_match": repair_status["repair_story_match"],
                    "source_gates": repair_status["source_gates"],
                    "final_gates": repair_status["final_gates"],
                    "gate_source": repair_status["gate_source"],
                },
            }
        )

    candidates = _rank_and_weight(raw_candidates, cfg, max_admission)
    return candidates, _summary(candidates, cfg, thresholds, score_source=score_source_summary)


def apply_candidate_weights(
    transitions: Iterable[dict[str, Any]],
    candidates: Iterable[dict[str, Any]],
    *,
    config: R2VTrafficSelectorConfig | None = None,
    summary: dict[str, Any] | None = None,
    summary_output: str | None = None,
) -> list[dict[str, Any]]:
    transition_rows = [dict(row) for row in transitions]
    candidate_rows = [dict(candidate) for candidate in candidates]
    validate_unique_transition_ids(transition_rows)
    validate_unique_transition_ids(candidate_rows)
    candidate_by_transition = {
        str(candidate.get("transition_id")): candidate
        for candidate in candidate_rows
    }
    weighted_rows = []
    cfg = config or R2VTrafficSelectorConfig()
    summary_schema = summary.get("schema_version") if isinstance(summary, dict) else None
    for row in transition_rows:
        copied = dict(row)
        metadata = dict(copied.get("metadata") or {})
        candidate = candidate_by_transition.get(str(copied.get("transition_id")))
        metadata["r2v_schema_version"] = "r2v-tsc-weighted-transition-v1"
        metadata["r2v_value_mode"] = cfg.value_mode
        metadata["r2v_repair_story"] = cfg.repair_story
        metadata["r2v_repair_metadata_policy"] = cfg.repair_metadata_policy
        metadata["r2v_admission_mode"] = cfg.admission_mode
        metadata["r2v_row_role"] = "source"
        metadata["r2v_repaired_from_transition_id"] = None
        metadata["r2v_proposal_source"] = None
        metadata["r2v_repair_rejected"] = False
        metadata["r2v_score_artifact_path"] = str(cfg.score_artifact_path) if cfg.score_artifact_path else None
        metadata["r2v_score_artifact_backend"] = _traffic_artifact_backend(cfg)
        metadata["r2v_source_summary"] = summary_output
        metadata["r2v_source_summary_schema_version"] = summary_schema
        if candidate is None:
            metadata["r2v_sample_weight"] = float(cfg.base_weight)
            metadata["r2v_admitted"] = False
            metadata["r2v_candidate_model"] = cfg.candidate_model
            metadata["r2v_admission_score"] = 0.0
            metadata["r2v_candidate_rank"] = None
            metadata["r2v_gate_variant"] = cfg.gate_variant
            metadata["r2v_active_gates"] = list(_active_gate_keys(cfg.gate_variant))
            metadata["r2v_gates"] = {
                "rare": False,
                "value": False,
                "support": False,
                "safety": False,
            }
            metadata["r2v_repair_story_match"] = False
            metadata["r2v_source_gates"] = None
            metadata["r2v_final_gates"] = dict(metadata["r2v_gates"])
            metadata["r2v_computed_gates"] = dict(metadata["r2v_gates"])
            metadata["r2v_gate_source"] = "missing_candidate"
        else:
            metadata["r2v_sample_weight"] = float(candidate["sample_weight"])
            metadata["r2v_admitted"] = bool(candidate["admitted"])
            metadata["r2v_candidate_model"] = candidate["candidate_model"]
            metadata["r2v_admission_score"] = float(candidate["admission_score"])
            metadata["r2v_candidate_rank"] = candidate.get("candidate_rank")
            metadata["r2v_gates"] = dict(candidate.get("gates") or {})
            metadata["r2v_rarity_score"] = float(candidate.get("rarity_score", 0.0))
            metadata["r2v_value_score"] = float(candidate.get("value_score", 0.0))
            metadata["r2v_support_score"] = float(candidate.get("support_score", 0.0))
            metadata["r2v_safety_score"] = float(candidate.get("safety_score", 0.0))
            metadata["r2v_repair_story"] = str(candidate.get("repair_story", cfg.repair_story))
            metadata["r2v_repair_metadata_policy"] = str(
                candidate.get("repair_metadata_policy", cfg.repair_metadata_policy)
            )
            metadata["r2v_gate_variant"] = str(candidate.get("gate_variant", cfg.gate_variant))
            metadata["r2v_active_gates"] = list(candidate.get("active_gates") or _active_gate_keys(cfg.gate_variant))
            metadata["r2v_repair_story_match"] = bool(candidate.get("repair_story_match", False))
            metadata["r2v_source_gates"] = candidate.get("source_gates")
            metadata["r2v_final_gates"] = candidate.get("final_gates")
            metadata["r2v_computed_gates"] = dict(candidate.get("debug", {}).get("computed_gates") or {})
            metadata["r2v_gate_source"] = str(candidate.get("gate_source", "computed"))
        copied["metadata"] = metadata
        weighted_rows.append(copied)
    if cfg.admission_mode == "weights_plus_repaired":
        weighted_rows.extend(
            _repaired_weighted_rows(
                transition_rows,
                candidate_by_transition,
                cfg=cfg,
                summary_schema=summary_schema,
                summary_output=summary_output,
            )
        )
    validate_weighted_transition_rows(weighted_rows)
    traffic_rows = [
        upgrade_weighted_row_to_v2_metadata(
            row,
            gate_variant=cfg.gate_variant,
            generative_backend=_traffic_artifact_backend(cfg),
            admission_mode=cfg.admission_mode,
        )
        for row in weighted_rows
    ]
    validate_r2v_traffic_artifact(traffic_rows)
    return traffic_rows


def _repaired_weighted_rows(
    transition_rows: list[dict[str, Any]],
    candidate_by_transition: dict[str, dict[str, Any]],
    *,
    cfg: R2VTrafficSelectorConfig,
    summary_schema: Any,
    summary_output: str | None,
) -> list[dict[str, Any]]:
    repaired_rows: list[dict[str, Any]] = []
    for source_row in transition_rows:
        transition_id = str(source_row.get("transition_id"))
        candidate = candidate_by_transition.get(transition_id)
        if candidate is None:
            continue
        admitted = bool(candidate.get("admitted", False))
        repaired_raw, proposal_source = _resolve_repaired_transition_payload(source_row, candidate, cfg=cfg)
        if repaired_raw is _MISSING:
            if not admitted:
                continue
            raise ValueError(
                "admission_mode='weights_plus_repaired' requires repaired transition "
                f"from score artifact or at {cfg.repaired_transition_key!r} "
                f"for admitted transition_id {transition_id!r}"
            )
        if repaired_raw is None:
            if not admitted:
                continue
            raise ValueError(
                "admission_mode='weights_plus_repaired' requires repaired transition "
                f"from score artifact or at {cfg.repaired_transition_key!r} "
                f"for admitted transition_id {transition_id!r}"
            )
        if not isinstance(repaired_raw, dict):
            raise ValueError(
                f"{cfg.repaired_transition_key!r} must be an object for transition_id {transition_id!r}"
            )
        repaired = dict(repaired_raw)
        repaired_transition_id = str(repaired.get("transition_id") or "")
        repaired_sample_id = str(repaired.get("sample_id") or "")
        if not repaired_transition_id:
            raise ValueError(
                "admission_mode='weights_plus_repaired' repaired transition missing transition_id "
                f"for source transition_id {transition_id!r}"
            )
        if not repaired_sample_id:
            raise ValueError(
                "admission_mode='weights_plus_repaired' repaired transition missing sample_id "
                f"for source transition_id {transition_id!r}"
            )
        metadata = dict(repaired.get("metadata") or {})
        metadata["r2v_schema_version"] = "r2v-tsc-weighted-transition-v1"
        metadata["r2v_value_mode"] = cfg.value_mode
        metadata["r2v_repair_story"] = str(candidate.get("repair_story", cfg.repair_story))
        metadata["r2v_repair_metadata_policy"] = str(
            candidate.get("repair_metadata_policy", cfg.repair_metadata_policy)
        )
        metadata["r2v_admission_mode"] = cfg.admission_mode
        metadata["r2v_row_role"] = "repaired" if admitted else "repair_rejected"
        metadata["r2v_repaired_from_transition_id"] = transition_id
        metadata["r2v_proposal_source"] = str(proposal_source)
        metadata["r2v_score_artifact_path"] = str(cfg.score_artifact_path) if cfg.score_artifact_path else None
        metadata["r2v_score_artifact_backend"] = _traffic_artifact_backend(cfg)
        metadata["r2v_source_summary"] = summary_output
        metadata["r2v_source_summary_schema_version"] = summary_schema
        metadata["r2v_sample_weight"] = float(candidate["sample_weight"]) if admitted else float(cfg.repair_rejected_weight)
        metadata["r2v_admitted"] = admitted
        metadata["r2v_repair_rejected"] = not admitted
        metadata["r2v_candidate_model"] = candidate["candidate_model"]
        metadata["r2v_admission_score"] = float(candidate["admission_score"])
        metadata["r2v_candidate_rank"] = candidate.get("candidate_rank")
        metadata["r2v_gates"] = dict(candidate.get("gates") or {})
        metadata["r2v_rarity_score"] = float(candidate.get("rarity_score", 0.0))
        metadata["r2v_value_score"] = float(candidate.get("value_score", 0.0))
        metadata["r2v_support_score"] = float(candidate.get("support_score", 0.0))
        metadata["r2v_safety_score"] = float(candidate.get("safety_score", 0.0))
        metadata["r2v_gate_variant"] = str(candidate.get("gate_variant", cfg.gate_variant))
        metadata["r2v_active_gates"] = list(candidate.get("active_gates") or _active_gate_keys(cfg.gate_variant))
        metadata["r2v_repair_story_match"] = bool(candidate.get("repair_story_match", False))
        metadata["r2v_source_gates"] = candidate.get("source_gates")
        metadata["r2v_final_gates"] = candidate.get("final_gates")
        metadata["r2v_computed_gates"] = dict(candidate.get("debug", {}).get("computed_gates") or {})
        metadata["r2v_gate_source"] = str(candidate.get("gate_source", "metadata_repaired_transition"))
        repaired["metadata"] = metadata
        repaired_rows.append(repaired)
    return repaired_rows


def _resolve_repaired_transition_payload(
    source_row: dict[str, Any],
    candidate: dict[str, Any],
    *,
    cfg: R2VTrafficSelectorConfig,
) -> tuple[Any, str]:
    repaired_raw = candidate.get("repaired_transition", _MISSING)
    proposal_source = candidate.get("repaired_transition_source") or "score_artifact.repaired_transition"
    if repaired_raw is _MISSING or repaired_raw is None:
        repaired_raw = get_path_value(source_row, cfg.repaired_transition_key, default=_MISSING)
        proposal_source = cfg.repaired_transition_key
    return repaired_raw, str(proposal_source)


def _transition_feature_vector(row: dict[str, Any]) -> list[float]:
    obs = [_safe_float(value) for value in row.get("obs_features", [])]
    next_obs = [_safe_float(value) for value in row.get("next_obs_features", [])]
    if len(obs) != len(next_obs):
        raise ValueError(f"feature length mismatch for transition {row.get('transition_id')}")
    delta = [right - left for left, right in zip(obs, next_obs)]
    return obs + delta


def _feature_center_scale(feature_rows: list[list[float]]) -> tuple[list[float], list[float]]:
    width = len(feature_rows[0])
    if any(len(row) != width for row in feature_rows):
        raise ValueError("all transition feature vectors must have the same length")
    center = []
    scale = []
    for idx in range(width):
        values = [row[idx] for row in feature_rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
        std = math.sqrt(max(variance, 0.0))
        center.append(mean)
        scale.append(std if std > 1e-9 else 1.0)
    return center, scale


def _standardize(features: list[float], center: list[float], scale: list[float]) -> list[float]:
    return [(value - mean) / std for value, mean, std in zip(features, center, scale)]


def _local_density_rarity(standardized_rows: list[list[float]], *, neighbor_count: int) -> list[float]:
    if len(standardized_rows) <= 1:
        return [0.0 for _ in standardized_rows]
    neighbor_count = min(max(1, int(neighbor_count)), len(standardized_rows) - 1)
    scores = []
    for idx, features in enumerate(standardized_rows):
        distances = [
            _euclidean_distance(features, other)
            for other_idx, other in enumerate(standardized_rows)
            if other_idx != idx
        ]
        distances.sort()
        scores.append(float(sum(distances[:neighbor_count]) / neighbor_count))
    return scores


def _resolve_candidate_scores(
    rows: list[dict[str, Any]],
    standardized_rows: list[list[float]],
    cfg: R2VTrafficSelectorConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[float], list[float]]:
    if cfg.score_artifact_path is None:
        if cfg.candidate_model == "feature_center_distance_proxy":
            rarity_scores = [_euclidean_distance(features, [0.0] * len(features)) for features in standardized_rows]
        else:
            rarity_scores = _local_density_rarity(standardized_rows, neighbor_count=cfg.density_neighbors)
        support_scores = [1.0 / (1.0 + score) for score in rarity_scores]
        summary = {
            "kind": cfg.candidate_model,
            "backend": cfg.candidate_model,
            "artifact_path": None,
            "matched_count": len(rows),
        }
        return summary, [dict(summary) for _ in rows], rarity_scores, support_scores

    score_rows, summary = load_generative_score_artifact(
        cfg.score_artifact_path,
        id_key=cfg.score_artifact_id_key,
        backend=cfg.score_artifact_backend,
    )
    rarity_scores = []
    support_scores = []
    sources = []
    for row in rows:
        transition_id = str(row.get("transition_id"))
        score_row = score_rows.get(transition_id)
        if score_row is None:
            raise ValueError(f"missing score artifact row for transition_id {transition_id!r}")
        rarity_scores.append(score_row.rarity_score)
        support_scores.append(score_row.support_score)
        sources.append(_score_row_source(score_row))
    summary = dict(summary)
    summary["matched_count"] = len(rows)
    return summary, sources, rarity_scores, support_scores


def _score_row_source(score_row: GenerativeScoreRow) -> dict[str, Any]:
    source = dict(score_row.source)
    source["transition_id"] = score_row.transition_id
    if score_row.repaired_transition is not None:
        source["repaired_transition"] = dict(score_row.repaired_transition)
        source["repaired_transition_source"] = "score_artifact.repaired_transition"
    return source


def _euclidean_distance(features: list[float], other: list[float]) -> float:
    if not features:
        return 0.0
    squared = 0.0
    for value, right in zip(features, other):
        squared += (value - right) ** 2
    return float(math.sqrt(squared / len(features)))


def _value_components(row: dict[str, Any], cfg: R2VTrafficSelectorConfig) -> dict[str, float | None]:
    utility_delta = _utility(row.get("objectives_tp1_norm", {}), cfg) - _utility(row.get("objectives_t_norm", {}), cfg)
    env_reward_component = cfg.env_reward_weight * _safe_float(row.get("env_reward", 0.0))
    frozen_target_utility = _frozen_target_utility(row) if cfg.value_mode == "frozen_target_utility" else None
    base_value = utility_delta if cfg.value_mode == "objective_delta" else float(frozen_target_utility)
    return {
        "value_score": float(base_value + env_reward_component),
        "utility_delta": float(utility_delta),
        "env_reward_component": float(env_reward_component),
        "frozen_target_utility": frozen_target_utility,
    }


def _frozen_target_utility(row: dict[str, Any]) -> float:
    missing = object()
    value = row.get("frozen_target_utility", missing)
    metadata = row.get("metadata")
    if value is missing and isinstance(metadata, dict):
        value = metadata.get("frozen_target_utility", missing)
    if value is missing:
        raise ValueError(
            "value_mode='frozen_target_utility' requires a frozen_target_utility field "
            "on each transition row or in row['metadata']"
        )
    return _safe_float(value)


def _repair_story_status(
    row: dict[str, Any],
    computed_gates: dict[str, bool],
    cfg: R2VTrafficSelectorConfig,
) -> dict[str, Any]:
    if cfg.repair_story == "none":
        final_gates = dict(computed_gates)
        return {
            "gates": final_gates,
            "source_gates": None,
            "final_gates": final_gates,
            "repair_story_match": True,
            "gate_source": "computed",
        }

    transition_id = row.get("transition_id")
    source_raw = get_path_value(row, cfg.source_gates_key, default=_MISSING)
    final_raw = get_path_value(row, cfg.final_gates_key, default=_MISSING)
    source_missing = source_raw is _MISSING
    final_missing = final_raw is _MISSING
    if source_missing and final_missing and cfg.repair_metadata_policy == "metadata_or_proxy":
        return _proxy_repair_story_status(computed_gates, cfg)
    if source_missing:
        raise ValueError(
            f"repair_story={cfg.repair_story!r} requires source gates at "
            f"{cfg.source_gates_key!r} for transition_id {transition_id!r}"
        )
    if final_missing:
        raise ValueError(
            f"repair_story={cfg.repair_story!r} requires final gates at "
            f"{cfg.final_gates_key!r} for transition_id {transition_id!r}"
        )

    source_gates = _coerce_gate_map(source_raw, label=cfg.source_gates_key, transition_id=transition_id)
    final_gates = _coerce_gate_map(final_raw, label=cfg.final_gates_key, transition_id=transition_id)
    if cfg.repair_story == "not_val_to_val":
        story_match = source_gates["value"] is False and final_gates["value"] is True
    elif cfg.repair_story == "not_rare_to_val":
        story_match = source_gates["rare"] is False and final_gates["value"] is True
    else:
        raise ValueError(f"unsupported repair_story {cfg.repair_story!r}")

    return {
        "gates": final_gates,
        "source_gates": source_gates,
        "final_gates": final_gates,
        "repair_story_match": bool(story_match),
        "gate_source": "metadata_final_gates",
    }


def _proxy_repair_story_status(
    computed_gates: dict[str, bool],
    cfg: R2VTrafficSelectorConfig,
) -> dict[str, Any]:
    source_gates = dict(computed_gates)
    final_gates = dict(computed_gates)
    if cfg.repair_story == "not_val_to_val":
        source_gates["value"] = False
        story_match = final_gates["value"] is True
    elif cfg.repair_story == "not_rare_to_val":
        source_gates["rare"] = False
        story_match = final_gates["value"] is True
    else:
        raise ValueError(f"unsupported repair_story {cfg.repair_story!r}")
    return {
        "gates": final_gates,
        "source_gates": source_gates,
        "final_gates": final_gates,
        "repair_story_match": bool(story_match),
        "gate_source": "computed_proxy_repair_metadata",
    }


def _coerce_gate_map(value: Any, *, label: str, transition_id: Any) -> dict[str, bool]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object of boolean gates for transition_id {transition_id!r}")
    missing = [name for name in REQUIRED_GATE_KEYS if name not in value]
    if missing:
        raise ValueError(f"{label} missing gate keys for transition_id {transition_id!r}: {missing}")
    gates: dict[str, bool] = {}
    for name in REQUIRED_GATE_KEYS:
        raw_value = value[name]
        if not isinstance(raw_value, bool):
            raise ValueError(
                f"{label}.{name} must be a boolean for transition_id {transition_id!r}: {raw_value!r}"
            )
        gates[name] = raw_value
    return gates


def _active_gate_keys(gate_variant: str) -> tuple[str, ...]:
    if gate_variant not in SUPPORTED_GATE_VARIANTS:
        raise ValueError(f"gate_variant must be one of {sorted(SUPPORTED_GATE_VARIANTS)}")
    if gate_variant == "no_support":
        return ("value", "safety")
    if gate_variant == "no_ood":
        return ("support", "safety")
    if gate_variant == "no_dynamics":
        return ("value", "support")
    return ("value", "support", "safety")


def _traffic_artifact_backend(cfg: R2VTrafficSelectorConfig) -> str:
    if cfg.score_artifact_path is None:
        return str(cfg.candidate_model)
    return str(cfg.score_artifact_backend or "generative_score_artifact")


def _utility(objectives: dict[str, Any], cfg: R2VTrafficSelectorConfig) -> float:
    weights = cfg.resolved_utility_weights()
    return float(sum(weights[name] * _safe_float(objectives.get(name, 0.0)) for name in OBJECTIVE_NAMES))


def _admission_score_components(
    *,
    rarity: float,
    value: float,
    support: float,
    safety: float,
    thresholds: dict[str, float],
) -> dict[str, float]:
    rarity_margin = max(0.0, rarity - thresholds["rarity_min"])
    value_margin = max(0.0, value - thresholds["value_min"])
    support_margin = max(0.0, support - thresholds["support_min"])
    safety_margin = max(0.0, safety - thresholds["safety_min"])
    support_component = 0.25 * support_margin
    safety_component = 0.25 * safety_margin
    raw_score = rarity_margin + value_margin + support_component + safety_component
    return {
        "rarity_margin": float(rarity_margin),
        "value_margin": float(value_margin),
        "support_margin": float(support_margin),
        "safety_margin": float(safety_margin),
        "support_component": float(support_component),
        "safety_component": float(safety_component),
        "raw_admission_score": float(raw_score),
    }


def _admission_score(
    *,
    components: dict[str, float],
    admitted: bool,
) -> float:
    if not admitted:
        return 0.0
    return float(components["raw_admission_score"])


def _rank_and_weight(
    candidates: list[dict[str, Any]],
    cfg: R2VTrafficSelectorConfig,
    max_admission: float,
) -> list[dict[str, Any]]:
    ranked_indices = sorted(
        range(len(candidates)),
        key=lambda idx: (
            candidates[idx]["admitted"],
            candidates[idx]["admission_score"],
            candidates[idx]["value_score"],
            candidates[idx]["rarity_score"],
        ),
        reverse=True,
    )
    rank_by_idx = {idx: rank + 1 for rank, idx in enumerate(ranked_indices)}
    for idx, candidate in enumerate(candidates):
        candidate["candidate_rank"] = rank_by_idx[idx]
        if candidate["admitted"] and cfg.admitted_weight is not None:
            candidate["sample_weight"] = float(cfg.admitted_weight)
        elif candidate["admitted"] and max_admission > 0.0:
            bonus = cfg.admitted_weight_bonus * (candidate["admission_score"] / max_admission)
            candidate["sample_weight"] = min(cfg.max_weight, cfg.base_weight + bonus)
    return sorted(candidates, key=lambda row: row["candidate_rank"])


def _summary(
    candidates: list[dict[str, Any]],
    cfg: R2VTrafficSelectorConfig,
    thresholds: dict[str, float],
    *,
    score_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    admitted = [row for row in candidates if row.get("admitted")]
    gate_counts: dict[str, int] = {}
    gate_failure_counts: dict[str, int] = {}
    for name in REQUIRED_GATE_KEYS:
        gate_counts[name] = sum(1 for row in candidates if row.get("gates", {}).get(name))
        gate_failure_counts[name] = sum(1 for row in candidates if not row.get("gates", {}).get(name))
    weights = [float(row.get("sample_weight", 1.0)) for row in candidates]
    repair_story_match_count = sum(1 for row in candidates if row.get("repair_story_match"))
    gate_sources = {str(row.get("gate_source", "")) for row in candidates if row.get("gate_source")}
    if not gate_sources:
        gate_source = "metadata_final_gates" if cfg.repair_story != "none" else "computed"
    elif len(gate_sources) == 1:
        gate_source = next(iter(gate_sources))
    else:
        gate_source = "mixed"
    return {
        "schema_version": "r2v-tsc-candidate-summary-v1",
        "record_count": len(candidates),
        "candidate_count": len(candidates),
        "admitted_count": len(admitted),
        "admitted_rate": float(len(admitted) / len(candidates)) if candidates else 0.0,
        "candidate_model": cfg.candidate_model,
        "value_mode": cfg.value_mode,
        "repair_story": cfg.repair_story,
        "repair_metadata_policy": cfg.repair_metadata_policy,
        "gate_variant": cfg.gate_variant,
        "admission_mode": cfg.admission_mode,
        "active_gates": list(_active_gate_keys(cfg.gate_variant)),
        "repair_story_required": cfg.repair_story != "none",
        "repair_story_match_count": int(repair_story_match_count),
        "repair_story_match_rate": float(repair_story_match_count / len(candidates)) if candidates else 0.0,
        "gate_source": gate_source,
        "score_source": score_source or {
            "kind": "feature_density_proxy",
            "backend": cfg.candidate_model,
            "artifact_path": None,
            "matched_count": len(candidates),
        },
        "rarity_value_correlation": _correlation(
            [float(row.get("rarity_score", 0.0)) for row in candidates],
            [float(row.get("value_score", 0.0)) for row in candidates],
        ),
        "gate_counts": gate_counts,
        "gate_failure_counts": gate_failure_counts,
        "admission_score_component_means": _admission_score_component_means(candidates),
        "admitted_admission_score_component_means": _admission_score_component_means(admitted),
        "admission_score_mean": _mean([float(row.get("admission_score", 0.0)) for row in candidates]),
        "admitted_admission_score_mean": _mean([float(row.get("admission_score", 0.0)) for row in admitted]),
        "thresholds": dict(thresholds),
        "config": cfg.to_dict(),
        "weight_min": min(weights) if weights else cfg.base_weight,
        "weight_max": max(weights) if weights else cfg.base_weight,
        "death_condition_flags": _death_condition_flags(candidates, admitted),
    }


def _death_condition_flags(candidates: list[dict[str, Any]], admitted: list[dict[str, Any]]) -> dict[str, bool]:
    total = len(candidates)
    admitted_count = len(admitted)
    rare_count = sum(1 for row in candidates if row.get("gates", {}).get("rare"))
    rare_only_count = sum(
        1
        for row in candidates
        if row.get("gates", {}).get("rare") and not row.get("gates", {}).get("value")
    )
    safety_blocked_count = sum(
        1
        for row in candidates
        if row.get("gates", {}).get("rare")
        and row.get("gates", {}).get("value")
        and not row.get("gates", {}).get("safety")
    )
    return {
        "no_admitted_samples": total > 0 and admitted_count == 0,
        "too_many_admitted_samples": total > 0 and admitted_count / total > 0.5,
        "rare_value_collapse": rare_count > 0 and rare_only_count == 0,
        "safety_filters_most_high_value_rare": rare_count > 0 and safety_blocked_count / max(rare_count, 1) > 0.5,
    }


def _admission_score_component_means(candidates: list[dict[str, Any]]) -> dict[str, float]:
    names = (
        "rarity_margin",
        "value_margin",
        "support_margin",
        "safety_margin",
        "support_component",
        "safety_component",
        "raw_admission_score",
    )
    if not candidates:
        return {name: 0.0 for name in names}
    means: dict[str, float] = {}
    for name in names:
        means[name] = float(
            sum(float(row.get("admission_score_components", {}).get(name, 0.0)) for row in candidates)
            / len(candidates)
        )
    return means


def _empty_thresholds(cfg: R2VTrafficSelectorConfig) -> dict[str, float]:
    return {
        "rarity_min": 0.0,
        "value_min": 0.0,
        "support_min": 0.0,
        "safety_min": float(cfg.safety_min),
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_variance = sum((value - left_mean) ** 2 for value in left)
    right_variance = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_variance * right_variance)
    if denominator <= 1e-12:
        return 0.0
    return float(numerator / denominator)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = max(0.0, min(1.0, float(q))) * (len(ordered) - 1)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return float(ordered[lower])
    frac = pos - lower
    return float(ordered[lower] * (1.0 - frac) + ordered[upper] * frac)


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(result) or math.isinf(result):
        return 0.0
    return result
