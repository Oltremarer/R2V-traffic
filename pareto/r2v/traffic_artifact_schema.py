from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from pareto.r2v.artifact_validation import (
    EXPECTED_WEIGHTED_SCHEMA_VERSION,
    get_path_value,
    validate_weighted_transition_rows,
)


TRAFFIC_ARTIFACT_SCHEMA_VERSION = "r2v-traffic-artifact-v2"
TRAFFIC_ARTIFACT_SUMMARY_VERSION = "r2v-traffic-artifact-summary-v2"

SUPPORTED_R2V_FLAGS = {"on", "off"}
SUPPORTED_R2V_MODES = {"traffic"}
SUPPORTED_GENERATIVE_BACKENDS = {"diffusion"}
SUPPORTED_REPAIR_STORIES = {"not_rare_to_val", "not_val_to_val"}
SUPPORTED_REPAIR_METADATA_POLICIES = {"require_metadata", "metadata_or_proxy"}
SUPPORTED_GATE_VARIANTS = {"full", "no_support", "no_ood", "no_dynamics"}
SUPPORTED_ADMISSION_MODES = {"weights_only", "weights_plus_repaired"}
SUPPORTED_ROW_ROLES = {"source", "repaired", "repair_rejected"}
SUPPORTED_SAMPLING_MODES = {
    "off",
    "full_r2v",
    "admitted_only",
    "rare_only",
    "value_only",
    "random_same_count",
    "same_candidates_random_weights",
    "shuffled_value",
    "inverted_rarity",
}
TRAFFIC_GATE_KEYS = ("rare", "ood", "support", "dynamics")
LEGACY_TO_TRAFFIC_GATES = {
    "rare": "rare",
    "value": "ood",
    "support": "support",
    "safety": "dynamics",
}


@dataclass(frozen=True)
class R2VTrafficConfig:
    r2v: str = "off"
    r2v_mode: str = "traffic"
    generative_backend: str = "diffusion"
    repair_story: str = "not_rare_to_val"
    repair_metadata_policy: str = "metadata_or_proxy"
    gate_variant: str = "full"
    r2v_sampling_mode: str = "full_r2v"
    r2v_output_dir: str = "records/r2v_traffic"
    r2v_artifact_path: str = ""
    r2v_admitted_weight: float = 2.0
    r2v_repair_rejected_weight: float = 2.0
    r2v_admission_mode: str = "weights_only"
    rare_fraction: float = 0.2
    support_gate: bool | None = None
    ood_gate: bool | None = None
    dynamics_gate: bool | None = None

    def __post_init__(self) -> None:
        if self.r2v not in SUPPORTED_R2V_FLAGS:
            raise ValueError(f"r2v must be one of {sorted(SUPPORTED_R2V_FLAGS)}")
        if self.r2v_mode not in SUPPORTED_R2V_MODES:
            raise ValueError(f"r2v_mode must be one of {sorted(SUPPORTED_R2V_MODES)}")
        if self.generative_backend not in SUPPORTED_GENERATIVE_BACKENDS:
            raise ValueError(f"generative_backend must be one of {sorted(SUPPORTED_GENERATIVE_BACKENDS)}")
        if self.repair_story not in SUPPORTED_REPAIR_STORIES:
            raise ValueError(f"repair_story must be one of {sorted(SUPPORTED_REPAIR_STORIES)}")
        if self.repair_metadata_policy not in SUPPORTED_REPAIR_METADATA_POLICIES:
            raise ValueError(f"repair_metadata_policy must be one of {sorted(SUPPORTED_REPAIR_METADATA_POLICIES)}")
        if self.gate_variant not in SUPPORTED_GATE_VARIANTS:
            raise ValueError(f"gate_variant must be one of {sorted(SUPPORTED_GATE_VARIANTS)}")
        if self.r2v_sampling_mode not in SUPPORTED_SAMPLING_MODES:
            raise ValueError(f"r2v_sampling_mode must be one of {sorted(SUPPORTED_SAMPLING_MODES)}")
        if self.r2v_admission_mode not in SUPPORTED_ADMISSION_MODES:
            raise ValueError(f"r2v_admission_mode must be one of {sorted(SUPPORTED_ADMISSION_MODES)}")
        if self.r2v_enabled and self.r2v_sampling_mode == "off":
            raise ValueError("r2v_sampling_mode cannot be off when r2v is on")
        if self.r2v_admitted_weight <= 0.0:
            raise ValueError("r2v_admitted_weight must be positive")
        if self.r2v_repair_rejected_weight <= 0.0:
            raise ValueError("r2v_repair_rejected_weight must be positive")
        if not 0.0 < float(self.rare_fraction) <= 1.0:
            raise ValueError("rare_fraction must be in (0, 1]")

    @property
    def r2v_enabled(self) -> bool:
        return self.r2v == "on"

    @property
    def gate_config(self) -> dict[str, bool]:
        config = build_gate_variant_config(self.gate_variant)
        if self.support_gate is not None:
            config["support_gate"] = bool(self.support_gate)
        if self.ood_gate is not None:
            config["ood_gate"] = bool(self.ood_gate)
        if self.dynamics_gate is not None:
            config["dynamics_gate"] = bool(self.dynamics_gate)
        return config

    def to_cli_flags(self) -> list[str]:
        sampling_mode = self.r2v_sampling_mode if self.r2v_enabled else "off"
        flags = [
            "--r2v",
            self.r2v,
            "--r2v_mode",
            self.r2v_mode,
            "--generative_backend",
            self.generative_backend,
            "--repair_story",
            self.repair_story,
            "--repair_metadata_policy",
            self.repair_metadata_policy,
            "--gate_variant",
            self.gate_variant,
            "--r2v_sampling_mode",
            sampling_mode,
            "--r2v_output_dir",
            self.r2v_output_dir,
            "--r2v_admitted_weight",
            str(self.r2v_admitted_weight),
            "--r2v_repair_rejected_weight",
            str(self.r2v_repair_rejected_weight),
            "--r2v_admission_mode",
            self.r2v_admission_mode,
            "--rare_fraction",
            str(self.rare_fraction),
        ]
        if self.r2v_artifact_path:
            flags.extend(["--r2v_artifact_path", self.r2v_artifact_path])
        for name, enabled in self.gate_config.items():
            flags.extend([f"--{name}", "on" if enabled else "off"])
        return flags


def build_gate_variant_config(gate_variant: str) -> dict[str, bool]:
    if gate_variant not in SUPPORTED_GATE_VARIANTS:
        raise ValueError(f"gate_variant must be one of {sorted(SUPPORTED_GATE_VARIANTS)}")
    config = {
        "support_gate": True,
        "ood_gate": True,
        "dynamics_gate": True,
    }
    if gate_variant == "no_support":
        config["support_gate"] = False
    elif gate_variant == "no_ood":
        config["ood_gate"] = False
    elif gate_variant == "no_dynamics":
        config["dynamics_gate"] = False
    return config


def upgrade_weighted_row_to_v2_metadata(
    row: dict[str, Any],
    *,
    gate_variant: str = "full",
    generative_backend: str = "diffusion",
    admission_mode: str | None = None,
) -> dict[str, Any]:
    build_gate_variant_config(gate_variant)
    copied = dict(row)
    metadata = dict(copied.get("metadata") or {})
    if metadata.get("r2v_schema_version") != EXPECTED_WEIGHTED_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported r2v_schema_version for traffic upgrade: {metadata.get('r2v_schema_version')!r}"
        )
    legacy_gates = metadata.get("r2v_gates")
    if not isinstance(legacy_gates, dict):
        raise ValueError("missing r2v_gates for traffic upgrade")
    missing = [name for name in LEGACY_TO_TRAFFIC_GATES if name not in legacy_gates]
    if missing:
        raise ValueError(f"missing legacy R2V gate keys for traffic upgrade: {missing}")
    traffic_gates = {
        target: bool(legacy_gates[source])
        for source, target in LEGACY_TO_TRAFFIC_GATES.items()
    }
    resolved_admission_mode = str(admission_mode or metadata.get("r2v_admission_mode") or "weights_only")
    if resolved_admission_mode not in SUPPORTED_ADMISSION_MODES:
        raise ValueError(f"unsupported r2v_admission_mode for traffic upgrade: {resolved_admission_mode!r}")
    metadata.update(
        {
            "r2v_traffic_schema_version": TRAFFIC_ARTIFACT_SCHEMA_VERSION,
            "r2v_traffic_transition_id": copied.get("transition_id"),
            "r2v_traffic_sample_id": copied.get("sample_id"),
            "r2v_gate_variant": gate_variant,
            "r2v_generative_backend": generative_backend,
            "r2v_admission_mode": resolved_admission_mode,
            "r2v_row_role": str(metadata.get("r2v_row_role") or "source"),
            "r2v_repair_rejected": bool(metadata.get("r2v_repair_rejected", False)),
            "r2v_repaired_from_transition_id": metadata.get("r2v_repaired_from_transition_id"),
            "r2v_traffic_gates": traffic_gates,
            "r2v_gate_aliases": {
                "ood": "metadata.r2v_gates.value",
                "dynamics": "metadata.r2v_gates.safety",
            },
        }
    )
    copied["metadata"] = metadata
    return copied


def validate_r2v_traffic_artifact(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    validate_weighted_transition_rows(materialized)
    transition_seen: dict[str, int] = {}
    weights: list[float] = []
    admitted_count = 0
    gate_counts = {name: 0 for name in TRAFFIC_GATE_KEYS}
    variants: set[str] = set()
    backends: set[str] = set()
    admission_modes: set[str] = set()
    row_roles: set[str] = set()
    score_artifact_paths: set[str] = set()
    for idx, row in enumerate(materialized):
        transition_id = str(row.get("transition_id") or "")
        if not transition_id:
            raise ValueError(f"missing transition_id at row {idx}")
        if transition_id in transition_seen:
            raise ValueError(f"duplicate transition_id {transition_id!r} at rows {transition_seen[transition_id]} and {idx}")
        transition_seen[transition_id] = idx

        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"missing metadata at row {idx}")
        schema_version = metadata.get("r2v_traffic_schema_version")
        if schema_version != TRAFFIC_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported r2v_traffic_schema_version at row {idx}: {schema_version!r}")
        metadata_transition_id = str(metadata.get("r2v_traffic_transition_id") or "")
        if metadata_transition_id != transition_id:
            raise ValueError(
                "r2v_traffic_transition_id must match row transition_id "
                f"at row {idx}: {metadata_transition_id!r} != {transition_id!r}"
            )
        sample_id = str(row.get("sample_id") or "")
        metadata_sample_id = str(metadata.get("r2v_traffic_sample_id") or "")
        if metadata_sample_id != sample_id:
            raise ValueError(
                "r2v_traffic_sample_id must match row sample_id "
                f"at row {idx}: {metadata_sample_id!r} != {sample_id!r}"
            )
        gates = metadata.get("r2v_traffic_gates")
        if not isinstance(gates, dict):
            raise ValueError(f"missing r2v_traffic_gates at row {idx}")
        missing_gates = [name for name in TRAFFIC_GATE_KEYS if name not in gates]
        if missing_gates:
            raise ValueError(f"missing r2v_traffic_gates keys at row {idx}: {missing_gates}")
        for name in TRAFFIC_GATE_KEYS:
            if not isinstance(gates[name], bool):
                raise ValueError(f"r2v_traffic_gates.{name} must be boolean at row {idx}")
            if gates[name]:
                gate_counts[name] += 1
        weight = _positive_finite(get_path_value(row, "metadata.r2v_sample_weight"), label="r2v_sample_weight")
        weights.append(weight)
        admitted = bool(metadata.get("r2v_admitted", False))
        row_role = str(metadata.get("r2v_row_role") or "")
        if row_role not in SUPPORTED_ROW_ROLES:
            raise ValueError(f"unsupported r2v_row_role at row {idx}: {row_role!r}")
        repair_rejected = bool(metadata.get("r2v_repair_rejected", False))
        repaired_from = metadata.get("r2v_repaired_from_transition_id")
        if row_role == "source":
            if repair_rejected:
                raise ValueError(f"source rows must not be marked r2v_repair_rejected at row {idx}")
            if repaired_from is not None and str(repaired_from).strip() != "":
                raise ValueError(f"source rows must not set r2v_repaired_from_transition_id at row {idx}")
        elif row_role == "repaired":
            if repair_rejected:
                raise ValueError(f"repaired rows must not be marked r2v_repair_rejected at row {idx}")
            if not admitted:
                raise ValueError(f"repaired rows must be admitted at row {idx}")
            if repaired_from is None or str(repaired_from).strip() == "":
                raise ValueError(f"repaired row requires r2v_repaired_from_transition_id at row {idx}")
        elif row_role == "repair_rejected":
            if not repair_rejected:
                raise ValueError(f"repair_rejected rows must set r2v_repair_rejected=true at row {idx}")
            if admitted:
                raise ValueError(f"repair_rejected rows must not be admitted at row {idx}")
            if repaired_from is None or str(repaired_from).strip() == "":
                raise ValueError(f"repair_rejected row requires r2v_repaired_from_transition_id at row {idx}")
        row_roles.add(row_role)
        if admitted:
            admitted_count += 1
        variants.add(str(metadata.get("r2v_gate_variant", "")))
        backends.add(str(metadata.get("r2v_generative_backend", "")))
        admission_mode = str(metadata.get("r2v_admission_mode", ""))
        if admission_mode not in SUPPORTED_ADMISSION_MODES:
            raise ValueError(f"unsupported r2v_admission_mode at row {idx}: {admission_mode!r}")
        admission_modes.add(admission_mode)
        score_artifact_path = metadata.get("r2v_score_artifact_path")
        if score_artifact_path is not None and str(score_artifact_path).strip():
            score_artifact_paths.add(str(score_artifact_path))
    return {
        "schema_version": TRAFFIC_ARTIFACT_SUMMARY_VERSION,
        "row_count": len(materialized),
        "admitted_count": admitted_count,
        "gate_counts": gate_counts,
        "weight_min": min(weights) if weights else 0.0,
        "weight_max": max(weights) if weights else 0.0,
        "weight_mean": sum(weights) / len(weights) if weights else 0.0,
        "gate_variants": sorted(value for value in variants if value),
        "generative_backends": sorted(value for value in backends if value),
        "admission_modes": sorted(value for value in admission_modes if value),
        "row_roles": sorted(value for value in row_roles if value),
        "score_artifact_paths": sorted(score_artifact_paths),
    }


def _positive_finite(value: Any, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid {label}: {value!r}") from None
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"invalid {label}: {value!r}")
    return parsed
