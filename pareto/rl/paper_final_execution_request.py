from __future__ import annotations

import copy
from typing import Any

from pareto.eval.paper_representation_formal_pass_guard import (
    representation_formal_pass_blockers,
    validate_representation_scope_limitation,
)
from pareto.eval.paper_metric_sources import build_metric_source_policy
from pareto.rl.paper_baseline_registry import BASELINE_REGISTRY, registry_with_learned_artifact_inventory
from pareto.rl.paper_c2t_scalar_guard import validate_c2t_scalar_readiness
from pareto.rl.paper_final_experiment_manifest import (
    REQUIRED_PAPER_BASELINES,
    REQUIRED_PREFERENCE_TEMPLATES,
    manifest_with_baseline_registry,
    manifest_with_metric_source_policy,
    paper_final_blockers,
    validate_paper_final_manifest,
)
from pareto.rl.paper_final_root_policy import build_paper_final_roots
from pareto.rl.paper_weighted_mapping_policy import validate_weighted_mapping_policy


REQUIRED_PASS_AUDIT_FLAGS = (
    "hashes_match",
    "targeted_pytest_passed",
    "full_pytest_passed",
    "root_empty_check_passed",
    "no_result_value_reading",
    "preference_sweep_guard_implemented",
    "representation_diagnostics_implemented",
    "action_diagnostics_implemented",
    "result_value_policy_declared",
)


def _assert_preflight_audit_passed(preflight_audit: dict[str, Any]) -> None:
    if preflight_audit.get("audit_status") != "PASS":
        raise ValueError("preflight audit must have audit_status=PASS")
    failed = [flag for flag in REQUIRED_PASS_AUDIT_FLAGS if preflight_audit.get(flag) is not True]
    if failed:
        raise ValueError(f"preflight audit missing passing flags: {failed}")


def _assert_representation_formal_pass_audit_passed(representation_formal_pass_audit: dict[str, Any] | None) -> None:
    if representation_formal_pass_audit is None:
        raise ValueError("representation formal-pass audit is required")
    blockers = representation_formal_pass_blockers(representation_formal_pass_audit)
    if representation_formal_pass_audit.get("status") != "pass" or blockers:
        raise ValueError(f"representation formal gate blocked: {blockers}")


def validate_scope_limited_readiness(packet: dict[str, Any]) -> dict[str, Any]:
    if packet.get("packet_type") != "paper_final_scope_limited_readiness":
        raise ValueError("scope-limited readiness packet_type must be paper_final_scope_limited_readiness")
    if packet.get("executes_training_now") is not False:
        raise ValueError("scope-limited readiness must be non-executing")
    if packet.get("reads_final_traffic_result_values") is not False:
        raise ValueError("scope-limited readiness must not read final traffic result values")
    if packet.get("paper_result_claim") is not False:
        raise ValueError("scope-limited readiness must not make paper claims")
    representation = validate_representation_scope_limitation(packet.get("representation") or {})
    c2t_scalar = validate_c2t_scalar_readiness(packet.get("c2t_scalar") or {})
    weighted_rl = validate_weighted_mapping_policy(packet.get("weighted_rl") or {})
    if representation.get("status") != "diagnostic_limitation_by_reviewer":
        raise ValueError("scope-limited readiness requires approved representation diagnostic limitation")
    if c2t_scalar.get("status") != "excluded_by_reviewer":
        raise ValueError("scope-limited readiness requires approved C2T-scalar exclusion")
    if weighted_rl.get("status") != "implemented_guarded_preview":
        raise ValueError("scope-limited readiness requires approved Weighted-RL mapping policy")
    validated = dict(packet)
    validated["representation"] = representation
    validated["c2t_scalar"] = c2t_scalar
    validated["weighted_rl"] = weighted_rl
    return validated


def _accepted_scope_limitations(scope_limitations: dict[str, Any] | None) -> dict[str, Any] | None:
    if scope_limitations is None:
        return None
    return validate_scope_limited_readiness(scope_limitations)


def _assert_manifest_ready_or_limited(
    manifest: dict[str, Any],
    scope_limitations: dict[str, Any] | None,
) -> dict[str, Any]:
    validated = validate_paper_final_manifest(manifest)
    blockers = paper_final_blockers(validated)
    scope = _accepted_scope_limitations(scope_limitations)
    if blockers:
        if scope is None:
            raise ValueError(f"final execution blocked: {blockers}")
        remaining = []
        for blocker in blockers:
            if blocker.startswith("baseline C2T-scalar:"):
                continue
            if blocker.startswith("baseline Weighted-RL:"):
                continue
            if blocker.startswith("metric family representation:"):
                continue
            remaining.append(blocker)
        if remaining:
            raise ValueError(f"final execution blocked: {remaining}")
    updated = copy.deepcopy(validated)
    if scope is not None:
        updated["scope_limitations"] = scope
    return updated


def _assert_representation_formal_pass_or_limited(
    representation_formal_pass_audit: dict[str, Any] | None,
    scope_limitations: dict[str, Any] | None,
) -> None:
    try:
        _assert_representation_formal_pass_audit_passed(representation_formal_pass_audit)
    except ValueError:
        if representation_formal_pass_audit is None:
            raise
        if scope_limitations is None:
            raise
        representation = scope_limitations.get("representation") or {}
        if representation.get("status") != "diagnostic_limitation_by_reviewer":
            raise
        blockers = representation_formal_pass_blockers(representation_formal_pass_audit)
        non_limitable = [
            blocker
            for blocker in blockers
            if (
                "wrong packet_type" in blocker
                or "rows must be a list" in blocker
                or "missing required city" in blocker
                or "must contain one 64-char packet hash" in blocker
                or "missing packet path" in blocker
                or "outside paper-final representation packet output root" in blocker
            )
        ]
        if non_limitable:
            raise ValueError(f"representation formal-pass audit invalid for scope limitation: {non_limitable}")


def assert_can_prepare_final_execution_request(
    manifest: dict[str, Any],
    preflight_audit: dict[str, Any],
    representation_formal_pass_audit: dict[str, Any] | None = None,
    *,
    scope_limitations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validated = _assert_manifest_ready_or_limited(manifest, scope_limitations)
    _assert_preflight_audit_passed(preflight_audit)
    _assert_representation_formal_pass_or_limited(
        representation_formal_pass_audit,
        validated.get("scope_limitations"),
    )
    return validated


def manifest_with_dynamic_execution_readiness(
    manifest: dict[str, Any],
    *,
    learned_artifact_inventory: dict[str, Any] | None = None,
    representation_artifact_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = validate_paper_final_manifest(manifest)
    if learned_artifact_inventory is not None:
        registry = registry_with_learned_artifact_inventory(BASELINE_REGISTRY, learned_artifact_inventory)
        updated = manifest_with_baseline_registry(updated, registry)
    if representation_artifact_audit is not None:
        metric_policy = build_metric_source_policy(representation_artifact_audit=representation_artifact_audit)
        updated = manifest_with_metric_source_policy(updated, metric_policy)
    return validate_paper_final_manifest(updated)


def build_dynamic_paper_final_command_previews(
    manifest: dict[str, Any],
    preflight_audit: dict[str, Any],
    representation_formal_pass_audit: dict[str, Any] | None = None,
    *,
    learned_artifact_inventory: dict[str, Any] | None = None,
    representation_artifact_audit: dict[str, Any] | None = None,
    scope_limitations: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return build_paper_final_command_previews(
        manifest_with_dynamic_execution_readiness(
            manifest,
            learned_artifact_inventory=learned_artifact_inventory,
            representation_artifact_audit=representation_artifact_audit,
        ),
        preflight_audit,
        representation_formal_pass_audit,
        scope_limitations=scope_limitations,
    )


def _baseline_by_name(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["name"]): dict(row) for row in manifest.get("baselines") or []}


def _preference_rows_for_baseline(baseline_name: str) -> list[tuple[str, list[float]]]:
    if baseline_name == "Weighted-RL":
        return [(name, list(weights)) for name, weights in REQUIRED_PREFERENCE_TEMPLATES.items()]
    return [("balanced", list(REQUIRED_PREFERENCE_TEMPLATES["balanced"]))]


def build_paper_final_command_previews(
    manifest: dict[str, Any],
    preflight_audit: dict[str, Any],
    representation_formal_pass_audit: dict[str, Any] | None = None,
    *,
    scope_limitations: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    validated = assert_can_prepare_final_execution_request(
        manifest,
        preflight_audit,
        representation_formal_pass_audit,
        scope_limitations=scope_limitations,
    )
    baseline_rows = _baseline_by_name(validated)
    accepted_limitations = validated.get("scope_limitations") or {}
    previews: list[dict[str, Any]] = []
    for city_row in validated["cities"]:
        city = str(city_row["scenario"])
        traffic_file = str(city_row["traffic_file"])
        for baseline_name in REQUIRED_PAPER_BASELINES:
            baseline = baseline_rows[baseline_name]
            for seed in validated["main_seed_ids"]:
                for preference_name, preference_weights in _preference_rows_for_baseline(baseline_name):
                    roots = build_paper_final_roots(
                        city=city,
                        traffic_file=traffic_file,
                        method=baseline_name,
                        seed=int(seed),
                        preference_id=preference_name,
                    )
                    if (
                        baseline_name == "C2T-scalar"
                        and (accepted_limitations.get("c2t_scalar") or {}).get("status") == "excluded_by_reviewer"
                    ):
                        previews.append(
                            {
                                "city": city,
                                "traffic_file": traffic_file,
                                "method": baseline_name,
                                "seed": int(seed),
                                "preference_template": preference_name,
                                "preference_weights": preference_weights,
                                "out_dir": str(roots.train),
                                "command": None,
                                "approval_phrase_required": True,
                                "scope_limitation": "excluded_by_reviewer",
                                "paper_claim_limitation": accepted_limitations["c2t_scalar"]["paper_claim_limitation"],
                                "executes_training_now": False,
                                "ranking_generated": False,
                                "paper_table_generated": False,
                                "paper_result_text_generated": False,
                            }
                        )
                        continue
                    command = (
                        f"{baseline['command_family']} --city {city} --traffic_file {traffic_file} "
                        f"--method {baseline_name} --seed_id {int(seed)} "
                        f"--fixed_preference_template {preference_name} --out_dir {roots.train}"
                    )
                    row = {
                        "city": city,
                        "traffic_file": traffic_file,
                        "method": baseline_name,
                        "seed": int(seed),
                        "preference_template": preference_name,
                        "preference_weights": preference_weights,
                        "out_dir": str(roots.train),
                        "command": command,
                        "approval_phrase_required": True,
                        "executes_training_now": False,
                        "ranking_generated": False,
                        "paper_table_generated": False,
                        "paper_result_text_generated": False,
                    }
                    if baseline_name == "Weighted-RL":
                        row.update(
                            {
                                "fixed_preference_template": preference_name,
                                "controller_scope": "single_fixed_preference",
                                "policy_conditioned_on_w": False,
                                "critic_conditioned_on_w": False,
                            }
                        )
                        if (accepted_limitations.get("weighted_rl") or {}).get("status") == "implemented_guarded_preview":
                            row.update(
                                {
                                    "method_id": accepted_limitations["weighted_rl"]["method_id"],
                                    "weighted_mapping_approved": True,
                                    "mapping_candidate": accepted_limitations["weighted_rl"]["mapping_candidate"],
                                    "paper_claim_limitation": accepted_limitations["weighted_rl"]["paper_claim_limitation"],
                                }
                            )
                    if accepted_limitations:
                        row["scope_limitations_declared"] = True
                    previews.append(row)
    return previews
