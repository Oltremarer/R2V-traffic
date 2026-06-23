from __future__ import annotations

import copy

import pytest

from pareto.rl.paper_final_experiment_manifest import REQUIRED_PAPER_BASELINES, load_paper_final_manifest
from pareto.eval.paper_representation_formal_pass_guard import (
    REQUIRED_REPRESENTATION_DIAGNOSTIC_LIMITATION_APPROVAL,
    build_representation_scope_limitation,
)
from pareto.eval.paper_representation_artifact_sources import (
    REQUIRED_REPRESENTATION_ARTIFACT_CITIES,
    REQUIRED_REPRESENTATION_METRIC_KEYS,
    REQUIRED_REPRESENTATION_MODEL_FAMILIES,
)
from pareto.rl.paper_c2t_scalar_guard import (
    REQUIRED_C2T_SCALAR_EXCLUSION_APPROVAL,
    build_c2t_scalar_readiness,
)
from pareto.rl.paper_final_execution_request import (
    assert_can_prepare_final_execution_request,
    build_dynamic_paper_final_command_previews,
    build_paper_final_command_previews,
    validate_scope_limited_readiness,
)
from pareto.rl.paper_weighted_mapping_policy import (
    REQUIRED_WEIGHTED_MAPPING_APPROVAL,
    build_weighted_mapping_policy,
)


def _pass_audit() -> dict:
    return {
        "audit_status": "PASS",
        "hashes_match": True,
        "targeted_pytest_passed": True,
        "full_pytest_passed": True,
        "root_empty_check_passed": True,
        "no_result_value_reading": True,
        "preference_sweep_guard_implemented": True,
        "representation_diagnostics_implemented": True,
        "action_diagnostics_implemented": True,
        "result_value_policy_declared": True,
    }


def _pass_representation_audit() -> dict:
    return {
        "packet_type": "paper_representation_formal_pass_audit",
        "status": "pass",
        "blockers": [],
        "rows": [
            {
                "city": "jinan",
                "status": "pass",
                "packet_hashes": ["a" * 64],
                "packet_paths": ["docs/pro_reviews/pareto_ppo_final_representation_packet_generation_execution_2026-06-03/jinan/vectorq_ppo/representation_formal_gate_packet.json"],
                "formal_representation_pass": True,
                "claim_modes": ["formal_pass"],
                "blockers": [],
            },
            {
                "city": "hangzhou",
                "status": "pass",
                "packet_hashes": ["b" * 64],
                "packet_paths": ["docs/pro_reviews/pareto_ppo_final_representation_packet_generation_execution_2026-06-03/hangzhou/vectorq_ppo/representation_formal_gate_packet.json"],
                "formal_representation_pass": True,
                "claim_modes": ["formal_pass"],
                "blockers": [],
            },
            {
                "city": "newyork_28x7",
                "status": "pass",
                "packet_hashes": ["c" * 64],
                "packet_paths": ["docs/pro_reviews/pareto_ppo_final_representation_packet_generation_execution_2026-06-03/newyork_28x7/vectorq_ppo/representation_formal_gate_packet.json"],
                "formal_representation_pass": True,
                "claim_modes": ["formal_pass"],
                "blockers": [],
            },
        ],
        "executes_now": False,
        "reads_final_traffic_result_values": False,
        "paper_result_claim": False,
    }


def _blocked_representation_audit() -> dict:
    audit = _pass_representation_audit()
    audit["status"] = "blocked"
    audit["blockers"] = ["representation formal gate jinan: formal_representation_pass=false"]
    audit["rows"][0] = dict(audit["rows"][0], status="blocked")
    return audit


def _ready_manifest() -> dict:
    manifest = load_paper_final_manifest("configs/formal/paper_final_experiment_manifest_2026-06-02.json")
    manifest = copy.deepcopy(manifest)
    manifest["baselines"] = [
        {
            "name": name,
            "status": "implemented",
            "command_family": f"{name.lower()}_command",
            "city_support": ["jinan", "hangzhou", "newyork_28x7"],
        }
        for name in REQUIRED_PAPER_BASELINES
    ]
    for row in manifest["metric_families"].values():
        row["status"] = "implemented"
        row.pop("blocker", None)
    return manifest


def _scope_limitations() -> dict:
    return validate_scope_limited_readiness(
        {
            "packet_type": "paper_final_scope_limited_readiness",
            "representation": build_representation_scope_limitation(
                approval_phrase=REQUIRED_REPRESENTATION_DIAGNOSTIC_LIMITATION_APPROVAL,
                paper_claim_limitation="Representation formal gate remains diagnostics-only and is not claimed as formal evidence.",
            ),
            "c2t_scalar": build_c2t_scalar_readiness(
                exclusion_approval_phrase=REQUIRED_C2T_SCALAR_EXCLUSION_APPROVAL,
                paper_claim_limitation="C2T-scalar is excluded from final baseline comparisons.",
            ),
            "weighted_rl": build_weighted_mapping_policy(approval_phrase=REQUIRED_WEIGHTED_MAPPING_APPROVAL),
            "executes_training_now": False,
            "reads_final_traffic_result_values": False,
            "paper_result_claim": False,
        }
    )


def _complete_learned_artifact_inventory() -> dict:
    rows = []
    for baseline, family in (("Cond-Scalar-RL", "cond_scalar"), ("VectorQ-PPO", "pareto_quality")):
        for city in ("jinan", "hangzhou", "newyork_28x7"):
            rows.append(
                {
                    "baseline": baseline,
                    "city": city,
                    "status": "implemented_guarded_preview",
                    "model_path": f"model_weights/{family}/{city}/paper_final/run/model.pt",
                    "model_hash": "a" * 64,
                    "objective_normalizer_path": f"model_weights/{family}/{city}/paper_final/run/objective_normalizer.json",
                    "objective_normalizer_hash": "b" * 64,
                    "executes_training_now": False,
                }
            )
    return {
        "packet_type": "paper_learned_artifact_inventory",
        "coverage_status": "complete",
        "rows": rows,
        "executes_training_now": False,
    }


def _complete_representation_artifact_audit() -> dict:
    return {
        "packet_type": "paper_representation_artifact_sources",
        "coverage_status": "complete",
        "rows": [
            {
                "city": city,
                "model_family": model_family,
                "status": "implemented_guarded_preview",
                "packet_path": f"docs/pro_reviews/{city}/{model_family}/representation_formal_gate_packet.json",
                "packet_hash": "c" * 64,
                "metrics": list(REQUIRED_REPRESENTATION_METRIC_KEYS),
                "packet_keys": {
                    "obj_acc": ["obj_acc_mean"],
                    "pref_acc": ["pref_acc"],
                    "rev_acc": ["rev_acc"],
                    "dpr": ["dpr_head", "dpr_utility"],
                },
                "executes_training_now": False,
                "reads_final_traffic_result_values": False,
                "paper_result_claim": False,
            }
            for city in REQUIRED_REPRESENTATION_ARTIFACT_CITIES
            for model_family in REQUIRED_REPRESENTATION_MODEL_FAMILIES
        ],
        "executes_training_now": False,
    }


def test_execution_request_refuses_manifest_with_known_blockers():
    manifest = load_paper_final_manifest("configs/formal/paper_final_experiment_manifest_2026-06-02.json")

    with pytest.raises(ValueError, match="final execution blocked"):
        assert_can_prepare_final_execution_request(manifest, _pass_audit(), _pass_representation_audit())


def test_execution_request_requires_successful_preflight_audit():
    audit = _pass_audit()
    audit["full_pytest_passed"] = False

    with pytest.raises(ValueError, match="preflight audit"):
        assert_can_prepare_final_execution_request(_ready_manifest(), audit, _pass_representation_audit())


def test_execution_request_requires_representation_formal_pass_audit():
    with pytest.raises(ValueError, match="representation formal-pass audit"):
        assert_can_prepare_final_execution_request(_ready_manifest(), _pass_audit())


def test_execution_request_blocks_failing_representation_formal_pass():
    with pytest.raises(ValueError, match="representation formal gate"):
        assert_can_prepare_final_execution_request(_ready_manifest(), _pass_audit(), _blocked_representation_audit())


def test_execution_request_allows_blocked_representation_only_with_exact_scope_limitation():
    validated = assert_can_prepare_final_execution_request(
        _ready_manifest(),
        _pass_audit(),
        _blocked_representation_audit(),
        scope_limitations=_scope_limitations(),
    )

    assert validated["scope_limitations"]["representation"]["status"] == "diagnostic_limitation_by_reviewer"


def test_scope_limitation_still_requires_valid_representation_audit():
    with pytest.raises(ValueError, match="representation formal-pass audit"):
        assert_can_prepare_final_execution_request(
            _ready_manifest(),
            _pass_audit(),
            None,
            scope_limitations=_scope_limitations(),
        )

    forged = {
        "packet_type": "paper_representation_formal_pass_audit",
        "status": "blocked",
        "blockers": [],
        "rows": [],
        "executes_now": False,
        "reads_final_traffic_result_values": False,
        "paper_result_claim": False,
    }
    with pytest.raises(ValueError, match="missing required city"):
        assert_can_prepare_final_execution_request(
            _ready_manifest(),
            _pass_audit(),
            forged,
            scope_limitations=_scope_limitations(),
        )


def test_execution_request_rejects_forged_minimal_representation_audit():
    forged = {
        "packet_type": "paper_representation_formal_pass_audit",
        "status": "pass",
        "blockers": [],
        "rows": [],
        "executes_now": False,
        "reads_final_traffic_result_values": False,
        "paper_result_claim": False,
    }

    with pytest.raises(ValueError, match="missing required city"):
        assert_can_prepare_final_execution_request(_ready_manifest(), _pass_audit(), forged)


def test_execution_request_rejects_representation_audit_outside_output_root():
    audit = _pass_representation_audit()
    audit["rows"][0]["packet_paths"] = ["docs/pro_reviews/paper_final_packets/jinan/vectorq_ppo/representation_formal_gate_packet.json"]

    with pytest.raises(ValueError, match="outside paper-final representation packet output root"):
        assert_can_prepare_final_execution_request(_ready_manifest(), _pass_audit(), audit)


def test_command_previews_include_city_seed_method_root_and_no_execution():
    previews = build_paper_final_command_previews(_ready_manifest(), _pass_audit(), _pass_representation_audit())

    assert previews
    first = previews[0]
    assert first["executes_training_now"] is False
    assert first["approval_phrase_required"] is True
    assert "records/paper_final/train_20260602_v1" in first["out_dir"]
    assert {"city", "traffic_file", "method", "seed", "command"} <= set(first)


def test_command_previews_expand_weighted_rl_true_family_preferences():
    previews = build_paper_final_command_previews(_ready_manifest(), _pass_audit(), _pass_representation_audit())
    weighted_rows = [row for row in previews if row["method"] == "Weighted-RL"]

    assert len(weighted_rows) == 75
    assert {row["fixed_preference_template"] for row in weighted_rows} == {
        "balanced",
        "efficiency_focused",
        "fairness_focused",
        "safety_focused",
        "stability_focused",
    }
    assert all(row["policy_conditioned_on_w"] is False for row in weighted_rows)
    assert all(row["critic_conditioned_on_w"] is False for row in weighted_rows)
    assert all(row["executes_training_now"] is False for row in weighted_rows)
    assert len({(row["city"], row["seed"], row["preference_template"], row["out_dir"]) for row in weighted_rows}) == len(
        weighted_rows
    )
    assert len({row["out_dir"] for row in weighted_rows}) == len(weighted_rows)


def test_scope_limited_command_previews_surface_exclusions_and_mapping():
    manifest = _ready_manifest()
    for row in manifest["baselines"]:
        if row["name"] == "C2T-scalar":
            row["status"] = "missing_blocker"
            row["blocker"] = "C2T-scalar command or reviewer-approved exclusion is missing"
        if row["name"] == "Weighted-RL":
            row["status"] = "missing_blocker"
            row["blocker"] = "Weighted-RL -> weighted_proxy exact reviewer approval phrase is missing"
    manifest["metric_families"]["representation"]["status"] = "missing_blocker"
    manifest["metric_families"]["representation"]["blocker"] = "representation formal gate blocked"

    previews = build_paper_final_command_previews(
        manifest,
        _pass_audit(),
        _blocked_representation_audit(),
        scope_limitations=_scope_limitations(),
    )

    c2t_rows = [row for row in previews if row["method"] == "C2T-scalar"]
    weighted_rows = [row for row in previews if row["method"] == "Weighted-RL"]

    assert c2t_rows
    assert all(row["scope_limitation"] == "excluded_by_reviewer" for row in c2t_rows)
    assert all(row["command"] is None for row in c2t_rows)
    assert weighted_rows
    assert all(row["weighted_mapping_approved"] is True for row in weighted_rows)
    assert all(row["method_id"] == "weighted_proxy" for row in weighted_rows)


def test_dynamic_command_previews_clear_artifact_backed_manifest_blockers_under_scope_limitations():
    previews = build_dynamic_paper_final_command_previews(
        load_paper_final_manifest("configs/formal/paper_final_experiment_manifest_2026-06-02.json"),
        _pass_audit(),
        _blocked_representation_audit(),
        learned_artifact_inventory=_complete_learned_artifact_inventory(),
        representation_artifact_audit=_complete_representation_artifact_audit(),
        scope_limitations=_scope_limitations(),
    )

    assert len(previews) == 225
    assert sum(1 for row in previews if row["method"] == "C2T-scalar" and row["command"] is None) == 15
    assert sum(1 for row in previews if row["method"] == "Cond-Scalar-RL") == 15
    assert sum(1 for row in previews if row["method"] == "VectorQ-PPO") == 15


def test_dynamic_command_previews_keep_learned_blocker_when_inventory_is_incomplete():
    incomplete_inventory = _complete_learned_artifact_inventory()
    incomplete_inventory["coverage_status"] = "missing_blocker"
    incomplete_inventory["rows"][0] = dict(
        incomplete_inventory["rows"][0],
        status="missing_blocker",
        blocker="model hash missing",
    )

    with pytest.raises(ValueError, match="Cond-Scalar-RL"):
        build_dynamic_paper_final_command_previews(
            load_paper_final_manifest("configs/formal/paper_final_experiment_manifest_2026-06-02.json"),
            _pass_audit(),
            _blocked_representation_audit(),
            learned_artifact_inventory=incomplete_inventory,
            representation_artifact_audit=_complete_representation_artifact_audit(),
            scope_limitations=_scope_limitations(),
        )
