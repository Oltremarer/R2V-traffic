from __future__ import annotations

import pytest

from pareto.r2v.traffic_artifact_schema import (
    R2VTrafficConfig,
    build_gate_variant_config,
    upgrade_weighted_row_to_v2_metadata,
    validate_r2v_traffic_artifact,
)


def _weighted_row(sample_id: str = "s0", transition_id: str = "t0") -> dict:
    return {
        "sample_id": sample_id,
        "transition_id": transition_id,
        "metadata": {
            "r2v_schema_version": "r2v-tsc-weighted-transition-v1",
            "r2v_sample_weight": 3.0,
            "r2v_admitted": True,
            "r2v_row_role": "source",
            "r2v_repair_rejected": False,
            "r2v_repaired_from_transition_id": None,
            "r2v_gates": {
                "rare": True,
                "value": True,
                "support": True,
                "safety": True,
            },
        },
    }


def test_gate_variant_config_maps_full_and_ablations_to_independent_gates():
    assert build_gate_variant_config("full") == {
        "support_gate": True,
        "ood_gate": True,
        "dynamics_gate": True,
    }
    assert build_gate_variant_config("no_support") == {
        "support_gate": False,
        "ood_gate": True,
        "dynamics_gate": True,
    }
    assert build_gate_variant_config("no_ood") == {
        "support_gate": True,
        "ood_gate": False,
        "dynamics_gate": True,
    }
    assert build_gate_variant_config("no_dynamics") == {
        "support_gate": True,
        "ood_gate": True,
        "dynamics_gate": False,
    }


def test_config_defaults_to_explicit_r2v_off_and_validates_required_modes():
    baseline = R2VTrafficConfig(r2v="off")
    assert baseline.r2v_enabled is False
    assert baseline.gate_config == {
        "support_gate": True,
        "ood_gate": True,
        "dynamics_gate": True,
    }

    enabled = R2VTrafficConfig(
        r2v="on",
        repair_story="not_rare_to_val",
        gate_variant="no_ood",
        r2v_sampling_mode="full_r2v",
    )
    assert enabled.r2v_enabled is True
    assert enabled.gate_config["ood_gate"] is False
    assert "--r2v_admission_mode" in enabled.to_cli_flags()
    assert enabled.to_cli_flags()[enabled.to_cli_flags().index("--r2v_admission_mode") + 1] == "weights_only"


def test_config_rejects_unknown_repair_story_gate_variant_and_backend():
    with pytest.raises(ValueError, match="repair_story"):
        R2VTrafficConfig(repair_story="rare_is_value")
    with pytest.raises(ValueError, match="gate_variant"):
        R2VTrafficConfig(gate_variant="no_value")
    with pytest.raises(ValueError, match="generative_backend"):
        R2VTrafficConfig(generative_backend="gan")
    with pytest.raises(ValueError, match="r2v_admission_mode"):
        R2VTrafficConfig(r2v_admission_mode="rare_is_value")


def test_upgrade_weighted_row_exposes_v2_support_ood_dynamics_view():
    upgraded = upgrade_weighted_row_to_v2_metadata(_weighted_row())

    assert upgraded["metadata"]["r2v_traffic_schema_version"] == "r2v-traffic-artifact-v2"
    assert upgraded["metadata"]["r2v_traffic_transition_id"] == "t0"
    assert upgraded["metadata"]["r2v_gate_variant"] == "full"
    assert upgraded["metadata"]["r2v_traffic_gates"] == {
        "rare": True,
        "ood": True,
        "support": True,
        "dynamics": True,
    }
    assert upgraded["metadata"]["r2v_admission_mode"] == "weights_only"


def test_upgrade_weighted_row_fails_closed_when_legacy_gate_is_missing():
    row = _weighted_row()
    row["metadata"]["r2v_gates"].pop("safety")

    with pytest.raises(ValueError, match="missing legacy R2V gate keys"):
        upgrade_weighted_row_to_v2_metadata(row)


def test_validate_r2v_traffic_artifact_checks_shapes_ids_and_weights():
    rows = [
        upgrade_weighted_row_to_v2_metadata(_weighted_row("s0", "t0")),
        upgrade_weighted_row_to_v2_metadata(_weighted_row("s1", "t1")),
    ]

    summary = validate_r2v_traffic_artifact(rows)

    assert summary["schema_version"] == "r2v-traffic-artifact-summary-v2"
    assert summary["row_count"] == 2
    assert summary["admitted_count"] == 2
    assert summary["gate_counts"]["dynamics"] == 2
    assert summary["weight_min"] == 3.0
    assert summary["row_roles"] == ["source"]


def test_validate_r2v_traffic_artifact_rejects_duplicate_transition_id():
    rows = [
        upgrade_weighted_row_to_v2_metadata(_weighted_row("s0", "dup")),
        upgrade_weighted_row_to_v2_metadata(_weighted_row("s1", "dup")),
    ]

    with pytest.raises(ValueError, match="duplicate transition_id"):
        validate_r2v_traffic_artifact(rows)


def test_validate_r2v_traffic_artifact_rejects_metadata_id_mismatch():
    row = upgrade_weighted_row_to_v2_metadata(_weighted_row("s0", "t0"))
    row["metadata"]["r2v_traffic_transition_id"] = "other_transition"
    row["metadata"]["r2v_traffic_sample_id"] = "other_sample"

    with pytest.raises(ValueError, match="r2v_traffic_transition_id"):
        validate_r2v_traffic_artifact([row])


def test_validate_r2v_traffic_artifact_rejects_missing_dynamics_gate():
    row = upgrade_weighted_row_to_v2_metadata(_weighted_row())
    row["metadata"]["r2v_traffic_gates"].pop("dynamics")

    with pytest.raises(ValueError, match="missing r2v_traffic_gates"):
        validate_r2v_traffic_artifact([row])


def test_validate_r2v_traffic_artifact_rejects_unknown_row_role():
    row = upgrade_weighted_row_to_v2_metadata(_weighted_row())
    row["metadata"]["r2v_row_role"] = "generator_decides_value"

    with pytest.raises(ValueError, match="unsupported r2v_row_role"):
        validate_r2v_traffic_artifact([row])


def test_validate_r2v_traffic_artifact_rejects_repair_rejected_as_admitted():
    row = upgrade_weighted_row_to_v2_metadata(_weighted_row())
    row["metadata"]["r2v_row_role"] = "repair_rejected"
    row["metadata"]["r2v_repair_rejected"] = True
    row["metadata"]["r2v_repaired_from_transition_id"] = "source_t0"

    with pytest.raises(ValueError, match="repair_rejected rows must not be admitted"):
        validate_r2v_traffic_artifact([row])


def test_validate_r2v_traffic_artifact_requires_repaired_source_link():
    row = upgrade_weighted_row_to_v2_metadata(_weighted_row())
    row["metadata"]["r2v_row_role"] = "repaired"

    with pytest.raises(ValueError, match="requires r2v_repaired_from_transition_id"):
        validate_r2v_traffic_artifact([row])
