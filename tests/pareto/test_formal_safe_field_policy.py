from __future__ import annotations

import json
from pathlib import Path

import pytest

from pareto.rl.formal_safe_field_policy import (
    FORMAL_SAFE_FIELD_POLICY_APPROVAL_PHRASE,
    generate_safe_field_policy,
)
from pareto.rl.formal_safe_field_policy_validator import validate_safe_field_policy_packet


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _inventory(tmp_path: Path) -> Path:
    path = tmp_path / "formal_jinan_result_field_inventory.json"
    _write_json(
        path,
        {
            "report_status": "FORMAL_JINAN_RESULT_FIELD_INVENTORY_PASS",
            "scope": "field_inventory_only_no_metric_values",
            "field_categories": {
                "total_reward": "traffic_like_forbidden",
                "potential_reward": "traffic_like_forbidden",
                "weighted_proxy_reward": "traffic_like_forbidden",
                "scalar_quality_score_t": "traffic_like_forbidden",
                "scalar_quality_score_tp1": "traffic_like_forbidden",
                "proxy_objectives_norm_tp1": "unknown_requires_review",
                "env_reward": "traffic_like_forbidden",
                "env_reward_source": "traffic_like_forbidden",
                "env_reward_info_source": "traffic_like_forbidden",
                "reward_adapter_semantics": "traffic_like_forbidden",
                "approx_kl": "training_stability",
                "grad_norm": "training_stability",
                "policy_loss": "training_stability",
                "action": "unknown_requires_review",
                "w": "unknown_requires_review",
                "preference_name": "unknown_requires_review",
                "method": "diagnostic",
            },
            "files": {},
        },
    )
    return path


def test_safe_field_policy_assigns_required_statuses(tmp_path: Path):
    out_dir = tmp_path / "out"
    report = generate_safe_field_policy(
        inventory_json=_inventory(tmp_path),
        out_dir=out_dir,
        approval_phrase=FORMAL_SAFE_FIELD_POLICY_APPROVAL_PHRASE,
    )

    assert report["report_status"] == "FORMAL_JINAN_SAFE_FIELD_POLICY_PASS"
    assert sorted(path.name for path in out_dir.iterdir()) == [
        "formal_jinan_safe_field_policy.json",
        "formal_jinan_safe_field_policy.md",
    ]
    policy = json.loads((out_dir / "formal_jinan_safe_field_policy.json").read_text(encoding="utf-8"))
    fields = policy["fields"]
    assert fields["total_reward"]["status"] == "forbidden_proxy_result_metric"
    assert fields["potential_reward"]["status"] == "forbidden_proxy_result_metric"
    assert fields["weighted_proxy_reward"]["status"] == "forbidden_proxy_result_metric"
    assert fields["scalar_quality_score_t"]["status"] == "forbidden_proxy_result_metric"
    assert fields["scalar_quality_score_tp1"]["status"] == "forbidden_proxy_result_metric"
    assert fields["proxy_objectives_norm_tp1"]["status"] == "forbidden_proxy_result_metric"
    assert fields["env_reward"]["status"] == "forbidden_proxy_result_metric"
    assert fields["env_reward_source"]["status"] == "allowed_guard_metadata"
    assert fields["env_reward_info_source"]["status"] == "allowed_guard_metadata"
    assert fields["reward_adapter_semantics"]["status"] == "allowed_guard_metadata"
    assert fields["approx_kl"]["status"] == "allowed_training_stability_sanity_only"
    assert fields["grad_norm"]["status"] == "allowed_training_stability_sanity_only"
    assert fields["policy_loss"]["status"] == "allowed_training_stability_sanity_only"
    assert fields["action"]["status"] == "unknown_requires_new_pro_review"
    assert fields["w"]["status"] == "unknown_requires_new_pro_review"
    assert fields["preference_name"]["status"] == "unknown_requires_new_pro_review"

    validate_safe_field_policy_packet(out_dir, inventory_json=_inventory(tmp_path))


def test_safe_field_policy_rejects_wrong_phrase(tmp_path: Path):
    with pytest.raises(ValueError, match="exact Pro approval phrase"):
        generate_safe_field_policy(inventory_json=_inventory(tmp_path), out_dir=tmp_path / "out", approval_phrase="wrong")


def test_safe_field_policy_validator_rejects_missing_inventory_field(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_json(
        out_dir / "formal_jinan_safe_field_policy.json",
        {
            "report_status": "FORMAL_JINAN_SAFE_FIELD_POLICY_PASS",
            "scope": "safe_field_policy_only_no_numeric_aggregation",
            "fields": {"total_reward": {"status": "forbidden_proxy_result_metric"}},
        },
    )
    (out_dir / "formal_jinan_safe_field_policy.md").write_text("packet", encoding="utf-8")

    with pytest.raises(ValueError, match="missing policy fields"):
        validate_safe_field_policy_packet(out_dir, inventory_json=_inventory(tmp_path))


def test_safe_field_policy_validator_rejects_required_forbidden_field_allowed(tmp_path: Path):
    out_dir = tmp_path / "out"
    report = generate_safe_field_policy(
        inventory_json=_inventory(tmp_path),
        out_dir=out_dir,
        approval_phrase=FORMAL_SAFE_FIELD_POLICY_APPROVAL_PHRASE,
    )
    payload = json.loads((out_dir / "formal_jinan_safe_field_policy.json").read_text(encoding="utf-8"))
    payload["fields"]["total_reward"]["status"] = "allowed_training_stability_sanity_only"
    _write_json(out_dir / "formal_jinan_safe_field_policy.json", payload)

    with pytest.raises(ValueError, match="total_reward"):
        validate_safe_field_policy_packet(out_dir, inventory_json=_inventory(tmp_path))
