from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FORMAL_PILOT_READINESS_PACKET_STATUS = "PARETO_PPO_FORMAL_PILOT_READINESS_PACKET_VALID"
ALLOWED_FORMAL_PILOT_READINESS_OUTPUTS = {
    "formal_pilot_readiness_packet.json",
    "formal_pilot_readiness_packet.md",
}
EXPECTED_METHODS = ("film_scalar_potential", "weighted_proxy", "env_reward")
EXPECTED_SCOPE = {
    "city": "jinan",
    "seed": 0,
    "traffic_file": "anon_3_4_jinan_real.json",
    "run_type": "exploratory_pilot_only",
}
EXPECTED_BUDGET = {
    "episodes": 1,
    "max_decision_steps_per_episode": 120,
}
REQUIRED_METHOD_CHECKS = {
    "closed_loop_executed",
    "reward_finite_nonzero",
    "loss_finite",
    "ppo_update",
    "action_non_collapse",
    "checkpoint_roundtrip",
    "artifact_allowlist",
    "formal_flags",
}
FORBIDDEN_PERMISSION_KEYS = {
    "formal_ppo_allowed",
    "formal_experiment_allowed",
    "performance_claim_allowed",
    "method_comparison_allowed",
    "method_ranking_allowed",
    "performance_table_allowed",
    "seed_expansion_allowed",
    "city_expansion_allowed",
    "traffic_metric_reading_allowed",
    "paper_result_allowed",
}
FORBIDDEN_PACKET_CONTENT_KEYS = {
    "traffic_performance_values_included",
    "method_comparison_included",
    "ranking_included",
    "formal_table_included",
    "winner_claim_included",
    "improvement_claim_included",
    "paper_result_included",
}
FORBIDDEN_CLAIM_WORDING = {
    "beats",
    "better than",
    "leaderboard",
    "main result",
    "outperforms",
    "paper-ready result",
    "performance gain",
    "ranked",
    "state-of-the-art",
    "traffic improvement",
    "wins",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_false(mapping: dict[str, Any], keys: set[str], context: str) -> None:
    for key in sorted(keys):
        if mapping.get(key) is not False:
            raise ValueError(f"{context}.{key} must be false")


def _require_status_pass(entry: dict[str, Any], context: str) -> None:
    if entry.get("status") != "PASS":
        raise ValueError(f"{context}.status must be PASS")


def validate_formal_pilot_readiness_packet(out_dir: str | Path) -> None:
    root = Path(out_dir)
    existing = {path.name for path in root.iterdir() if path.is_file()}
    missing = sorted(ALLOWED_FORMAL_PILOT_READINESS_OUTPUTS - existing)
    if missing:
        raise ValueError(f"missing formal pilot readiness outputs: {missing}")
    unexpected = sorted(existing - ALLOWED_FORMAL_PILOT_READINESS_OUTPUTS)
    if unexpected:
        raise ValueError(f"unexpected formal pilot readiness outputs: {unexpected}")

    packet = _load_json(root / "formal_pilot_readiness_packet.json")
    if packet.get("packet_type") != "pareto_ppo_formal_pilot_readiness_packet":
        raise ValueError("packet_type must be pareto_ppo_formal_pilot_readiness_packet")
    if packet.get("packet_status") != FORMAL_PILOT_READINESS_PACKET_STATUS:
        raise ValueError("formal pilot readiness packet status mismatch")
    if packet.get("scope") != "formal_pilot_readiness_packet_only_no_new_run":
        raise ValueError("formal pilot readiness packet scope mismatch")

    permissions = packet.get("permissions") or {}
    _require_false(permissions, FORBIDDEN_PERMISSION_KEYS, "permissions")

    forbidden_contents = packet.get("forbidden_contents") or {}
    _require_false(forbidden_contents, FORBIDDEN_PACKET_CONTENT_KEYS, "forbidden_contents")

    budget = packet.get("exploratory_budget") or {}
    for key, expected in EXPECTED_BUDGET.items():
        if budget.get(key) != expected:
            raise ValueError(f"exploratory_budget.{key} must be {expected}")

    engineering_gate = packet.get("closed_loop_engineering_gate") or {}
    if engineering_gate.get("status") != "PASS":
        raise ValueError("closed_loop_engineering_gate.status must be PASS")
    if tuple(engineering_gate.get("methods") or ()) != EXPECTED_METHODS:
        raise ValueError("closed_loop_engineering_gate.methods must match expected order")

    method_audits = packet.get("method_guard_audits") or {}
    if set(method_audits) != set(EXPECTED_METHODS):
        raise ValueError("method_guard_audits must contain exactly the expected methods")
    for method in EXPECTED_METHODS:
        audit = method_audits[method]
        audit_scope = audit.get("scope") or {}
        for key, expected in EXPECTED_SCOPE.items():
            if audit_scope.get(key) != expected:
                raise ValueError(f"{method}.scope.{key} must be {expected}")
        checks = audit.get("checks") or {}
        missing_checks = sorted(REQUIRED_METHOD_CHECKS - set(checks))
        if missing_checks:
            raise ValueError(f"{method} missing checks: {missing_checks}")
        for check_name in REQUIRED_METHOD_CHECKS:
            _require_status_pass(checks[check_name], f"{method}.{check_name}")
        if checks["ppo_update"].get("policy_update_count", 0) <= 0:
            raise ValueError(f"{method}.ppo_update.policy_update_count must be positive")
        if checks["action_non_collapse"].get("unique_actions", 0) <= 1:
            raise ValueError(f"{method}.action_non_collapse.unique_actions must exceed 1")
        if checks["action_non_collapse"].get("global_single_action_rate", 1.0) > checks[
            "action_non_collapse"
        ].get("threshold", 0.95):
            raise ValueError(f"{method}.action_non_collapse exceeds threshold")

    env_reward = method_audits["env_reward"]
    if env_reward.get("reward_adapter_semantics") != "queue_length_penalty_proxy":
        raise ValueError("env_reward reward_adapter_semantics must remain queue_length_penalty_proxy")
    if env_reward.get("allowed_role") != "diagnostic_ablation_only":
        raise ValueError("env_reward allowed_role must remain diagnostic_ablation_only")

    representation = packet.get("offline_representation_gate") or {}
    if representation.get("status") != "partial_pass":
        raise ValueError("offline_representation_gate.status must remain partial_pass")
    if representation.get("formal_pass") is not False:
        raise ValueError("offline_representation_gate.formal_pass must be false")

    next_gate = packet.get("next_gate_request") or {}
    if next_gate.get("formal_experiment_permission_requested") is not False:
        raise ValueError("next gate must not request formal experiment permission")

    prose = (root / "formal_pilot_readiness_packet.md").read_text(encoding="utf-8").lower()
    wording_hits = sorted(word for word in FORBIDDEN_CLAIM_WORDING if word in prose)
    if wording_hits:
        raise ValueError(f"formal pilot readiness packet contains forbidden claim wording: {wording_hits}")
