#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pareto.common.artifact_guard import FORBIDDEN_PREFLIGHT_ARTIFACTS
from pareto.rl.formal_ppo_config import load_formal_ppo_dryrun_config


FORMAL_JINAN_3SEED_EXECUTION_GUARD_BUILD_APPROVAL_PHRASE = (
    "PARETO PPO FORMAL JINAN 3-SEED EXECUTION-GUARD BUILD GO"
)
FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE = "PARETO PPO FORMAL JINAN 3-SEED EXECUTION GO"
APPROVED_FORMAL_JINAN_PPO_METHODS = (
    "vector_quality_potential",
    "film_scalar_potential",
    "weighted_proxy",
    "env_reward",
)
REFERENCE_ONLY_METHODS = ("MaxPressure", "AdvancedMaxPressure")
FORMAL_JINAN_SEEDS = (0, 1, 2)
FORMAL_JINAN_SCENARIO = "jinan"
FORMAL_JINAN_TRAFFIC_FILE = "anon_3_4_jinan_real.json"
STATE_ENCODER_HASH = "4d1c2b4e276043ac"
OBJECTIVE_NORMALIZER_HASH = "b2c55e7d2c42856a"
VECTORQ_RUN_ID = "v3_rev15_m02_iso3_c15_u03"
VECTORQ_MODEL_HASH = "ea616a51cecbc96bd5f70ba06730eeb22756254187ba972c63e9502012c61cdb"
FILM_MODEL_HASH = "08ffb15f2f33ddb0e9f6fbfd1f6a0ba107d901c3297014cc7dd5b2f94ff4e642"
METHOD_DISPLAY_NAMES = {
    "vector_quality_potential": "VectorQ-PPO",
    "film_scalar_potential": "FiLMScalar-PPO",
    "weighted_proxy": "WeightedProxy-PPO",
    "env_reward": "EnvReward-QueuePenalty-PPO",
}
FORMAL_ALLOWED_EXECUTION_OUTPUTS = {
    "command.txt",
    "metadata.json",
    "status.json",
    "stdout.txt",
    "stderr.txt",
    "train_metrics.jsonl",
    "reward_components.jsonl",
    "loss_debug.jsonl",
    "action_debug.jsonl",
    "checkpoint_last.pt",
    "training_checkpoint_last.pt",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_formal_experiment_preregistration(path: str | Path) -> dict[str, Any]:
    packet = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_formal_experiment_preregistration(packet)
    return packet


def _require_false(path: str, value: Any) -> None:
    if value is not False:
        raise ValueError(f"{path} must be false")


def _method_ids(packet: dict[str, Any]) -> tuple[str, ...]:
    methods = (packet.get("methods") or {}).get("ppo_methods") or []
    return tuple(str(item.get("method_id")) for item in methods)


def validate_formal_experiment_preregistration(packet: dict[str, Any]) -> None:
    if packet.get("packet_type") != "formal_experiment_preregistration_only":
        raise ValueError("packet_type must be formal_experiment_preregistration_only")

    packet_scope = packet.get("packet_scope") or {}
    for key in (
        "code_change_in_this_packet",
        "cityflow_run_in_this_packet",
        "ppo_training_run_in_this_packet",
        "traffic_value_reading_in_this_packet",
        "numeric_aggregation_in_this_packet",
        "ranking_or_performance_table_in_this_packet",
    ):
        _require_false(f"packet_scope.{key}", packet_scope.get(key))
    if packet_scope.get("documentation_only") is not True:
        raise ValueError("packet_scope.documentation_only must be true")

    approval = packet.get("approval") or {}
    if approval.get("received_phrase") != "PARETO PPO FORMAL-EXPERIMENT PREREGISTRATION GO":
        raise ValueError("approval.received_phrase mismatch")
    _require_false("approval.formal_experiment_allowed_now", approval.get("formal_experiment_allowed_now"))
    if approval.get("requested_next_exact_phrase") != FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE:
        raise ValueError("approval.requested_next_exact_phrase mismatch")
    if approval.get("formal_experiment_requires_new_external_approval") is not True:
        raise ValueError("approval.formal_experiment_requires_new_external_approval must be true")

    scope = packet.get("scope") or {}
    if scope.get("scenario") != FORMAL_JINAN_SCENARIO:
        raise ValueError("scope.scenario must be jinan")
    if scope.get("traffic_file") != FORMAL_JINAN_TRAFFIC_FILE:
        raise ValueError("scope.traffic_file mismatch")
    if tuple(int(seed) for seed in scope.get("seed_ids", ())) != FORMAL_JINAN_SEEDS:
        raise ValueError("scope.seed_ids must be [0, 1, 2]")
    for key in (
        "city_expansion_allowed",
        "seed_expansion_allowed_beyond_list",
        "formal_experiment_execution_in_this_packet",
        "cityflow_or_ppo_run_in_this_packet",
        "traffic_result_value_reading_in_this_packet",
    ):
        _require_false(f"scope.{key}", scope.get(key))

    method_ids = _method_ids(packet)
    if method_ids != APPROVED_FORMAL_JINAN_PPO_METHODS:
        raise ValueError(f"approved PPO methods must be {APPROVED_FORMAL_JINAN_PPO_METHODS}, got {method_ids}")
    reference_methods = tuple((packet.get("methods") or {}).get("reference_only_methods") or ())
    if reference_methods != REFERENCE_ONLY_METHODS:
        raise ValueError("reference_only_methods mismatch")
    if (packet.get("methods") or {}).get("method_ranking_allowed") is not False:
        raise ValueError("methods.method_ranking_allowed must be false")
    if (packet.get("methods") or {}).get("best_method_claim_allowed") is not False:
        raise ValueError("methods.best_method_claim_allowed must be false")
    env_method = next(item for item in (packet.get("methods") or {}).get("ppo_methods", []) if item.get("method_id") == "env_reward")
    if env_method.get("may_be_called_llmlight_original_reward") is not False:
        raise ValueError("env_reward may_be_called_llmlight_original_reward must be false")
    if env_method.get("reward_adapter_semantics") != "queue_length_penalty_proxy":
        raise ValueError("env_reward reward_adapter_semantics mismatch")

    budget = packet.get("budget") or {}
    expected_budget = {
        "episodes_per_method_seed": 5,
        "max_decision_steps_per_episode": 120,
        "min_action_time": 30,
        "sim_seconds_per_method_seed": 3600,
        "rollout_steps": 120,
    }
    for key, expected in expected_budget.items():
        if int(budget.get(key, -1)) != expected:
            raise ValueError(f"budget.{key} must be {expected}")
    if budget.get("adaptive_early_stop_allowed") is not False:
        raise ValueError("budget.adaptive_early_stop_allowed must be false")
    if budget.get("same_budget_for_all_ppo_methods") is not True:
        raise ValueError("budget.same_budget_for_all_ppo_methods must be true")

    hash_locks = packet.get("hash_locks") or {}
    if hash_locks.get("state_encoder_hash") != STATE_ENCODER_HASH:
        raise ValueError("state_encoder_hash mismatch")
    if hash_locks.get("objective_normalizer_hash") != OBJECTIVE_NORMALIZER_HASH:
        raise ValueError("objective_normalizer_hash mismatch")
    if hash_locks.get("vector_model_hash_required_for_vector_quality_potential") is not True:
        raise ValueError("vector model hash must be required for vector_quality_potential")
    if hash_locks.get("checkpoint_roundtrip_required") is not True:
        raise ValueError("checkpoint roundtrip must be required")

    seed_binding = packet.get("seed_binding") or {}
    for key in ("cityflow_seed", "policy_seed", "model_seed"):
        if seed_binding.get(key) != "seed_id":
            raise ValueError(f"seed_binding.{key} must bind to seed_id")

    allowed_outputs = set(packet.get("allowed_execution_outputs_if_later_approved") or [])
    missing_allowed = sorted(FORMAL_ALLOWED_EXECUTION_OUTPUTS - allowed_outputs)
    if missing_allowed:
        raise ValueError(f"allowed execution outputs missing: {missing_allowed}")
    forbidden_outputs = set(packet.get("forbidden_outputs_before_analysis_approval") or [])
    missing_forbidden = sorted(FORBIDDEN_PREFLIGHT_ARTIFACTS - forbidden_outputs)
    if missing_forbidden:
        raise ValueError(f"forbidden output list missing: {missing_forbidden}")


def _command_preview(
    *,
    method: str,
    seed_id: int,
    base_spec_path: str,
    out_root: str,
    objective_normalizer: str,
    rollout_steps: int,
) -> str:
    out_dir = f"{out_root}/seed{seed_id}/{method}"
    parts = [
        "python",
        "pareto/rl/formal_pilot_runner.py",
        "--spec",
        base_spec_path,
        "--method",
        method,
        "--formal_jinan_3seed_execution",
        "--approval_phrase",
        f"'{FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE}'",
        "--seed_id",
        str(seed_id),
        "--episodes",
        "5",
        "--max_decision_steps_per_episode",
        "120",
        "--rollout_steps",
        str(int(rollout_steps)),
        "--objective_normalizer",
        objective_normalizer,
        "--objective_normalizer_hash",
        OBJECTIVE_NORMALIZER_HASH,
        "--out_dir",
        out_dir,
    ]
    if method == "vector_quality_potential":
        parts.extend(
            [
                "--vector_model_dir",
                "model_weights/pareto_quality/jinan/eval_consistency_remediation_v3/v3_rev15_m02_iso3_c15_u03",
                "--vector_model_hash",
                VECTORQ_MODEL_HASH,
            ]
        )
    if method == "film_scalar_potential":
        parts.extend(
            [
                "--film_model_dir",
                "model_weights/cond_scalar/jinan/preformal_final/film_rich_v2",
                "--film_model_hash",
                FILM_MODEL_HASH,
            ]
        )
    return " ".join(parts)


def _load_and_validate_base_spec(base_spec_path: str | Path, packet: dict[str, Any]) -> dict[str, Any]:
    config = load_formal_ppo_dryrun_config(base_spec_path)
    budget = packet["budget"]
    if int(config.ppo["rollout_steps"]) != int(budget["rollout_steps"]):
        raise ValueError(
            "base_spec ppo.rollout_steps must match preregistered budget.rollout_steps "
            f"({config.ppo['rollout_steps']} != {budget['rollout_steps']})"
        )
    if str(config.pilot.get("scenario")) != FORMAL_JINAN_SCENARIO:
        raise ValueError("base_spec pilot.scenario must be jinan")
    if str(config.pilot.get("traffic_file")) != FORMAL_JINAN_TRAFFIC_FILE:
        raise ValueError("base_spec pilot.traffic_file mismatch")
    if str(config.pilot.get("state_encoder_hash")) != STATE_ENCODER_HASH:
        raise ValueError("base_spec state_encoder_hash mismatch")
    if str(config.pilot.get("objective_normalizer_hash")) != OBJECTIVE_NORMALIZER_HASH:
        raise ValueError("base_spec objective_normalizer_hash mismatch")
    if str(config.pilot.get("vector_model_hash")) != VECTORQ_MODEL_HASH:
        raise ValueError("base_spec vector_model_hash mismatch")
    if str(config.pilot.get("film_model_hash")) != FILM_MODEL_HASH:
        raise ValueError("base_spec film_model_hash mismatch")
    return {
        "path": str(base_spec_path),
        "ppo_config_hash": config.ppo_config_hash(),
        "rollout_steps": int(config.ppo["rollout_steps"]),
        "scenario": str(config.pilot.get("scenario")),
        "traffic_file": str(config.pilot.get("traffic_file")),
    }


def build_formal_jinan_3seed_execution_manifest(
    packet: dict[str, Any],
    *,
    base_spec_path: str,
    out_root: str,
    objective_normalizer: str = "records/eval_consistency_remediation_v3/objective_norm_smoke3600.json",
) -> list[dict[str, Any]]:
    validate_formal_experiment_preregistration(packet)
    budget = packet["budget"]
    base_spec = _load_and_validate_base_spec(base_spec_path, packet)
    manifest: list[dict[str, Any]] = []
    for seed_id in FORMAL_JINAN_SEEDS:
        for method in APPROVED_FORMAL_JINAN_PPO_METHODS:
            row = {
                "command_kind": "template_only_not_executed",
                "formal_execution_allowed_now": False,
                "approval_phrase_required": FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE,
                "method": method,
                "method_display_name": METHOD_DISPLAY_NAMES[method],
                "seed_id": seed_id,
                "scenario": FORMAL_JINAN_SCENARIO,
                "traffic_file": FORMAL_JINAN_TRAFFIC_FILE,
                "cityflow_seed": seed_id,
                "policy_seed": seed_id,
                "model_seed": seed_id,
                "episodes": int(budget["episodes_per_method_seed"]),
                "max_decision_steps_per_episode": int(budget["max_decision_steps_per_episode"]),
                "min_action_time": int(budget["min_action_time"]),
                "sim_seconds_per_method_seed": int(budget["sim_seconds_per_method_seed"]),
                "rollout_steps": int(budget["rollout_steps"]),
                "base_spec_path": base_spec["path"],
                "base_spec_ppo_config_hash": base_spec["ppo_config_hash"],
                "base_spec_rollout_steps": base_spec["rollout_steps"],
                "state_encoder_hash": STATE_ENCODER_HASH,
                "objective_normalizer_hash": OBJECTIVE_NORMALIZER_HASH,
                "vector_model_hash": VECTORQ_MODEL_HASH if method == "vector_quality_potential" else None,
                "film_model_hash": FILM_MODEL_HASH if method == "film_scalar_potential" else None,
                "reward_adapter_semantics": "queue_length_penalty_proxy" if method == "env_reward" else method,
                "out_dir": f"{out_root}/seed{seed_id}/{method}",
                "command_preview": _command_preview(
                    method=method,
                    seed_id=seed_id,
                    base_spec_path=base_spec_path,
                    out_root=out_root,
                    objective_normalizer=objective_normalizer,
                    rollout_steps=int(budget["rollout_steps"]),
                ),
            }
            manifest.append(row)
    return manifest


def _check_manifest_shape(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    methods = {row["method"] for row in manifest}
    seeds = {int(row["seed_id"]) for row in manifest}
    expected_count = len(APPROVED_FORMAL_JINAN_PPO_METHODS) * len(FORMAL_JINAN_SEEDS)
    passed = (
        len(manifest) == expected_count
        and methods == set(APPROVED_FORMAL_JINAN_PPO_METHODS)
        and seeds == set(FORMAL_JINAN_SEEDS)
    )
    return {"pass": passed, "row_count": len(manifest), "methods": sorted(methods), "seeds": sorted(seeds)}


def _check_budget_and_hashes(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    for row in manifest:
        if row["episodes"] != 5:
            failures.append(f"{row['method']} seed{row['seed_id']}: episodes mismatch")
        if row["max_decision_steps_per_episode"] != 120:
            failures.append(f"{row['method']} seed{row['seed_id']}: decision steps mismatch")
        if row["min_action_time"] != 30:
            failures.append(f"{row['method']} seed{row['seed_id']}: min action time mismatch")
        if row["rollout_steps"] != 120:
            failures.append(f"{row['method']} seed{row['seed_id']}: rollout steps mismatch")
        if row["base_spec_rollout_steps"] != row["rollout_steps"]:
            failures.append(f"{row['method']} seed{row['seed_id']}: base spec rollout mismatch")
        if row["state_encoder_hash"] != STATE_ENCODER_HASH:
            failures.append(f"{row['method']} seed{row['seed_id']}: state hash mismatch")
        if row["objective_normalizer_hash"] != OBJECTIVE_NORMALIZER_HASH:
            failures.append(f"{row['method']} seed{row['seed_id']}: normalizer hash mismatch")
        if row["method"] == "vector_quality_potential" and row["vector_model_hash"] != VECTORQ_MODEL_HASH:
            failures.append(f"{row['method']} seed{row['seed_id']}: vector hash mismatch")
        if row["method"] == "film_scalar_potential" and row["film_model_hash"] != FILM_MODEL_HASH:
            failures.append(f"{row['method']} seed{row['seed_id']}: film hash mismatch")
    return {"pass": not failures, "failures": failures}


def _check_base_spec_budget_consistency(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    rollouts = sorted({int(row["rollout_steps"]) for row in manifest})
    base_rollouts = sorted({int(row["base_spec_rollout_steps"]) for row in manifest})
    ppo_hashes = sorted({str(row["base_spec_ppo_config_hash"]) for row in manifest})
    passed = rollouts == [120] and base_rollouts == [120] and len(ppo_hashes) == 1
    return {
        "pass": passed,
        "rollout_steps": rollouts,
        "base_spec_rollout_steps": base_rollouts,
        "base_spec_ppo_config_hashes": ppo_hashes,
    }


def _check_forbidden_actions_blocked(packet: dict[str, Any]) -> dict[str, Any]:
    blocked_states = {
        "approval.formal_experiment_allowed_now": (packet.get("approval") or {}).get("formal_experiment_allowed_now"),
        "packet_scope.cityflow_run_in_this_packet": (packet.get("packet_scope") or {}).get("cityflow_run_in_this_packet"),
        "packet_scope.ppo_training_run_in_this_packet": (packet.get("packet_scope") or {}).get("ppo_training_run_in_this_packet"),
        "packet_scope.traffic_value_reading_in_this_packet": (
            packet.get("packet_scope") or {}
        ).get("traffic_value_reading_in_this_packet"),
        "methods.method_ranking_allowed": (packet.get("methods") or {}).get("method_ranking_allowed"),
        "methods.best_method_claim_allowed": (packet.get("methods") or {}).get("best_method_claim_allowed"),
    }
    lock_checks = {
        f"{key}.locked_false": value is False
        for key, value in blocked_states.items()
    }
    return {"pass": all(lock_checks.values()), "blocked_states": blocked_states, "lock_checks": lock_checks}


def build_execution_guard_packet(
    packet: dict[str, Any],
    *,
    out_dir: str | Path,
    base_spec_path: str,
    out_root: str,
    preregistration_commit: str,
    guard_build_commit: str,
    preregistration_packet_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = build_formal_jinan_3seed_execution_manifest(
        packet,
        base_spec_path=base_spec_path,
        out_root=out_root,
    )
    checks = {
        "manifest_shape": _check_manifest_shape(manifest),
        "base_spec_budget_consistency": _check_base_spec_budget_consistency(manifest),
        "budget_and_hashes": _check_budget_and_hashes(manifest),
        "forbidden_actions_blocked": _check_forbidden_actions_blocked(packet),
        "artifact_allowlist_declared": {
            "pass": set(packet.get("allowed_execution_outputs_if_later_approved") or []) >= FORMAL_ALLOWED_EXECUTION_OUTPUTS,
            "allowed_outputs": sorted(packet.get("allowed_execution_outputs_if_later_approved") or []),
        },
        "forbidden_artifacts_declared": {
            "pass": set(packet.get("forbidden_outputs_before_analysis_approval") or []) >= FORBIDDEN_PREFLIGHT_ARTIFACTS,
            "forbidden_outputs": sorted(packet.get("forbidden_outputs_before_analysis_approval") or []),
        },
    }
    guard = {
        "packet_type": "formal_jinan_3seed_execution_guard",
        "approval_phrase_used_for_build": FORMAL_JINAN_3SEED_EXECUTION_GUARD_BUILD_APPROVAL_PHRASE,
        "formal_experiment_execution_in_this_packet": False,
        "formal_execution_allowed_now": False,
        "traffic_value_reading_in_this_packet": False,
        "numeric_aggregation_in_this_packet": False,
        "method_ranking_in_this_packet": False,
        "provenance": {
            "preregistration_commit": preregistration_commit,
            "guard_build_commit": guard_build_commit,
            "preregistration_packet": str(
                preregistration_packet_path
                or (
                    "docs/pro_reviews/pareto_ppo_formal_experiment_preregistration_2026-06-01/"
                    "formal_experiment_preregistration.json"
                )
            ),
        },
        "execution_guard_checks": checks,
        "run_manifest": manifest,
        "next_gate": {
            "packet_type": "formal_jinan_3seed_execution_request",
            "required_exact_phrase": FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE,
            "analysis_approval_requested": False,
            "ranking_approval_requested": False,
            "traffic_value_reading_approval_requested": False,
        },
        "forbidden_until_later_analysis_gate": [
            "traffic_value_reading",
            "numeric_aggregation",
            "method_ranking",
            "performance_table",
            "best_method_claim",
            "traffic_improvement_claim",
            "paper_ready_claim",
        ],
    }
    guard["overall_pass"] = all(item.get("pass") for item in checks.values())
    out_path = Path(out_dir)
    _write_json(out_path / "formal_jinan_3seed_execution_guard.json", guard)
    write_markdown(guard, out_path / "formal_jinan_3seed_execution_guard.md")
    return guard


def write_markdown(guard: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# Formal Jinan 3-Seed Execution Guard Packet",
        "",
        f"- overall_pass: `{guard['overall_pass']}`",
        f"- build approval phrase: `{guard['approval_phrase_used_for_build']}`",
        f"- formal execution allowed now: `{guard['formal_execution_allowed_now']}`",
        f"- formal experiment executed in this packet: `{guard['formal_experiment_execution_in_this_packet']}`",
        f"- next required exact phrase: `{guard['next_gate']['required_exact_phrase']}`",
        "",
        "## Guard Checks",
    ]
    for name, result in guard["execution_guard_checks"].items():
        lines.append(f"- {name}: `{'PASS' if result.get('pass') else 'FAIL'}`")
    lines.extend(
        [
            "",
            "## Manifest Summary",
            "",
            f"- manifest rows: `{len(guard['run_manifest'])}`",
            f"- methods: `{', '.join(APPROVED_FORMAL_JINAN_PPO_METHODS)}`",
            f"- seeds: `{', '.join(str(seed) for seed in FORMAL_JINAN_SEEDS)}`",
            "",
            "This packet builds and validates command templates only. It does not run CityFlow, PPO, formal execution, traffic-value reading, aggregation, ranking, or performance-table generation.",
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--base_spec", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--preregistration_commit", required=True)
    parser.add_argument("--guard_build_commit", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet = load_formal_experiment_preregistration(args.preregistration)
    guard = build_execution_guard_packet(
        packet,
        out_dir=args.out_dir,
        base_spec_path=args.base_spec,
        out_root=args.out_root,
        preregistration_commit=args.preregistration_commit,
        guard_build_commit=args.guard_build_commit,
        preregistration_packet_path=args.preregistration,
    )
    print(json.dumps({"overall_pass": guard["overall_pass"], "manifest_rows": len(guard["run_manifest"])}, sort_keys=True))


if __name__ == "__main__":
    main()
