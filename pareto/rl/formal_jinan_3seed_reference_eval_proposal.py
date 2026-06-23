#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.rl.formal_jinan_3seed_artifact_hash_verification import sha256_file
from pareto.rl.formal_jinan_3seed_execution_guard import (
    APPROVED_FORMAL_JINAN_PPO_METHODS,
    FORMAL_JINAN_SEEDS,
    FORMAL_JINAN_TRAFFIC_FILE,
    REFERENCE_ONLY_METHODS,
)


FUTURE_REFERENCE_EVAL_APPROVAL_PHRASE = "PARETO PPO FORMAL JINAN 3-SEED REFERENCE-EVAL GO"
PROPOSAL_PACKET_TYPE = "formal_jinan_3seed_reference_eval_proposal"
DEFAULT_ANALYSIS_PACKET = (
    "docs/pro_reviews/pareto_ppo_formal_jinan_3seed_analysis_2026-06-01/"
    "formal_jinan_3seed_analysis.json"
)
DEFAULT_EVAL_OUTPUT_ROOT = "records/formal_jinan_3seed_eval_20260601"
DEFAULT_TRAIN_RUN_ROOT = "records/formal_jinan_3seed_guarded_20260601"
DEFAULT_PROPOSAL_DIR = (
    "docs/pro_reviews/pareto_ppo_formal_jinan_3seed_reference_eval_proposal_2026-06-01"
)

FUTURE_EVAL_ALLOWED_RAW_OUTPUTS = (
    "metadata.json",
    "status.json",
    "eval_metrics.json",
    "eval_metrics.jsonl",
    "eval_guard_report.json",
    "stdout.txt",
    "stderr.txt",
)
FUTURE_EVAL_FORBIDDEN_OUTPUTS = (
    "performance_table.csv",
    "performance_table.json",
    "performance_table.md",
    "performance_table.tex",
    "ranking.csv",
    "leaderboard.csv",
    "main_results.csv",
    "method_ranking.csv",
    "best_method.json",
    "best_method.txt",
    "traffic_improvement.json",
    "traffic_improvement.md",
    "paper_results.csv",
)
FORBIDDEN_PROPOSAL_TRUE_FLAGS = (
    "formal_reference_eval_execution_allowed_now",
    "cityflow_run_in_this_packet",
    "model_rollout_in_this_packet",
    "traffic_result_value_reading_in_this_packet",
    "numeric_traffic_aggregation_in_this_packet",
    "method_ranking_in_this_packet",
    "performance_table_in_this_packet",
    "best_method_claim_in_this_packet",
    "traffic_improvement_claim_in_this_packet",
    "paper_result_claim_in_this_packet",
    "seed_expansion_in_this_packet",
    "city_expansion_in_this_packet",
    "new_method_in_this_packet",
)
REQUIRED_ANALYSIS_PERMISSION_FALSE_FLAGS = (
    "method_ranking_executed",
    "performance_table_generated",
    "best_method_claim_generated",
    "traffic_improvement_claim_generated",
    "paper_ready_claim_generated",
    "seed_expansion_executed",
    "city_expansion_executed",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _check_analysis_packet(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"pass": False, "path": str(path), "failure": f"missing analysis packet: {path}"}
    payload = _read_json(path)
    permissions = payload.get("permissions") or {}
    failures: list[str] = []
    if payload.get("packet_type") != "formal_jinan_3seed_descriptive_analysis":
        failures.append("packet_type must be formal_jinan_3seed_descriptive_analysis")
    if payload.get("analysis_status") != "PASS":
        failures.append("analysis_status must be PASS")
    if permissions.get("comparison_requires_new_gate") is not True:
        failures.append("comparison_requires_new_gate must be true")
    if permissions.get("not_for_main_results") is not True:
        failures.append("not_for_main_results must be true")
    if permissions.get("exclude_from_paper") is not True:
        failures.append("exclude_from_paper must be true")
    for key in REQUIRED_ANALYSIS_PERMISSION_FALSE_FLAGS:
        if permissions.get(key) is not False:
            failures.append(f"permissions.{key} must be false")
    provenance = payload.get("provenance") or {}
    required_provenance = (
        "guard_packet_sha256",
        "verification_packet_sha256",
        "request_packet_sha256",
        "execution_audit_packet_sha256",
        "execution_audit_commit",
    )
    for key in required_provenance:
        if not provenance.get(key):
            failures.append(f"provenance.{key} missing")
    return {
        "pass": not failures,
        "path": str(path),
        "file_sha256": sha256_file(path),
        "packet_type": payload.get("packet_type"),
        "analysis_status": payload.get("analysis_status"),
        "permissions_checked": sorted(permissions),
        "provenance": {key: provenance.get(key) for key in required_provenance},
        "failure": "; ".join(failures) if failures else None,
    }


def _future_eval_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for seed in FORMAL_JINAN_SEEDS:
        for method in APPROVED_FORMAL_JINAN_PPO_METHODS:
            runs.append(
                {
                    "run_kind": "ppo_checkpoint_eval",
                    "seed": int(seed),
                    "method": method,
                    "checkpoint_source_run_dir": f"{DEFAULT_TRAIN_RUN_ROOT}/seed{seed}/{method}",
                    "checkpoint_file": "checkpoint_last.pt",
                    "training_checkpoint_file": "training_checkpoint_last.pt",
                    "checkpoint_sha256_required_before_eval": True,
                    "checkpoint_load_required_before_eval": True,
                }
            )
        for baseline in REFERENCE_ONLY_METHODS:
            runs.append(
                {
                    "run_kind": "reference_policy_eval",
                    "seed": int(seed),
                    "method": baseline,
                    "checkpoint_source_run_dir": None,
                    "checkpoint_file": None,
                    "same_eval_protocol_required": True,
                }
            )
    return runs


def _base_packet(analysis_check: dict[str, Any], *, out_dir: str | Path) -> dict[str, Any]:
    analysis_provenance = analysis_check.get("provenance") or {}
    return {
        "packet_type": PROPOSAL_PACKET_TYPE,
        "proposal_name": "formal_jinan_3seed_reference_eval_proposal",
        "proposal_status": "PASS" if analysis_check.get("pass") else "FAIL",
        "overall_pass": bool(analysis_check.get("pass")),
        "failures": [] if analysis_check.get("pass") else [f"analysis packet: {analysis_check.get('failure')}"],
        "formal_reference_eval_execution_allowed_now": False,
        "cityflow_run_in_this_packet": False,
        "model_rollout_in_this_packet": False,
        "traffic_result_value_reading_in_this_packet": False,
        "numeric_traffic_aggregation_in_this_packet": False,
        "method_ranking_in_this_packet": False,
        "performance_table_in_this_packet": False,
        "best_method_claim_in_this_packet": False,
        "traffic_improvement_claim_in_this_packet": False,
        "paper_result_claim_in_this_packet": False,
        "seed_expansion_in_this_packet": False,
        "city_expansion_in_this_packet": False,
        "new_method_in_this_packet": False,
        "locked_future_eval_scope": {
            "scenario": "jinan",
            "traffic_file": FORMAL_JINAN_TRAFFIC_FILE,
            "seed_ids": [int(seed) for seed in FORMAL_JINAN_SEEDS],
            "methods": list(APPROVED_FORMAL_JINAN_PPO_METHODS),
            "reference_baselines": list(REFERENCE_ONLY_METHODS),
            "llmlight_llm_baselines_included": False,
            "llmlight_llm_baseline_requires_separate_gate": True,
            "min_action_time": 30,
            "episodes_per_method_seed": 5,
            "max_decision_steps_per_episode": 120,
            "sim_seconds_per_method_seed": 3600,
            "same_protocol_for_methods_and_references": True,
            "training_budget_not_used_as_eval_metric": True,
            "eval_runs": _future_eval_runs(),
        },
        "locked_future_eval_metrics": {
            "primary_metric": "average_travel_time",
            "secondary_metrics": ["throughput", "mean_queue_length"],
            "training_reward_excluded": True,
            "proxy_reward_excluded_from_traffic_claims": True,
            "reward_components_jsonl_excluded": True,
            "train_metrics_jsonl_excluded": True,
            "result_interpretation_requires_later_comparison_gate": True,
        },
        "future_eval_artifact_policy": {
            "eval_output_root": DEFAULT_EVAL_OUTPUT_ROOT,
            "must_not_write_to_training_run_root": DEFAULT_TRAIN_RUN_ROOT,
            "allowed_raw_outputs": list(FUTURE_EVAL_ALLOWED_RAW_OUTPUTS),
            "raw_metric_output_names": ["eval_metrics.json", "eval_metrics.jsonl"],
            "forbidden_outputs": list(FUTURE_EVAL_FORBIDDEN_OUTPUTS),
            "performance_table_allowed": False,
            "ranking_allowed": False,
            "best_method_artifact_allowed": False,
            "paper_result_artifact_allowed": False,
            "artifact_allowlist_validator_required_before_execution": True,
        },
        "future_eval_guard_requirements": {
            "checkpoint_to_eval_binding_required": True,
            "real_checkpoint_file_sha256_required": True,
            "checkpoint_load_required": True,
            "new_eval_directory_required": True,
            "forbidden_artifact_scan_required": True,
            "traffic_value_reading_only_after_reference_eval_gate": True,
            "ranking_or_claim_generation_requires_later_comparison_gate": True,
            "targeted_tests_required_before_execution": [
                "tests/pareto/test_formal_jinan_3seed_reference_eval_proposal.py",
            ],
            "remote_full_tests_required_before_execution": True,
        },
        "provenance": {
            "analysis_packet": str(analysis_check.get("path")),
            "analysis_packet_sha256": analysis_check.get("file_sha256"),
            "guard_packet_sha256": analysis_provenance.get("guard_packet_sha256"),
            "verification_packet_sha256": analysis_provenance.get("verification_packet_sha256"),
            "request_packet_sha256": analysis_provenance.get("request_packet_sha256"),
            "execution_audit_packet_sha256": analysis_provenance.get("execution_audit_packet_sha256"),
            "execution_audit_commit": analysis_provenance.get("execution_audit_commit"),
            "analysis_commit": "92df005",
            "proposal_output_dir": str(out_dir),
        },
        "blockers_addressed_by_proposal": {
            "B-E1_protocol_not_locked": "addressed_by_locked_future_eval_scope_and_metrics",
            "B-E2_reference_same_budget_protocol_undefined": "addressed_by_reference_baselines_and_same_protocol_flag",
            "B-E3_checkpoint_binding_and_sha_not_confirmed": "future_guard_requirement_locked_no_eval_executed",
            "B-E4_eval_artifact_allowlist_not_locked": "addressed_by_future_eval_artifact_policy",
            "B-E5_boundary_false_flags_missing": "addressed_by_false_scope_flags",
            "B-E6_validator_tests_not_done": "this_packet_adds_validator_tests_but_execution_gate_still_pending",
        },
        "next_gate": {
            "required_exact_phrase": FUTURE_REFERENCE_EVAL_APPROVAL_PHRASE,
            "reference_eval_execution_requested": True,
            "comparison_or_ranking_requested": False,
            "performance_table_requested": False,
            "paper_result_requested": False,
        },
    }


def validate_reference_eval_proposal_packet(packet: dict[str, Any]) -> None:
    if packet.get("packet_type") != PROPOSAL_PACKET_TYPE:
        raise ValueError(f"packet_type must be {PROPOSAL_PACKET_TYPE}")
    for key in FORBIDDEN_PROPOSAL_TRUE_FLAGS:
        if packet.get(key) is not False:
            raise ValueError(f"{key} must be false")
    scope = packet.get("locked_future_eval_scope") or {}
    if scope.get("scenario") != "jinan":
        raise ValueError("locked_future_eval_scope.scenario must be jinan")
    if scope.get("traffic_file") != FORMAL_JINAN_TRAFFIC_FILE:
        raise ValueError("locked_future_eval_scope.traffic_file mismatch")
    if tuple(int(seed) for seed in scope.get("seed_ids", ())) != FORMAL_JINAN_SEEDS:
        raise ValueError("locked_future_eval_scope.seed_ids must be [0, 1, 2]")
    if tuple(scope.get("methods") or ()) != APPROVED_FORMAL_JINAN_PPO_METHODS:
        raise ValueError("locked_future_eval_scope.methods mismatch")
    if tuple(scope.get("reference_baselines") or ()) != REFERENCE_ONLY_METHODS:
        raise ValueError("locked_future_eval_scope.reference_baselines mismatch")
    if scope.get("same_protocol_for_methods_and_references") is not True:
        raise ValueError("same_protocol_for_methods_and_references must be true")
    if int(scope.get("min_action_time", -1)) != 30:
        raise ValueError("min_action_time must be 30")

    metrics = packet.get("locked_future_eval_metrics") or {}
    if metrics.get("primary_metric") != "average_travel_time":
        raise ValueError("primary_metric must be average_travel_time")
    if metrics.get("training_reward_excluded") is not True:
        raise ValueError("training_reward_excluded must be true")
    forbidden_metrics = {"training_reward", "total_reward", "env_reward", "weighted_proxy_reward", "potential_reward"}
    leaked_metrics = sorted(forbidden_metrics & {str(item) for item in metrics.get("secondary_metrics") or []})
    if leaked_metrics:
        raise ValueError(f"secondary_metrics includes reward/proxy metrics: {leaked_metrics}")

    artifact_policy = packet.get("future_eval_artifact_policy") or {}
    if artifact_policy.get("eval_output_root") != DEFAULT_EVAL_OUTPUT_ROOT:
        raise ValueError("eval_output_root mismatch")
    allowed = set(artifact_policy.get("allowed_raw_outputs") or [])
    forbidden = set(artifact_policy.get("forbidden_outputs") or [])
    if not {"eval_metrics.json", "eval_metrics.jsonl"} <= allowed:
        raise ValueError("allowed_raw_outputs must include eval_metrics.json and eval_metrics.jsonl")
    overlap = sorted(allowed & forbidden)
    if overlap:
        raise ValueError(f"allowed_raw_outputs overlaps forbidden_outputs: {overlap}")
    for required_forbidden in ("performance_table.csv", "ranking.csv", "best_method.json", "paper_results.csv"):
        if required_forbidden not in forbidden:
            raise ValueError(f"forbidden_outputs missing {required_forbidden}")
    for key in ("performance_table_allowed", "ranking_allowed", "best_method_artifact_allowed", "paper_result_artifact_allowed"):
        if artifact_policy.get(key) is not False:
            raise ValueError(f"future_eval_artifact_policy.{key} must be false")

    next_gate = packet.get("next_gate") or {}
    if next_gate.get("required_exact_phrase") != FUTURE_REFERENCE_EVAL_APPROVAL_PHRASE:
        raise ValueError("next_gate.required_exact_phrase mismatch")
    if next_gate.get("comparison_or_ranking_requested") is not False:
        raise ValueError("comparison_or_ranking_requested must be false")

    if packet.get("overall_pass") is not (not packet.get("failures")):
        raise ValueError("overall_pass must match failures")


def build_reference_eval_proposal_packet(
    *,
    out_dir: str | Path = DEFAULT_PROPOSAL_DIR,
    analysis_packet: str | Path = DEFAULT_ANALYSIS_PACKET,
) -> dict[str, Any]:
    output = Path(out_dir)
    analysis_check = _check_analysis_packet(Path(analysis_packet))
    packet = _base_packet(analysis_check, out_dir=output)
    if packet["overall_pass"]:
        validate_reference_eval_proposal_packet(packet)
    _write_json(output / "formal_jinan_3seed_reference_eval_proposal.json", packet)
    write_markdown(packet, output / "formal_jinan_3seed_reference_eval_proposal.md")
    return packet


def write_markdown(packet: dict[str, Any], path: str | Path) -> None:
    scope = packet["locked_future_eval_scope"]
    metrics = packet["locked_future_eval_metrics"]
    artifacts = packet["future_eval_artifact_policy"]
    lines = [
        "# Formal Jinan 3-Seed Reference-Eval Proposal",
        "",
        f"- proposal_status: `{packet['proposal_status']}`",
        f"- overall_pass: `{packet['overall_pass']}`",
        f"- formal reference-eval allowed now: `{packet['formal_reference_eval_execution_allowed_now']}`",
        f"- CityFlow run in this packet: `{packet['cityflow_run_in_this_packet']}`",
        f"- model rollout in this packet: `{packet['model_rollout_in_this_packet']}`",
        f"- traffic result value reading in this packet: `{packet['traffic_result_value_reading_in_this_packet']}`",
        f"- next required exact phrase: `{packet['next_gate']['required_exact_phrase']}`",
        "",
        "## Locked Future Scope",
        "",
        f"- scenario: `{scope['scenario']}`",
        f"- traffic_file: `{scope['traffic_file']}`",
        f"- seeds: `{scope['seed_ids']}`",
        f"- PPO methods: `{scope['methods']}`",
        f"- reference baselines: `{scope['reference_baselines']}`",
        f"- same protocol for methods and references: `{scope['same_protocol_for_methods_and_references']}`",
        f"- min_action_time: `{scope['min_action_time']}`",
        f"- max_decision_steps_per_episode: `{scope['max_decision_steps_per_episode']}`",
        "",
        "## Locked Future Metrics",
        "",
        f"- primary metric: `{metrics['primary_metric']}`",
        f"- secondary metrics: `{metrics['secondary_metrics']}`",
        f"- training reward excluded: `{metrics['training_reward_excluded']}`",
        f"- proxy reward excluded from traffic claims: `{metrics['proxy_reward_excluded_from_traffic_claims']}`",
        "",
        "## Future Artifact Policy",
        "",
        f"- eval output root: `{artifacts['eval_output_root']}`",
        f"- allowed raw outputs: `{artifacts['allowed_raw_outputs']}`",
        f"- forbidden outputs include: `{artifacts['forbidden_outputs']}`",
        "",
        "## Provenance",
        "",
        f"- analysis packet sha256: `{packet['provenance']['analysis_packet_sha256']}`",
        f"- guard packet sha256: `{packet['provenance']['guard_packet_sha256']}`",
        f"- verification packet sha256: `{packet['provenance']['verification_packet_sha256']}`",
        f"- request packet sha256: `{packet['provenance']['request_packet_sha256']}`",
        f"- execution audit packet sha256: `{packet['provenance']['execution_audit_packet_sha256']}`",
        f"- execution audit commit: `{packet['provenance']['execution_audit_commit']}`",
        f"- analysis commit: `{packet['provenance']['analysis_commit']}`",
        "",
        "## Failures",
        "",
    ]
    if packet["failures"]:
        lines.extend(f"- {failure}" for failure in packet["failures"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "This packet only proposes and validates a future reference-eval protocol. It does not run CityFlow, load models for rollout, read traffic metric values, aggregate traffic values, rank methods, generate performance tables, or make traffic-improvement or paper-ready claims.",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=DEFAULT_PROPOSAL_DIR)
    parser.add_argument("--analysis_packet", default=DEFAULT_ANALYSIS_PACKET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet = build_reference_eval_proposal_packet(out_dir=args.out_dir, analysis_packet=args.analysis_packet)
    print(json.dumps({"overall_pass": packet["overall_pass"], "failure_count": len(packet["failures"])}, sort_keys=True))
    if not packet["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
