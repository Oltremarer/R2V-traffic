#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.rl.formal_jinan_3seed_artifact_hash_verification import sha256_file


DEFAULT_FAILURE_AUDIT_PACKET = (
    "docs/pro_reviews/pareto_ppo_formal_jinan_3seed_reference_eval_execution_audit_2026-06-01/"
    "formal_jinan_3seed_reference_eval_execution_audit.json"
)
DEFAULT_REMEDIATION_DIR = (
    "docs/pro_reviews/pareto_ppo_formal_jinan_3seed_reference_eval_remediation_2026-06-01"
)
REMEDIATION_PACKET_TYPE = "formal_jinan_3seed_reference_eval_remediation"

FORBIDDEN_TRUE_FLAGS = (
    "reference_eval_run_in_this_packet",
    "cityflow_run_in_this_packet",
    "model_rollout_in_this_packet",
    "checkpoint_real_inference_in_this_packet",
    "traffic_result_value_reading_in_this_packet",
    "numeric_traffic_aggregation_in_this_packet",
    "method_ranking_in_this_packet",
    "performance_table_in_this_packet",
    "best_method_claim_in_this_packet",
    "traffic_improvement_claim_in_this_packet",
    "paper_result_claim_in_this_packet",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def _closure_table() -> list[dict[str, Any]]:
    return [
        {
            "id": "B1",
            "status": "closed",
            "title": "PPO obs feature encoding fails closed",
            "closure_condition": "Reference-eval env wrapper raises explicit ValueError when feature encoding fails or returns empty/non-finite obs_features.",
            "evidence": [
                "pareto/rl/formal_jinan_3seed_reference_eval_evaluator_binding.py",
                "tests/pareto/test_formal_jinan_3seed_reference_eval_evaluator_binding.py::test_env_wrapper_fails_closed_when_feature_encoder_raises",
            ],
        },
        {
            "id": "B2",
            "status": "closed",
            "title": "Common metrics reject non-finite values before packet writing",
            "closure_condition": "NaN leave_time rows are counted as incomplete and skipped; invalid endpoints, durations, and queue observations are finite-checked in the evaluator wrapper.",
            "evidence": [
                "tests/pareto/test_formal_jinan_3seed_reference_eval_evaluator_binding.py::test_env_wrapper_rejects_nonfinite_travel_time_metric",
                "tests/pareto/test_formal_jinan_3seed_reference_eval_evaluator_binding.py::test_env_wrapper_skips_incomplete_nan_leave_times_and_records_count",
                "tests/pareto/test_formal_jinan_3seed_reference_eval_evaluator_binding.py::test_env_wrapper_rejects_nonfinite_queue_observation",
            ],
        },
        {
            "id": "B3",
            "status": "closed",
            "title": "Failure audit is treated as consumed GO boundary",
            "closure_condition": "Prior reference-eval GO is explicitly marked consumed and non-reusable after the failed execution audit.",
            "evidence": ["verify_report.json: prior_go_reusable=false"],
        },
        {
            "id": "B4",
            "status": "closed",
            "title": "New reference-eval GO required before execution",
            "closure_condition": "Remediation packet disallows reference eval execution and requests a fresh exact approval phrase.",
            "evidence": ["verify_report.json: new_go_required_before_reference_eval=true"],
        },
        {
            "id": "B5",
            "status": "closed",
            "title": "Output root strategy is explicit",
            "closure_condition": "Next execution must start from an empty eval output root or an explicitly archived failed root.",
            "evidence": ["protocol.json: output_root_strategy"],
        },
        {
            "id": "B6",
            "status": "closed",
            "title": "Executor protocol preserves checkpoint recheck before evaluator",
            "closure_condition": "Executor protocol keeps checkpoint SHA recheck before evaluator invocation and records failure-stage semantics.",
            "evidence": ["executor_protocol.json: checkpoint_binding_policy"],
        },
        {
            "id": "B7",
            "status": "closed",
            "title": "Allowed outputs remain raw-only",
            "closure_condition": "Protocol permits only raw eval artifacts and forbids comparison/ranking/performance artifacts.",
            "evidence": ["protocol.json: artifact_policy"],
        },
        {
            "id": "B8",
            "status": "closed",
            "title": "No aggregation or traffic-improvement claim in remediation",
            "closure_condition": "All ranking, performance-table, best-method, traffic-improvement, and paper-result flags are false.",
            "evidence": ["verify_report.json: forbidden flags"],
        },
        {
            "id": "B9",
            "status": "closed",
            "title": "Remote verification requirements are listed before next GO",
            "closure_condition": "Review packet names targeted and remote full-test expectations before the next execution gate.",
            "evidence": ["REVIEW.md: Verification"],
        },
        {
            "id": "B10",
            "status": "closed",
            "title": "Gate chain is machine-readable",
            "closure_condition": "chain_manifest.json pins upstream packets/commits and states that the previous GO cannot be reused.",
            "evidence": ["chain_manifest.json"],
        },
    ]


def _risk_register() -> list[dict[str, Any]]:
    return [
        {
            "id": "R1",
            "status": "closed",
            "risk": "Feature encoder errors could be silently swallowed and leave PPO obs_features absent.",
            "mitigation": "Wrapper now raises explicit ValueError for encoder failures, empty features, or non-finite features.",
        },
        {
            "id": "R2",
            "status": "closed",
            "risk": "CityFlow common metrics could propagate non-finite average_travel_time.",
            "mitigation": "Wrapper skips NaN leave_time incomplete vehicles, counts them, and validates completed travel-time endpoints/durations before returning common metrics.",
        },
        {
            "id": "R3",
            "status": "closed",
            "risk": "No completed vehicles can make average_travel_time undefined.",
            "mitigation": "Wrapper hard-fails when average_travel_time has no completed vehicle travel-time support.",
        },
        {
            "id": "R4",
            "status": "closed",
            "risk": "Partial failed output root could be reused after a failed attempt.",
            "mitigation": "Protocol requires empty or archived eval output root before any new execution.",
        },
        {
            "id": "R5",
            "status": "closed",
            "risk": "A consumed GO could be accidentally reused after remediation.",
            "mitigation": "Remediation packet requires a new exact approval phrase before further reference eval.",
        },
    ]


def _protocol() -> dict[str, Any]:
    return {
        "protocol_version": "reference_eval_remediation_v1",
        "reference_eval_execution_allowed_now": False,
        "previous_go_consumed": True,
        "prior_go_reusable": False,
        "new_go_required_before_reference_eval": True,
        "output_root_strategy": {
            "must_be_empty_before_next_run": True,
            "archive_or_delete_partial_failed_root_before_next_go": True,
            "partial_failed_root_must_not_be_mixed_with_next_execution": True,
        },
        "artifact_policy": {
            "allowed_raw_outputs": [
                "metadata.json",
                "status.json",
                "eval_metrics.json",
                "eval_metrics.jsonl",
                "eval_guard_report.json",
                "stdout.txt",
                "stderr.txt",
            ],
            "forbidden_outputs": [
                "performance_table",
                "ranking",
                "leaderboard",
                "best_method",
                "traffic_improvement",
                "paper_result",
            ],
        },
    }


def _executor_protocol() -> dict[str, Any]:
    return {
        "executor_protocol_version": "reference_eval_executor_remediation_v1",
        "reference_eval_execution_allowed_now": False,
        "checkpoint_binding_policy": {
            "recheck_checkpoint_sha_before_evaluator": True,
            "record_failure_stage_when_eval_fails_before_guard_report": True,
            "do_not_infer_checkpoint_failure_from_missing_eval_guard_report": True,
        },
        "obs_feature_policy": {
            "encoder_failure_action": "hard_fail",
            "empty_obs_features_action": "hard_fail",
            "nonfinite_obs_features_action": "hard_fail",
            "schema_hash_drift_action": "hard_fail",
            "schema_hash_recorded_in_observation": True,
        },
        "common_metric_policy": {
            "required_metrics": ["average_travel_time", "throughput", "mean_queue_length"],
            "att_definition": "completed_vehicle_mean_finite_leave_minus_enter",
            "nan_leave_time_action": "skip_and_count_incomplete",
            "incomplete_vehicle_count_recorded": True,
            "incomplete_vehicle_count_metadata_path": "metadata.json.common_metric_debug.incomplete_vehicle_count",
            "incomplete_vehicle_count_guard_report_path": (
                "eval_guard_report.json.common_metric_debug.incomplete_vehicle_count"
            ),
            "nonfinite_metric_action": "hard_fail",
            "undefined_average_travel_time_action": "hard_fail",
            "negative_travel_time_action": "hard_fail",
        },
        "metric_window_policy": {
            "queue_observations_reset_on_env_reset": True,
            "mean_queue_length_scope": "current_eval_episode_window",
        },
        "ppo_preference_policy": {
            "default_preference_locked": True,
            "default_preference": [0.25, 0.25, 0.25, 0.25],
            "preference_sum_normalized": True,
            "stochastic_preference_sampling_allowed": False,
        },
        "reference_policy_policy": {
            "encoder_required": False,
            "obs_feature_schema_required": False,
            "observation_source": "llmlight_state_list_only",
        },
    }


def _chain_manifest(
    *,
    failure_audit_packet: Path,
    failure_audit: dict[str, Any],
    remediation_commit: str,
) -> dict[str, Any]:
    provenance = failure_audit.get("provenance") or {}
    return {
        "chain_manifest_version": "reference_eval_remediation_chain_v1",
        "previous_go_consumed": True,
        "prior_go_reusable": False,
        "new_go_required_before_reference_eval": True,
        "failure_audit_packet": str(failure_audit_packet),
        "failure_audit_packet_sha256": sha256_file(failure_audit_packet),
        "failure_audit_status": failure_audit.get("audit_status"),
        "remediation_commit": remediation_commit,
        "upstream_commits": {
            "guard_commit": provenance.get("guard_commit") or "95fd805",
            "runner_commit": provenance.get("runner_commit") or "c980523",
            "adapter_commit": provenance.get("adapter_commit") or "e04d7ce",
            "previous_binding_commit": provenance.get("binding_commit") or "bc594a8",
            "binding_commit": remediation_commit,
            "failure_audit_commit": "01e6e78",
        },
        "upstream_packets": {
            "guard_packet": provenance.get("guard_packet"),
            "adapter_packet": provenance.get("adapter_packet"),
            "binding_packet": provenance.get("binding_packet"),
        },
    }


def _review_markdown(packet: dict[str, Any]) -> str:
    closures = packet["remediation_closure_table"]
    lines = [
        "# Formal Jinan 3-Seed Reference-Eval Remediation Review",
        "",
        f"- remediation_status: `{packet['remediation_status']}`",
        f"- overall_pass: `{packet['overall_pass']}`",
        f"- previous GO consumed: `{packet['prior_reference_eval_go_consumed']}`",
        f"- prior GO reusable: `{packet['prior_go_reusable']}`",
        f"- new GO required before reference eval: `{packet['new_go_required_before_reference_eval']}`",
        f"- reference eval run in this packet: `{packet['reference_eval_run_in_this_packet']}`",
        f"- CityFlow run in this packet: `{packet['cityflow_run_in_this_packet']}`",
        "",
        "## B1-B10 Closure Table",
        "",
    ]
    for row in closures:
        lines.append(f"- {row['id']} `{row['status']}`: {row['title']} - {row['closure_condition']}")
    lines.extend(
        [
            "",
            "## Risk Register",
            "",
        ]
    )
    for row in packet["risk_register"]:
        lines.append(f"- {row['id']} `{row['status']}`: {row['risk']} Mitigation: {row['mitigation']}")
    lines.extend(
        [
            "",
            "## Verification",
            "",
            "- Targeted tests must include evaluator binding fail-closed tests and remediation packet tests.",
            "- Remote full `tests/pareto` should pass before requesting a fresh execution GO.",
            "- This packet does not run reference eval, does not read traffic result values, and does not aggregate or rank methods.",
            "",
            "## Questions for External Reviewer",
            "",
            "1. Are B1-B10 closed strongly enough to request a fresh reference-eval execution gate?",
            "2. Is fail-closed average_travel_time for no completed vehicles acceptable?",
            "3. Should the next execution use a fresh empty output root or an archived failed-root replacement?",
            "4. If approved, provide a new exact GO phrase; do not authorize reuse of the consumed GO.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_reference_eval_remediation_packet(
    *,
    out_dir: str | Path = DEFAULT_REMEDIATION_DIR,
    failure_audit_packet: str | Path = DEFAULT_FAILURE_AUDIT_PACKET,
    remediation_commit: str = "unknown",
    remote_targeted_tests_result: str = "not_provided",
    remote_full_tests_result: str = "not_provided",
) -> dict[str, Any]:
    audit_path = Path(failure_audit_packet)
    failure_audit = _read_json(audit_path)
    failures: list[str] = []
    if failure_audit.get("packet_type") != "formal_jinan_3seed_reference_eval_execution_audit":
        failures.append("failure_audit_packet has wrong packet_type")
    if failure_audit.get("audit_status") != "FAIL":
        failures.append("failure_audit_packet must be a failed execution audit")

    closures = _closure_table()
    open_blockers = [row["id"] for row in closures if row["status"] != "closed"]
    if open_blockers:
        failures.append(f"open blockers remain: {open_blockers}")
    protocol = _protocol()
    executor_protocol = _executor_protocol()
    chain_manifest = _chain_manifest(
        failure_audit_packet=audit_path,
        failure_audit=failure_audit,
        remediation_commit=remediation_commit,
    )
    packet = {
        "packet_type": REMEDIATION_PACKET_TYPE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "remediation_status": "PASS" if not failures else "FAIL",
        "overall_pass": not failures,
        "failures": failures,
        "reference_eval_run_in_this_packet": False,
        "cityflow_run_in_this_packet": False,
        "model_rollout_in_this_packet": False,
        "checkpoint_real_inference_in_this_packet": False,
        "traffic_result_value_reading_in_this_packet": False,
        "numeric_traffic_aggregation_in_this_packet": False,
        "method_ranking_in_this_packet": False,
        "performance_table_in_this_packet": False,
        "best_method_claim_in_this_packet": False,
        "traffic_improvement_claim_in_this_packet": False,
        "paper_result_claim_in_this_packet": False,
        "prior_reference_eval_go_consumed": True,
        "prior_go_reusable": False,
        "new_go_required_before_reference_eval": True,
        "reference_eval_execution_allowed_now": False,
        "failure_audit_summary": {
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
            "audit_status": failure_audit.get("audit_status"),
            "failures": failure_audit.get("failures") or [],
            "attempted_run_count": len(failure_audit.get("runs") or []),
        },
        "remediation_closure_table": closures,
        "closure_summary": {
            "closed_count": len(closures) - len(open_blockers),
            "open_blockers": open_blockers,
        },
        "risk_register": _risk_register(),
        "protocol": protocol,
        "executor_protocol": executor_protocol,
        "chain_manifest": chain_manifest,
        "verify_report": {
            "overall_pass": not failures,
            "reference_eval_execution_allowed_now": False,
            "prior_go_reusable": False,
            "new_go_required_before_reference_eval": True,
            "closed_blockers": [row["id"] for row in closures if row["status"] == "closed"],
            "open_blockers": open_blockers,
            "script_sha256": sha256_file(Path(__file__)),
            "verification_evidence": {
                "remote_targeted_tests_result": remote_targeted_tests_result,
                "remote_full_tests_result": remote_full_tests_result,
            },
        },
        "next_expected_packet": "external_reviewer_remediation_review_before_any_reference_eval_execution",
    }
    validate_reference_eval_remediation_packet(packet)

    output = Path(out_dir)
    _write_json(output / "verify_report.json", packet)
    _write_json(output / "protocol.json", protocol)
    _write_json(output / "executor_protocol.json", executor_protocol)
    _write_json(output / "chain_manifest.json", chain_manifest)
    _write_text(output / "REVIEW.md", _review_markdown(packet))
    _write_text(
        output / "command.txt",
        (
            "python pareto/rl/formal_jinan_3seed_reference_eval_remediation.py "
            f"--out_dir {output} --failure_audit_packet {audit_path} "
            f"--remediation_commit {remediation_commit}\n"
            f"script_sha256={packet['verify_report']['script_sha256']}\n"
        ),
    )
    return packet


def validate_reference_eval_remediation_packet(packet: dict[str, Any]) -> None:
    if packet.get("packet_type") != REMEDIATION_PACKET_TYPE:
        raise ValueError(f"packet_type must be {REMEDIATION_PACKET_TYPE}")
    for key in FORBIDDEN_TRUE_FLAGS:
        if packet.get(key) is not False:
            raise ValueError(f"{key} must be false")
    if packet.get("prior_reference_eval_go_consumed") is not True:
        raise ValueError("prior_reference_eval_go_consumed must be true")
    if packet.get("prior_go_reusable") is not False:
        raise ValueError("prior_go_reusable must be false")
    if packet.get("new_go_required_before_reference_eval") is not True:
        raise ValueError("new_go_required_before_reference_eval must be true")
    if packet.get("reference_eval_execution_allowed_now") is not False:
        raise ValueError("reference_eval_execution_allowed_now must be false")
    if packet.get("overall_pass") is not (not packet.get("failures")):
        raise ValueError("overall_pass must match failures")
    verify = packet.get("verify_report") or {}
    if verify.get("overall_pass") is not packet.get("overall_pass"):
        raise ValueError("verify_report.overall_pass must match packet overall_pass")
    if verify.get("reference_eval_execution_allowed_now") is not False:
        raise ValueError("verify_report must not allow reference eval execution")
    closures = packet.get("remediation_closure_table") or []
    if len(closures) < 10:
        raise ValueError("remediation_closure_table must include B1-B10")
    if any(row.get("status") != "closed" for row in closures):
        raise ValueError("all remediation blockers must be closed before PASS")
    protocol = packet.get("protocol") or {}
    if (protocol.get("output_root_strategy") or {}).get("must_be_empty_before_next_run") is not True:
        raise ValueError("output root strategy must require an empty root before next run")
    executor = packet.get("executor_protocol") or {}
    obs_policy = executor.get("obs_feature_policy") or {}
    if obs_policy.get("schema_hash_drift_action") != "hard_fail":
        raise ValueError("obs feature schema hash drift must hard-fail")
    metric_policy = executor.get("common_metric_policy") or {}
    if metric_policy.get("incomplete_vehicle_count_metadata_path") != (
        "metadata.json.common_metric_debug.incomplete_vehicle_count"
    ):
        raise ValueError("incomplete vehicle count metadata path must be locked")
    metric_window = executor.get("metric_window_policy") or {}
    if metric_window.get("queue_observations_reset_on_env_reset") is not True:
        raise ValueError("queue observations must reset on env reset")
    preference_policy = executor.get("ppo_preference_policy") or {}
    if preference_policy.get("default_preference_locked") is not True:
        raise ValueError("PPO default preference must be locked")
    reference_policy = executor.get("reference_policy_policy") or {}
    if reference_policy.get("encoder_required") is not False:
        raise ValueError("reference policies must not require encoder")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=DEFAULT_REMEDIATION_DIR)
    parser.add_argument("--failure_audit_packet", default=DEFAULT_FAILURE_AUDIT_PACKET)
    parser.add_argument("--remediation_commit", default="unknown")
    parser.add_argument("--remote_targeted_tests_result", default="not_provided")
    parser.add_argument("--remote_full_tests_result", default="not_provided")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet = build_reference_eval_remediation_packet(
        out_dir=args.out_dir,
        failure_audit_packet=args.failure_audit_packet,
        remediation_commit=args.remediation_commit,
        remote_targeted_tests_result=args.remote_targeted_tests_result,
        remote_full_tests_result=args.remote_full_tests_result,
    )
    print(json.dumps({"overall_pass": packet["overall_pass"], "failure_count": len(packet["failures"])}, sort_keys=True))
    if not packet["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
