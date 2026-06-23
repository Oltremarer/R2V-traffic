#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.rl.formal_jinan_3seed_artifact_hash_verification import sha256_file
from pareto.rl.formal_jinan_3seed_reference_eval_guard import (
    DEFAULT_GUARD_DIR,
    DEFAULT_PROPOSAL_PACKET,
    _read_json,
    preflight_reference_eval_output_dir,
    validate_reference_eval_execution_request,
    validate_reference_eval_guard_packet,
    validate_reference_eval_output_allowlist,
)
from pareto.rl.formal_jinan_3seed_reference_eval_proposal import (
    DEFAULT_EVAL_OUTPUT_ROOT,
    FUTURE_EVAL_ALLOWED_RAW_OUTPUTS,
    FUTURE_REFERENCE_EVAL_APPROVAL_PHRASE,
)


DEFAULT_GUARD_PACKET = f"{DEFAULT_GUARD_DIR}/formal_jinan_3seed_reference_eval_guard.json"
DEFAULT_RUNNER_DIR = (
    "docs/pro_reviews/pareto_ppo_formal_jinan_3seed_reference_eval_runner_2026-06-01"
)
RUNNER_PACKET_TYPE = "formal_jinan_3seed_reference_eval_runner"
REQUIRED_COMMON_METRICS = ("average_travel_time", "throughput", "mean_queue_length")
PPO_EVAL_PREFERENCE_POLICY = "balanced_primary_v1"
PPO_EVAL_PREFERENCE = [0.25, 0.25, 0.25, 0.25]
REQUIRED_COMMON_METRIC_DEBUG_KEYS = (
    "att_definition",
    "completed_vehicle_count",
    "incomplete_vehicle_count",
    "observed_vehicle_rows",
)
EXPECTED_ATT_DEFINITION = "completed_vehicle_mean_finite_leave_minus_enter"
RUNNER_REQUIRED_OUTPUTS = (
    "metadata.json",
    "status.json",
    "eval_metrics.json",
    "eval_metrics.jsonl",
    "eval_guard_report.json",
)
FORBIDDEN_METRIC_FIELDS = (
    "rank",
    "ranking",
    "leaderboard",
    "best_method",
    "improvement",
    "traffic_improvement",
    "paper_result",
)
FORBIDDEN_RUNNER_TRUE_FLAGS = (
    "reference_eval_run_in_this_packet",
    "cityflow_run_in_this_packet",
    "model_rollout_in_this_packet",
    "traffic_result_value_reading_in_this_packet",
    "numeric_traffic_aggregation_in_this_packet",
    "method_ranking_in_this_packet",
    "performance_table_in_this_packet",
    "best_method_claim_in_this_packet",
    "traffic_improvement_claim_in_this_packet",
    "paper_result_claim_in_this_packet",
)

Evaluator = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _manifest_row(packet: dict[str, Any], *, method: str, seed_id: int) -> dict[str, Any]:
    for row in packet.get("future_eval_manifest") or []:
        if row.get("method") == method and int(row.get("seed", -1)) == int(seed_id):
            return dict(row)
    raise ValueError("method/seed is missing from guard manifest")


def _load_guard_packet(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    validate_reference_eval_guard_packet(payload)
    return payload


def validate_common_metric_payload(metrics: dict[str, Any]) -> dict[str, float]:
    failures: list[str] = []
    for key in FORBIDDEN_METRIC_FIELDS:
        if key in metrics:
            failures.append(f"forbidden metric field: {key}")
    normalized: dict[str, float] = {}
    for key in REQUIRED_COMMON_METRICS:
        if key not in metrics:
            failures.append(f"missing common metric: {key}")
            continue
        try:
            value = float(metrics[key])
        except (TypeError, ValueError):
            failures.append(f"non-numeric common metric: {key}")
            continue
        if not math.isfinite(value):
            failures.append(f"non-finite common metric: {key}")
            continue
        normalized[key] = value
    if failures:
        raise ValueError("; ".join(failures))
    return normalized


def validate_common_metric_debug(debug_payload: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    missing = [key for key in REQUIRED_COMMON_METRIC_DEBUG_KEYS if key not in debug_payload]
    if missing:
        failures.append(f"missing common_metric_debug keys: {missing}")
    if debug_payload.get("att_definition") != EXPECTED_ATT_DEFINITION:
        failures.append("common_metric_debug.att_definition mismatch")
    normalized: dict[str, Any] = {"att_definition": debug_payload.get("att_definition")}
    for key in ("completed_vehicle_count", "incomplete_vehicle_count", "observed_vehicle_rows"):
        if key not in debug_payload:
            continue
        try:
            value = int(debug_payload[key])
        except (TypeError, ValueError):
            failures.append(f"common_metric_debug.{key} must be an integer")
            continue
        if value < 0:
            failures.append(f"common_metric_debug.{key} must be non-negative")
            continue
        normalized[key] = value
    completed = normalized.get("completed_vehicle_count")
    observed = normalized.get("observed_vehicle_rows")
    incomplete = normalized.get("incomplete_vehicle_count")
    if isinstance(completed, int) and completed <= 0:
        failures.append("common_metric_debug.completed_vehicle_count must be positive for successful eval")
    if isinstance(observed, int) and isinstance(completed, int) and isinstance(incomplete, int):
        if observed < completed + incomplete:
            failures.append("common_metric_debug.observed_vehicle_rows must cover completed and incomplete vehicles")
    if failures:
        raise ValueError("; ".join(failures))
    return normalized


def _extract_evaluator_payload(evaluator_payload: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    if "common_metrics" not in evaluator_payload:
        raise ValueError("evaluator payload must include common_metrics and common_metric_debug")
    metrics_payload = evaluator_payload.get("common_metrics")
    if not isinstance(metrics_payload, dict):
        raise ValueError("common_metrics must be an object")
    debug_payload = evaluator_payload.get("common_metric_debug")
    if not isinstance(debug_payload, dict) or not debug_payload:
        raise ValueError("common_metric_debug must be a nonempty object")
    return validate_common_metric_payload(metrics_payload), validate_common_metric_debug(debug_payload)


def _seed_binding(validated: dict[str, Any]) -> dict[str, Any]:
    seed_id = int(validated["seed_id"])
    binding: dict[str, Any] = {
        "cityflow_seed": int(validated["cityflow_seed"]),
        "policy_seed": int(validated["policy_seed"]),
        "model_seed": int(validated["model_seed"]),
        "seed_binding": "cityflow_seed=policy_seed=model_seed=seed_id",
    }
    if validated["run_kind"] == "reference_policy_eval":
        binding["reference_policy_seed"] = int(validated["reference_policy_seed"])
        binding["reference_policy_seed_binding"] = "reference_policy_seed=seed_id"
    else:
        binding["reference_policy_seed"] = None
        binding["reference_policy_seed_binding"] = "not_applicable_for_ppo_checkpoint_eval"
    if binding["cityflow_seed"] != seed_id or binding["policy_seed"] != seed_id or binding["model_seed"] != seed_id:
        raise ValueError("seed binding drift after validation")
    if validated["run_kind"] == "reference_policy_eval" and binding["reference_policy_seed"] != seed_id:
        raise ValueError("reference policy seed binding drift after validation")
    return binding


def _eval_preference_binding(validated: dict[str, Any]) -> dict[str, Any]:
    if validated["run_kind"] == "reference_policy_eval":
        if validated.get("eval_preference") is not None:
            raise ValueError("eval_preference must be omitted for reference_policy_eval")
        return {
            "eval_preference_policy": "not_applicable_reference_policy",
            "eval_preference": None,
        }
    observed = validated.get("eval_preference", PPO_EVAL_PREFERENCE)
    try:
        values = [float(value) for value in observed]
    except (TypeError, ValueError) as exc:
        raise ValueError("eval_preference must be numeric weights") from exc
    if values != PPO_EVAL_PREFERENCE:
        raise ValueError("eval_preference must match balanced_primary_v1")
    return {
        "eval_preference_policy": PPO_EVAL_PREFERENCE_POLICY,
        "eval_preference": list(PPO_EVAL_PREFERENCE),
    }


def _metadata_binding_failures(metadata: Any, *, label: str, method: str, seed_id: int) -> list[str]:
    if not isinstance(metadata, dict):
        return [f"{label} metadata must be an object"]
    failures: list[str] = []
    expected_metadata = {
        "formal_jinan_3seed_execution": True,
        "scenario": "jinan",
        "traffic_file": "anon_3_4_jinan_real.json",
        "cityflow_seed": seed_id,
        "policy_seed": seed_id,
        "model_seed": seed_id,
        "performance_claim": False,
        "method_ranking_allowed": False,
        "performance_table_allowed": False,
        "traffic_result_value_reading_executed": False,
    }
    for key, expected_value in expected_metadata.items():
        if metadata.get(key) != expected_value:
            failures.append(f"{label} metadata.{key} mismatch")
    recorded_method = metadata.get("method") or metadata.get("method_id")
    if recorded_method is not None and recorded_method != method:
        failures.append(f"{label} metadata.method mismatch")
    return failures


def _load_checkpoint_payload_metadata(path: Path, *, label: str) -> dict[str, Any]:
    try:
        import torch

        payload = torch.load(path, map_location=torch.device("cpu"))
    except Exception as exc:
        raise ValueError(f"{label} payload load failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} payload must be an object")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{label} payload metadata must be an object")
    return metadata


def recheck_checkpoint_sha_binding(request: dict[str, Any], guard_packet: dict[str, Any]) -> dict[str, Any]:
    validate_reference_eval_guard_packet(guard_packet)
    method = str(request.get("method"))
    seed_id = int(request.get("seed_id", -1))
    row = _manifest_row(guard_packet, method=method, seed_id=seed_id)
    if row.get("run_kind") == "reference_policy_eval":
        return {
            "pass": True,
            "checkpoint_sha_rechecked": False,
            "run_kind": "reference_policy_eval",
            "method": method,
            "seed": seed_id,
        }

    failures: list[str] = []
    checks: dict[str, Any] = {
        "checkpoint_sha_rechecked": True,
        "run_kind": "ppo_checkpoint_eval",
        "method": method,
        "seed": seed_id,
    }
    for path_key, sha_key, label in (
        ("checkpoint_last_path", "checkpoint_last_sha256", "checkpoint_last.pt"),
        ("training_checkpoint_last_path", "training_checkpoint_last_sha256", "training_checkpoint_last.pt"),
        ("metadata_path", "metadata_sha256", "metadata.json"),
    ):
        path = Path(str(row.get(path_key)))
        expected = row.get(sha_key)
        if not path.is_file():
            failures.append(f"missing {label}: {path}")
            continue
        observed = sha256_file(path)
        checks[sha_key] = observed
        if observed != expected:
            failures.append(f"{label} sha256 mismatch")
            continue
        if label == "metadata.json":
            metadata = _read_json(path)
            failures.extend(_metadata_binding_failures(metadata, label=label, method=method, seed_id=seed_id))
        else:
            checkpoint_metadata = _load_checkpoint_payload_metadata(path, label=label)
            failures.extend(
                _metadata_binding_failures(checkpoint_metadata, label=label, method=method, seed_id=seed_id)
            )
    if failures:
        raise ValueError("; ".join(failures))
    checks["pass"] = True
    return checks


def execute_guarded_reference_eval_request(
    request: dict[str, Any],
    guard_packet: dict[str, Any],
    *,
    evaluator: Evaluator,
) -> dict[str, Any]:
    """Guarded execution shell.

    The evaluator is injected so tests can prove the runtime wiring without running
    CityFlow. A future approved eval can pass the real CityFlow evaluator through
    this same guarded entrypoint.
    """
    validate_reference_eval_guard_packet(guard_packet)
    validated = validate_reference_eval_execution_request(request, guard_packet)
    manifest_row = _manifest_row(
        guard_packet,
        method=str(validated["method"]),
        seed_id=int(validated["seed_id"]),
    )
    checkpoint_check = recheck_checkpoint_sha_binding(validated, guard_packet)
    eval_root = (guard_packet.get("future_eval_artifact_policy") or {}).get("eval_output_root", DEFAULT_EVAL_OUTPUT_ROOT)
    run_dir = Path(str(validated["eval_output_dir"]))
    preflight_reference_eval_output_dir(run_dir, eval_root=eval_root)
    metrics, common_metric_debug = _extract_evaluator_payload(evaluator(dict(validated), dict(manifest_row)))

    run_dir.mkdir(parents=True, exist_ok=True)
    seed_binding = _seed_binding(validated)
    preference_binding = _eval_preference_binding(validated)
    common = {
        "method": validated["method"],
        "seed": int(validated["seed_id"]),
        **seed_binding,
        **preference_binding,
        "run_kind": validated["run_kind"],
        "scenario": "jinan",
        "traffic_file": "anon_3_4_jinan_real.json",
        "metric_interpretation": "raw_common_metric_no_ranking_no_claim",
        "method_ranking_executed": False,
        "performance_table_generated": False,
        "best_method_claim_generated": False,
        "traffic_improvement_claim_generated": False,
        "paper_result_claim_generated": False,
    }
    metric_row = {**common, **metrics}
    guard_report = {
        "status": "PASS",
        "approval_phrase": FUTURE_REFERENCE_EVAL_APPROVAL_PHRASE,
        "runtime_guard_hooks": [
            "validate_reference_eval_execution_request",
            "recheck_checkpoint_sha_binding",
            "preflight_reference_eval_output_dir",
            "validate_common_metric_debug",
            "validate_reference_eval_output_allowlist",
        ],
        "checkpoint_check": checkpoint_check,
        "artifact_allowlist_checked": True,
        "forbidden_claim_flags": {
            "method_ranking_executed": False,
            "performance_table_generated": False,
            "best_method_claim_generated": False,
            "traffic_improvement_claim_generated": False,
            "paper_result_claim_generated": False,
        },
        "common_metric_debug": common_metric_debug,
        "seed_binding": seed_binding,
        **preference_binding,
    }
    metadata = {
        **common,
        "formal_jinan_3seed_reference_eval": True,
        "same_protocol_for_methods_and_references": True,
        "min_action_time": int(validated["min_action_time"]),
        "episodes": int(validated["episodes"]),
        "max_decision_steps_per_episode": int(validated["max_decision_steps_per_episode"]),
        "stochastic_sampling_allowed": False,
        "exploration_noise_allowed": False,
        "temperature": 0.0,
        "common_metric_debug": common_metric_debug,
        "seed_binding": seed_binding,
    }
    status = {
        "status": "PASS",
        "metrics_written": True,
        "checkpoint_sha_rechecked": bool(checkpoint_check["checkpoint_sha_rechecked"]),
        "artifact_allowlist_pass": True,
        "common_metric_debug_recorded": True,
        "seed_binding_recorded": True,
        "cityflow_seed": seed_binding["cityflow_seed"],
        "policy_seed": seed_binding["policy_seed"],
        "model_seed": seed_binding["model_seed"],
        "reference_policy_seed": seed_binding["reference_policy_seed"],
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(run_dir / "status.json", status)
    _write_json(run_dir / "eval_metrics.json", metric_row)
    _write_jsonl(run_dir / "eval_metrics.jsonl", metric_row)
    _write_json(run_dir / "eval_guard_report.json", guard_report)
    validate_reference_eval_output_allowlist(
        run_dir,
        eval_root=eval_root,
        required_outputs=RUNNER_REQUIRED_OUTPUTS,
    )
    return {
        "status": "PASS",
        "run_dir": str(run_dir),
        "checkpoint_sha_rechecked": bool(checkpoint_check["checkpoint_sha_rechecked"]),
        "metrics": metrics,
    }


def build_reference_eval_runner_packet(
    *,
    out_dir: str | Path = DEFAULT_RUNNER_DIR,
    guard_packet_path: str | Path = DEFAULT_GUARD_PACKET,
    guard_packet: dict[str, Any] | None = None,
    guard_commit: str = "95fd805",
) -> dict[str, Any]:
    guard_path = Path(guard_packet_path)
    payload = dict(guard_packet) if guard_packet is not None else _load_guard_packet(guard_path)
    validate_reference_eval_guard_packet(payload)
    failures: list[str] = []
    if guard_path.is_file() and guard_packet is not None:
        guard_sha = sha256_file(guard_path)
    elif guard_path.is_file():
        guard_sha = sha256_file(guard_path)
    else:
        guard_sha = None
        failures.append(f"missing guard packet path: {guard_path}")

    packet = {
        "packet_type": RUNNER_PACKET_TYPE,
        "runner_status": "PASS" if not failures else "FAIL",
        "overall_pass": not failures,
        "failures": failures,
        "reference_eval_run_in_this_packet": False,
        "cityflow_run_in_this_packet": False,
        "model_rollout_in_this_packet": False,
        "traffic_result_value_reading_in_this_packet": False,
        "numeric_traffic_aggregation_in_this_packet": False,
        "method_ranking_in_this_packet": False,
        "performance_table_in_this_packet": False,
        "best_method_claim_in_this_packet": False,
        "traffic_improvement_claim_in_this_packet": False,
        "paper_result_claim_in_this_packet": False,
        "runner_entrypoint": "execute_guarded_reference_eval_request",
        "runner_module": "pareto.rl.formal_jinan_3seed_reference_eval_runner",
        "runtime_guard_hooks": [
            "validate_reference_eval_execution_request",
            "recheck_checkpoint_sha_binding",
            "preflight_reference_eval_output_dir",
            "validate_common_metric_debug",
            "validate_reference_eval_output_allowlist",
        ],
        "guarded_metric_writer": {
            "required_common_metrics": list(REQUIRED_COMMON_METRICS),
            "allowed_outputs": list(FUTURE_EVAL_ALLOWED_RAW_OUTPUTS),
            "forbidden_metric_fields": list(FORBIDDEN_METRIC_FIELDS),
            "ranking_allowed": False,
            "performance_table_allowed": False,
            "traffic_improvement_claim_allowed": False,
            "paper_result_claim_allowed": False,
        },
        "future_execution_request": {
            "required_exact_phrase": FUTURE_REFERENCE_EVAL_APPROVAL_PHRASE,
            "real_cityflow_evaluator_required_after_gate": True,
            "this_packet_executes_real_evaluator": False,
        },
        "provenance": {
            "guard_packet": str(guard_path),
            "guard_packet_sha256": guard_sha,
            "guard_commit": guard_commit,
            "proposal_packet_sha256": (payload.get("provenance") or {}).get("proposal_packet_sha256"),
            "analysis_packet_sha256": (payload.get("provenance") or {}).get("analysis_packet_sha256"),
            "upstream_guard_packet_sha256": (payload.get("provenance") or {}).get("guard_packet_sha256"),
            "verification_packet_sha256": (payload.get("provenance") or {}).get("verification_packet_sha256"),
            "request_packet_sha256": (payload.get("provenance") or {}).get("request_packet_sha256"),
            "execution_audit_packet_sha256": (payload.get("provenance") or {}).get("execution_audit_packet_sha256"),
        },
    }
    if packet["overall_pass"]:
        validate_reference_eval_runner_packet(packet)
    output = Path(out_dir)
    _write_json(output / "formal_jinan_3seed_reference_eval_runner.json", packet)
    write_markdown(packet, output / "formal_jinan_3seed_reference_eval_runner.md")
    return packet


def validate_reference_eval_runner_packet(packet: dict[str, Any]) -> None:
    if packet.get("packet_type") != RUNNER_PACKET_TYPE:
        raise ValueError(f"packet_type must be {RUNNER_PACKET_TYPE}")
    for key in FORBIDDEN_RUNNER_TRUE_FLAGS:
        if packet.get(key) is not False:
            raise ValueError(f"{key} must be false")
    if packet.get("runner_entrypoint") != "execute_guarded_reference_eval_request":
        raise ValueError("runner_entrypoint must be execute_guarded_reference_eval_request")
    hooks = packet.get("runtime_guard_hooks") or []
    for hook in (
        "validate_reference_eval_execution_request",
        "recheck_checkpoint_sha_binding",
        "preflight_reference_eval_output_dir",
        "validate_common_metric_debug",
        "validate_reference_eval_output_allowlist",
    ):
        if hook not in hooks:
            raise ValueError(f"missing runtime guard hook: {hook}")
    writer = packet.get("guarded_metric_writer") or {}
    if tuple(writer.get("required_common_metrics") or []) != REQUIRED_COMMON_METRICS:
        raise ValueError("required_common_metrics must match locked common metrics")
    if writer.get("ranking_allowed") is not False:
        raise ValueError("ranking_allowed must be false")
    if packet.get("overall_pass") is not (not packet.get("failures")):
        raise ValueError("overall_pass must match failures")


def write_markdown(packet: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# Formal Jinan 3-Seed Reference-Eval Runner Guard",
        "",
        f"- runner_status: `{packet['runner_status']}`",
        f"- overall_pass: `{packet['overall_pass']}`",
        f"- reference eval run in this packet: `{packet['reference_eval_run_in_this_packet']}`",
        f"- CityFlow run in this packet: `{packet['cityflow_run_in_this_packet']}`",
        f"- model rollout in this packet: `{packet['model_rollout_in_this_packet']}`",
        f"- traffic result value reading in this packet: `{packet['traffic_result_value_reading_in_this_packet']}`",
        f"- numeric aggregation in this packet: `{packet['numeric_traffic_aggregation_in_this_packet']}`",
        f"- runner entrypoint: `{packet['runner_entrypoint']}`",
        "",
        "## Runtime Guard Hooks",
        "",
    ]
    lines.extend(f"- `{hook}`" for hook in packet["runtime_guard_hooks"])
    lines.extend(
        [
            "",
            "## Guarded Metric Writer",
            "",
            f"- required common metrics: `{packet['guarded_metric_writer']['required_common_metrics']}`",
            f"- allowed outputs: `{packet['guarded_metric_writer']['allowed_outputs']}`",
            f"- forbidden metric fields: `{packet['guarded_metric_writer']['forbidden_metric_fields']}`",
            f"- ranking allowed: `{packet['guarded_metric_writer']['ranking_allowed']}`",
            "",
            "## Provenance",
            "",
            f"- guard packet sha256: `{packet['provenance']['guard_packet_sha256']}`",
            f"- proposal packet sha256: `{packet['provenance']['proposal_packet_sha256']}`",
            "",
            "## Failures",
            "",
        ]
    )
    if packet["failures"]:
        lines.extend(f"- {failure}" for failure in packet["failures"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "This packet wires the future reference-eval runner to runtime guards only. It does not run CityFlow, load checkpoints for inference, read traffic metric values, aggregate traffic values, rank methods, generate performance tables, or make traffic-improvement or paper-ready claims.",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=DEFAULT_RUNNER_DIR)
    parser.add_argument("--guard_packet", default=DEFAULT_GUARD_PACKET)
    parser.add_argument("--guard_commit", default="95fd805")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet = build_reference_eval_runner_packet(
        out_dir=args.out_dir,
        guard_packet_path=args.guard_packet,
        guard_commit=args.guard_commit,
    )
    print(json.dumps({"overall_pass": packet["overall_pass"], "failure_count": len(packet["failures"])}, sort_keys=True))
    if not packet["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
