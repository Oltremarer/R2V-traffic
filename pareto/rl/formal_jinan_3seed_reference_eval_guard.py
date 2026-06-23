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
from pareto.rl.formal_jinan_3seed_reference_eval_proposal import (
    DEFAULT_EVAL_OUTPUT_ROOT,
    DEFAULT_PROPOSAL_DIR,
    DEFAULT_TRAIN_RUN_ROOT,
    FUTURE_EVAL_ALLOWED_RAW_OUTPUTS,
    FUTURE_EVAL_FORBIDDEN_OUTPUTS,
    FUTURE_REFERENCE_EVAL_APPROVAL_PHRASE,
    validate_reference_eval_proposal_packet,
)


DEFAULT_PROPOSAL_PACKET = (
    f"{DEFAULT_PROPOSAL_DIR}/formal_jinan_3seed_reference_eval_proposal.json"
)
DEFAULT_GUARD_DIR = (
    "docs/pro_reviews/pareto_ppo_formal_jinan_3seed_reference_eval_guard_2026-06-01"
)
GUARD_PACKET_TYPE = "formal_jinan_3seed_reference_eval_guard"

FORBIDDEN_GUARD_TRUE_FLAGS = (
    "formal_reference_eval_execution_allowed_now",
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


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _require_under_eval_root(path: str | Path, eval_root: str | Path) -> None:
    if not _is_relative_to(Path(path), Path(eval_root)):
        raise ValueError("eval_output_dir must be under eval_output_root")


def _check_metadata(path: Path, *, method: str, seed: int) -> dict[str, Any]:
    if not path.is_file():
        return {"pass": False, "path": str(path), "failure": f"missing metadata: {path}"}
    metadata = _read_json(path)
    failures: list[str] = []
    expected = {
        "formal_jinan_3seed_execution": True,
        "scenario": "jinan",
        "traffic_file": FORMAL_JINAN_TRAFFIC_FILE,
        "cityflow_seed": seed,
        "policy_seed": seed,
        "model_seed": seed,
        "performance_claim": False,
        "method_ranking_allowed": False,
        "performance_table_allowed": False,
        "traffic_result_value_reading_executed": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            failures.append(f"metadata.{key} mismatch")
    recorded_method = metadata.get("method") or metadata.get("method_id")
    if recorded_method is not None and recorded_method != method:
        failures.append("metadata.method mismatch")
    return {
        "pass": not failures,
        "path": str(path),
        "file_sha256": sha256_file(path),
        "checked_fields": sorted(expected),
        "failure": "; ".join(failures) if failures else None,
    }


def _checkpoint_entry(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "pass": False,
            "path": str(path),
            "file_sha256": None,
            "failure": f"missing {label}: {path}",
        }
    return {
        "pass": True,
        "path": str(path),
        "file_sha256": sha256_file(path),
        "failure": None,
    }


def _build_checkpoint_manifest(train_run_root: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    root = Path(train_run_root)
    manifest: list[dict[str, Any]] = []
    failures: list[str] = []
    for seed in FORMAL_JINAN_SEEDS:
        for method in APPROVED_FORMAL_JINAN_PPO_METHODS:
            run_dir = root / f"seed{seed}" / method
            checkpoint = _checkpoint_entry(run_dir / "checkpoint_last.pt", label="checkpoint_last.pt")
            training_checkpoint = _checkpoint_entry(
                run_dir / "training_checkpoint_last.pt",
                label="training_checkpoint_last.pt",
            )
            metadata = _check_metadata(run_dir / "metadata.json", method=method, seed=int(seed))
            for name, check in (
                ("checkpoint_last.pt", checkpoint),
                ("training_checkpoint_last.pt", training_checkpoint),
                ("metadata.json", metadata),
            ):
                if not check["pass"]:
                    failures.append(f"seed{seed}/{method}/{name}: {check['failure']}")
            manifest.append(
                {
                    "run_kind": "ppo_checkpoint_eval",
                    "seed": int(seed),
                    "method": method,
                    "source_run_dir": str(run_dir),
                    "checkpoint_last_path": checkpoint["path"],
                    "checkpoint_last_sha256": checkpoint["file_sha256"],
                    "training_checkpoint_last_path": training_checkpoint["path"],
                    "training_checkpoint_last_sha256": training_checkpoint["file_sha256"],
                    "metadata_path": metadata["path"],
                    "metadata_sha256": metadata.get("file_sha256"),
                    "checkpoint_load_required_before_eval": True,
                    "checkpoint_to_eval_binding_locked": True,
                }
            )
        for baseline in REFERENCE_ONLY_METHODS:
            manifest.append(
                {
                    "run_kind": "reference_policy_eval",
                    "seed": int(seed),
                    "method": baseline,
                    "source_run_dir": None,
                    "checkpoint_last_path": None,
                    "checkpoint_last_sha256": None,
                    "training_checkpoint_last_path": None,
                    "training_checkpoint_last_sha256": None,
                    "same_eval_protocol_required": True,
                    "deterministic_reference_policy_required": True,
                }
            )
    return manifest, failures


def _runtime_eval_policy() -> dict[str, Any]:
    return {
        "ppo_action_selection": "argmax_deterministic",
        "reference_action_selection": "deterministic_policy",
        "stochastic_sampling_allowed": False,
        "exploration_noise_allowed": False,
        "temperature": 0.0,
        "cityflow_seed_binding": "seed_id",
        "policy_seed_binding": "seed_id",
        "model_seed_binding": "seed_id",
        "reference_policy_seed_binding": "seed_id",
        "min_action_time": 30,
    }


def validate_reference_eval_output_allowlist(
    run_dir: str | Path,
    *,
    eval_root: str | Path = DEFAULT_EVAL_OUTPUT_ROOT,
    allowed_outputs: tuple[str, ...] = FUTURE_EVAL_ALLOWED_RAW_OUTPUTS,
    forbidden_outputs: tuple[str, ...] = FUTURE_EVAL_FORBIDDEN_OUTPUTS,
    required_outputs: tuple[str, ...] | None = None,
) -> None:
    run_path = Path(run_dir)
    _require_under_eval_root(run_path, eval_root)
    if not run_path.exists():
        return
    forbidden: list[str] = []
    non_allowlisted: list[str] = []
    nested: list[str] = []
    allowed = set(allowed_outputs)
    forbidden_names = set(forbidden_outputs)
    root_files: set[str] = set()
    for path in run_path.rglob("*"):
        rel_path = path.relative_to(run_path)
        rel = str(rel_path)
        if len(rel_path.parts) > 1:
            nested.append(rel)
            continue
        if path.is_dir():
            nested.append(rel)
            continue
        if not path.is_file():
            continue
        root_files.add(path.name)
        if path.name in forbidden_names:
            forbidden.append(rel)
        elif path.name not in allowed:
            non_allowlisted.append(rel)
    if nested:
        raise ValueError(f"nested reference-eval artifacts are forbidden: {sorted(set(nested))}")
    if forbidden:
        raise ValueError(f"forbidden reference-eval artifacts: {sorted(set(forbidden))}")
    if non_allowlisted:
        raise ValueError(f"non-allowlisted reference-eval artifacts: {sorted(set(non_allowlisted))}")
    if required_outputs is not None:
        required = set(required_outputs)
        if root_files != required:
            missing = sorted(required - root_files)
            extra = sorted(root_files - required)
            raise ValueError(
                "reference-eval artifacts must match the exact required root-level set "
                f"(missing={missing}, extra={extra})"
            )


def preflight_reference_eval_output_dir(run_dir: str | Path, *, eval_root: str | Path = DEFAULT_EVAL_OUTPUT_ROOT) -> None:
    run_path = Path(run_dir)
    _require_under_eval_root(run_path, eval_root)
    if not run_path.exists():
        return
    leftovers = sorted(str(path.relative_to(run_path)) for path in run_path.rglob("*"))
    if leftovers:
        raise ValueError(f"reference-eval output dir must be empty before execution: {leftovers}")


def build_reference_eval_guard_packet(
    *,
    out_dir: str | Path = DEFAULT_GUARD_DIR,
    proposal_packet: str | Path = DEFAULT_PROPOSAL_PACKET,
    train_run_root: str | Path = DEFAULT_TRAIN_RUN_ROOT,
    eval_output_root: str | Path = DEFAULT_EVAL_OUTPUT_ROOT,
    proposal_commit: str = "c973f9d",
) -> dict[str, Any]:
    proposal_path = Path(proposal_packet)
    proposal = _read_json(proposal_path)
    proposal_failure: str | None = None
    try:
        validate_reference_eval_proposal_packet(proposal)
    except ValueError as exc:
        proposal_failure = str(exc)
    manifest, manifest_failures = _build_checkpoint_manifest(train_run_root)
    failures: list[str] = []
    if proposal_failure:
        failures.append(f"proposal packet: {proposal_failure}")
    failures.extend(manifest_failures)
    packet = {
        "packet_type": GUARD_PACKET_TYPE,
        "guard_status": "PASS" if not failures else "FAIL",
        "overall_pass": not failures,
        "failures": failures,
        "formal_reference_eval_execution_allowed_now": False,
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
        "runtime_eval_policy": _runtime_eval_policy(),
        "future_eval_scope": {
            "scenario": "jinan",
            "traffic_file": FORMAL_JINAN_TRAFFIC_FILE,
            "seed_ids": [int(seed) for seed in FORMAL_JINAN_SEEDS],
            "methods": list(APPROVED_FORMAL_JINAN_PPO_METHODS),
            "reference_baselines": list(REFERENCE_ONLY_METHODS),
            "min_action_time": 30,
            "episodes": 5,
            "max_decision_steps_per_episode": 120,
            "sim_seconds_per_method_seed": 3600,
            "same_protocol_for_methods_and_references": True,
        },
        "future_eval_artifact_policy": {
            "eval_output_root": str(eval_output_root),
            "must_not_write_to_training_run_root": str(train_run_root),
            "allowed_raw_outputs": list(FUTURE_EVAL_ALLOWED_RAW_OUTPUTS),
            "forbidden_outputs": list(FUTURE_EVAL_FORBIDDEN_OUTPUTS),
            "artifact_allowlist_enforced_by": "validate_reference_eval_output_allowlist",
            "performance_table_allowed": False,
            "ranking_allowed": False,
        },
        "future_eval_manifest": manifest,
        "provenance": {
            "proposal_packet": str(proposal_path),
            "proposal_packet_sha256": sha256_file(proposal_path) if proposal_path.is_file() else None,
            "proposal_commit": proposal_commit,
            "analysis_packet_sha256": (proposal.get("provenance") or {}).get("analysis_packet_sha256"),
            "guard_packet_sha256": (proposal.get("provenance") or {}).get("guard_packet_sha256"),
            "verification_packet_sha256": (proposal.get("provenance") or {}).get("verification_packet_sha256"),
            "request_packet_sha256": (proposal.get("provenance") or {}).get("request_packet_sha256"),
            "execution_audit_packet_sha256": (proposal.get("provenance") or {}).get("execution_audit_packet_sha256"),
        },
        "next_gate": {
            "required_exact_phrase": FUTURE_REFERENCE_EVAL_APPROVAL_PHRASE,
            "reference_eval_execution_requested": True,
            "comparison_or_ranking_requested": False,
            "performance_table_requested": False,
            "paper_result_requested": False,
        },
    }
    if packet["overall_pass"]:
        validate_reference_eval_guard_packet(packet)
    output = Path(out_dir)
    _write_json(output / "formal_jinan_3seed_reference_eval_guard.json", packet)
    write_markdown(packet, output / "formal_jinan_3seed_reference_eval_guard.md")
    return packet


def validate_reference_eval_guard_packet(packet: dict[str, Any]) -> None:
    if packet.get("packet_type") != GUARD_PACKET_TYPE:
        raise ValueError(f"packet_type must be {GUARD_PACKET_TYPE}")
    for key in FORBIDDEN_GUARD_TRUE_FLAGS:
        if packet.get(key) is not False:
            raise ValueError(f"{key} must be false")
    policy = packet.get("runtime_eval_policy") or {}
    if policy.get("ppo_action_selection") != "argmax_deterministic":
        raise ValueError("runtime_eval_policy.ppo_action_selection must be argmax_deterministic")
    if policy.get("reference_action_selection") != "deterministic_policy":
        raise ValueError("runtime_eval_policy.reference_action_selection must be deterministic_policy")
    for key in ("stochastic_sampling_allowed", "exploration_noise_allowed"):
        if policy.get(key) is not False:
            raise ValueError(f"runtime_eval_policy.{key} must be false")
    if float(policy.get("temperature", -1.0)) != 0.0:
        raise ValueError("runtime_eval_policy.temperature must be 0.0")
    manifest = packet.get("future_eval_manifest") or []
    ppo = [run for run in manifest if run.get("run_kind") == "ppo_checkpoint_eval"]
    refs = [run for run in manifest if run.get("run_kind") == "reference_policy_eval"]
    if len(ppo) != len(FORMAL_JINAN_SEEDS) * len(APPROVED_FORMAL_JINAN_PPO_METHODS):
        raise ValueError("future_eval_manifest must include 12 PPO checkpoint eval rows")
    if len(refs) != len(FORMAL_JINAN_SEEDS) * len(REFERENCE_ONLY_METHODS):
        raise ValueError("future_eval_manifest must include 6 reference eval rows")
    for run in ppo:
        if not run.get("checkpoint_last_sha256") or not run.get("training_checkpoint_last_sha256"):
            raise ValueError("PPO eval rows must include checkpoint sha256 bindings")
    if packet.get("overall_pass") is not (not packet.get("failures")):
        raise ValueError("overall_pass must match failures")


def _manifest_row(packet: dict[str, Any], *, method: str, seed_id: int) -> dict[str, Any] | None:
    for row in packet.get("future_eval_manifest") or []:
        if row.get("method") == method and int(row.get("seed", -1)) == int(seed_id):
            return row
    return None


def validate_reference_eval_execution_request(request: dict[str, Any], guard_packet: dict[str, Any]) -> dict[str, Any]:
    validate_reference_eval_guard_packet(guard_packet)
    if request.get("approval_phrase") != FUTURE_REFERENCE_EVAL_APPROVAL_PHRASE:
        raise ValueError("exact reference-eval approval phrase is required")
    method = str(request.get("method"))
    seed_id = int(request.get("seed_id", -1))
    approved_methods = set(APPROVED_FORMAL_JINAN_PPO_METHODS) | set(REFERENCE_ONLY_METHODS)
    if method not in approved_methods:
        raise ValueError(f"method is not approved for reference eval: {method}")
    if seed_id not in FORMAL_JINAN_SEEDS:
        raise ValueError(f"seed_id is not approved for reference eval: {seed_id}")
    row = _manifest_row(guard_packet, method=method, seed_id=seed_id)
    if row is None:
        raise ValueError("method/seed is missing from guard manifest")
    expected_kind = "ppo_checkpoint_eval" if method in APPROVED_FORMAL_JINAN_PPO_METHODS else "reference_policy_eval"
    if request.get("run_kind") != expected_kind:
        raise ValueError(f"run_kind must be {expected_kind}")
    if int(request.get("min_action_time", -1)) != 30:
        raise ValueError("min_action_time must be 30")
    if int(request.get("episodes", -1)) != 5:
        raise ValueError("episodes must be 5")
    if int(request.get("max_decision_steps_per_episode", -1)) != 120:
        raise ValueError("max_decision_steps_per_episode must be 120")
    if int(request.get("cityflow_seed", -1)) != seed_id:
        raise ValueError("cityflow_seed must equal seed_id")
    if int(request.get("policy_seed", -1)) != seed_id:
        raise ValueError("policy_seed must equal seed_id")
    if int(request.get("model_seed", -1)) != seed_id:
        raise ValueError("model_seed must equal seed_id")
    if expected_kind == "reference_policy_eval" and int(request.get("reference_policy_seed", -1)) != seed_id:
        raise ValueError("reference_policy_seed must equal seed_id for reference policy eval")
    if expected_kind == "ppo_checkpoint_eval" and request.get("reference_policy_seed") is not None:
        raise ValueError("reference_policy_seed must be absent for PPO checkpoint eval")
    if request.get("stochastic_sampling_allowed") is not False:
        raise ValueError("stochastic_sampling_allowed must be false")
    if request.get("exploration_noise_allowed") is not False:
        raise ValueError("exploration_noise_allowed must be false")
    if float(request.get("temperature", -1.0)) != 0.0:
        raise ValueError("temperature must be 0.0")
    if expected_kind == "ppo_checkpoint_eval" and request.get("ppo_action_selection") != "argmax_deterministic":
        raise ValueError("ppo_action_selection must be argmax_deterministic")
    if expected_kind == "reference_policy_eval" and request.get("reference_action_selection") != "deterministic_policy":
        raise ValueError("reference_action_selection must be deterministic_policy")
    eval_root = (guard_packet.get("future_eval_artifact_policy") or {}).get("eval_output_root")
    _require_under_eval_root(request.get("eval_output_dir"), eval_root)
    expected_dir = Path(eval_root) / f"seed{seed_id}" / method
    if Path(str(request.get("eval_output_dir"))).resolve() != expected_dir.resolve():
        raise ValueError("eval_output_dir must match locked seed/method path")
    return dict(request)


def write_markdown(packet: dict[str, Any], path: str | Path) -> None:
    policy = packet["runtime_eval_policy"]
    artifacts = packet["future_eval_artifact_policy"]
    lines = [
        "# Formal Jinan 3-Seed Reference-Eval Guard",
        "",
        f"- guard_status: `{packet['guard_status']}`",
        f"- overall_pass: `{packet['overall_pass']}`",
        f"- formal reference-eval allowed now: `{packet['formal_reference_eval_execution_allowed_now']}`",
        f"- reference eval run in this packet: `{packet['reference_eval_run_in_this_packet']}`",
        f"- CityFlow run in this packet: `{packet['cityflow_run_in_this_packet']}`",
        f"- model rollout in this packet: `{packet['model_rollout_in_this_packet']}`",
        f"- traffic result value reading in this packet: `{packet['traffic_result_value_reading_in_this_packet']}`",
        f"- next required exact phrase: `{packet['next_gate']['required_exact_phrase']}`",
        "",
        "## Runtime Determinism",
        "",
        f"- PPO action selection: `{policy['ppo_action_selection']}`",
        f"- reference action selection: `{policy['reference_action_selection']}`",
        f"- stochastic sampling allowed: `{policy['stochastic_sampling_allowed']}`",
        f"- exploration noise allowed: `{policy['exploration_noise_allowed']}`",
        f"- temperature: `{policy['temperature']}`",
        f"- CityFlow seed binding: `{policy['cityflow_seed_binding']}`",
        "",
        "## Artifact Policy",
        "",
        f"- eval output root: `{artifacts['eval_output_root']}`",
        f"- training root must not be written: `{artifacts['must_not_write_to_training_run_root']}`",
        f"- allowed raw outputs: `{artifacts['allowed_raw_outputs']}`",
        f"- forbidden outputs: `{artifacts['forbidden_outputs']}`",
        "",
        "## Manifest",
        "",
        f"- rows: `{len(packet['future_eval_manifest'])}`",
        f"- PPO checkpoint rows: `{sum(1 for row in packet['future_eval_manifest'] if row['run_kind'] == 'ppo_checkpoint_eval')}`",
        f"- reference rows: `{sum(1 for row in packet['future_eval_manifest'] if row['run_kind'] == 'reference_policy_eval')}`",
        "",
        "## Provenance",
        "",
        f"- proposal packet sha256: `{packet['provenance']['proposal_packet_sha256']}`",
        f"- analysis packet sha256: `{packet['provenance']['analysis_packet_sha256']}`",
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
            "This packet only builds execution-side guards for a future reference eval. It does not run CityFlow, load checkpoints for inference, read traffic metric values, aggregate traffic values, rank methods, generate performance tables, or make traffic-improvement or paper-ready claims.",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=DEFAULT_GUARD_DIR)
    parser.add_argument("--proposal_packet", default=DEFAULT_PROPOSAL_PACKET)
    parser.add_argument("--train_run_root", default=DEFAULT_TRAIN_RUN_ROOT)
    parser.add_argument("--eval_output_root", default=DEFAULT_EVAL_OUTPUT_ROOT)
    parser.add_argument("--proposal_commit", default="c973f9d")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet = build_reference_eval_guard_packet(
        out_dir=args.out_dir,
        proposal_packet=args.proposal_packet,
        train_run_root=args.train_run_root,
        eval_output_root=args.eval_output_root,
        proposal_commit=args.proposal_commit,
    )
    print(json.dumps({"overall_pass": packet["overall_pass"], "failure_count": len(packet["failures"])}, sort_keys=True))
    if not packet["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
