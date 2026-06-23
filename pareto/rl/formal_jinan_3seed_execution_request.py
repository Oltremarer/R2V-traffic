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
from pareto.rl.formal_jinan_3seed_execution_guard import FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE


DEFAULT_GUARD_PACKET = (
    "docs/pro_reviews/pareto_ppo_formal_jinan_3seed_execution_guard_2026-06-01/"
    "formal_jinan_3seed_execution_guard.json"
)
DEFAULT_VERIFICATION_PACKET = (
    "docs/pro_reviews/pareto_ppo_formal_jinan_3seed_real_artifact_hash_verification_2026-06-01/"
    "formal_jinan_3seed_real_artifact_hash_verification.json"
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _no_execution_packet(packet: dict[str, Any]) -> bool:
    return (
        packet.get("formal_execution_allowed_now") is False
        and packet.get("formal_experiment_execution_in_this_packet") is False
    )


def _check_guard_packet(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    file_sha256 = sha256_file(path)
    passed = (
        payload.get("packet_type") == "formal_jinan_3seed_execution_guard"
        and payload.get("overall_pass") is True
        and _no_execution_packet(payload)
        and len(payload.get("run_manifest") or []) == 12
    )
    return {
        "pass": passed,
        "path": str(path),
        "file_sha256": file_sha256,
        "packet_type": payload.get("packet_type"),
        "overall_pass": payload.get("overall_pass"),
        "formal_execution_allowed_now": payload.get("formal_execution_allowed_now"),
        "formal_experiment_execution_in_this_packet": payload.get("formal_experiment_execution_in_this_packet"),
        "manifest_rows": len(payload.get("run_manifest") or []),
        "failure": None if passed else "guard packet is not a passing 12-run guard-only packet",
    }


def _check_verification_packet(path: Path, guard_sha256: str) -> dict[str, Any]:
    payload = _read_json(path)
    file_sha256 = sha256_file(path)
    guard_check = (payload.get("artifact_checks") or {}).get("guard_packet") or {}
    guard_sha_recorded = guard_check.get("file_sha256")
    guard_expected_recorded = guard_check.get("expected_file_sha256")
    passed = (
        payload.get("packet_type") == "formal_jinan_3seed_real_artifact_hash_verification"
        and payload.get("overall_pass") is True
        and _no_execution_packet(payload)
        and guard_sha_recorded == guard_sha256
        and guard_expected_recorded == guard_sha256
    )
    return {
        "pass": passed,
        "path": str(path),
        "file_sha256": file_sha256,
        "packet_type": payload.get("packet_type"),
        "overall_pass": payload.get("overall_pass"),
        "formal_execution_allowed_now": payload.get("formal_execution_allowed_now"),
        "formal_experiment_execution_in_this_packet": payload.get("formal_experiment_execution_in_this_packet"),
        "guard_packet_file_sha256_recorded": guard_sha_recorded,
        "guard_packet_expected_file_sha256_recorded": guard_expected_recorded,
        "failure": None if passed else "verification packet does not anchor the guard packet sha256",
    }


def build_execution_request_packet(
    *,
    out_dir: str | Path,
    guard_packet: str | Path = DEFAULT_GUARD_PACKET,
    verification_packet: str | Path = DEFAULT_VERIFICATION_PACKET,
    guard_build_commit: str = "0740db2",
    verification_commit: str = "8019dfb",
    verification_packet_commit: str = "unknown",
    request_commit: str = "unknown",
) -> dict[str, Any]:
    guard_path = Path(guard_packet)
    verification_path = Path(verification_packet)
    guard_check = _check_guard_packet(guard_path)
    verification_check = _check_verification_packet(verification_path, str(guard_check["file_sha256"]))
    checks = {
        "guard_packet": guard_check,
        "real_artifact_hash_verification_packet": verification_check,
        "no_execution_scope": {
            "pass": True,
            "formal_execution_allowed_now": False,
            "formal_experiment_execution_in_this_packet": False,
            "traffic_value_reading_in_this_packet": False,
            "numeric_aggregation_in_this_packet": False,
            "method_ranking_in_this_packet": False,
        },
    }
    failures = [
        f"{name}: {result.get('failure')}"
        for name, result in checks.items()
        if not result.get("pass")
    ]
    guard_payload = _read_json(guard_path)
    packet = {
        "packet_type": "formal_jinan_3seed_execution_request",
        "formal_execution_allowed_now": False,
        "formal_experiment_execution_in_this_packet": False,
        "cityflow_run_in_this_packet": False,
        "ppo_training_run_in_this_packet": False,
        "traffic_value_reading_in_this_packet": False,
        "numeric_aggregation_in_this_packet": False,
        "method_ranking_in_this_packet": False,
        "performance_table_in_this_packet": False,
        "paper_result_claim_in_this_packet": False,
        "approval_phrase_required_for_execution": FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE,
        "provenance": {
            "guard_build_commit": guard_build_commit,
            "verification_commit": verification_commit,
            "verification_packet_commit": verification_packet_commit,
            "request_commit": request_commit,
            "guard_packet": str(guard_path),
            "guard_packet_sha256": guard_check["file_sha256"],
            "verification_packet": str(verification_path),
            "verification_packet_sha256": verification_check["file_sha256"],
        },
        "request_checks": checks,
        "overall_pass": not failures,
        "failures": failures,
        "run_manifest": guard_payload.get("run_manifest") or [],
        "forbidden_until_later_analysis_gate": [
            "traffic_value_reading",
            "numeric_aggregation",
            "method_ranking",
            "performance_table",
            "best_method_claim",
            "traffic_improvement_claim",
            "paper_ready_claim",
        ],
        "next_gate": {
            "required_exact_phrase": FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE,
            "analysis_approval_requested": False,
            "ranking_approval_requested": False,
            "traffic_value_reading_approval_requested": False,
        },
    }
    output = Path(out_dir)
    _write_json(output / "formal_jinan_3seed_execution_request.json", packet)
    write_markdown(packet, output / "formal_jinan_3seed_execution_request.md")
    return packet


def write_markdown(packet: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# Formal Jinan 3-Seed Execution Request Packet",
        "",
        f"- overall_pass: `{packet['overall_pass']}`",
        f"- formal execution allowed now: `{packet['formal_execution_allowed_now']}`",
        f"- formal experiment executed in this packet: `{packet['formal_experiment_execution_in_this_packet']}`",
        f"- required exact phrase: `{packet['approval_phrase_required_for_execution']}`",
        f"- manifest rows: `{len(packet['run_manifest'])}`",
        "",
        "## Provenance",
        "",
        f"- guard_packet_sha256: `{packet['provenance']['guard_packet_sha256']}`",
        f"- verification_packet_sha256: `{packet['provenance']['verification_packet_sha256']}`",
        "",
        "## Request Checks",
    ]
    for name, result in packet["request_checks"].items():
        lines.append(f"- {name}: `{'PASS' if result.get('pass') else 'FAIL'}`")
        if result.get("file_sha256"):
            lines.append(f"  - file_sha256: `{result['file_sha256']}`")
    lines.extend(
        [
            "",
            "This packet requests the next external execution decision but does not execute CityFlow, PPO, traffic-value reading, aggregation, ranking, or performance-table generation.",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--guard_packet", default=DEFAULT_GUARD_PACKET)
    parser.add_argument("--verification_packet", default=DEFAULT_VERIFICATION_PACKET)
    parser.add_argument("--guard_build_commit", default="0740db2")
    parser.add_argument("--verification_commit", default="8019dfb")
    parser.add_argument("--verification_packet_commit", default="unknown")
    parser.add_argument("--request_commit", default="unknown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet = build_execution_request_packet(
        out_dir=args.out_dir,
        guard_packet=args.guard_packet,
        verification_packet=args.verification_packet,
        guard_build_commit=args.guard_build_commit,
        verification_commit=args.verification_commit,
        verification_packet_commit=args.verification_packet_commit,
        request_commit=args.request_commit,
    )
    print(json.dumps({"overall_pass": packet["overall_pass"], "failure_count": len(packet["failures"])}, sort_keys=True))
    if not packet["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
