#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.rl.formal_jinan_3seed_execution_guard import (
    FILM_MODEL_HASH,
    FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE,
    OBJECTIVE_NORMALIZER_HASH,
    STATE_ENCODER_HASH,
    VECTORQ_MODEL_HASH,
)


DEFAULT_VECTOR_MODEL_DIR = "model_weights/pareto_quality/jinan/eval_consistency_remediation_v3/v3_rev15_m02_iso3_c15_u03"
DEFAULT_FILM_MODEL_DIR = "model_weights/cond_scalar/jinan/preformal_final/film_rich_v2"
DEFAULT_OBJECTIVE_NORMALIZER = "records/eval_consistency_remediation_v3/objective_norm_smoke3600.json"
DEFAULT_GUARD_PACKET = (
    "docs/pro_reviews/pareto_ppo_formal_jinan_3seed_execution_guard_2026-06-01/"
    "formal_jinan_3seed_execution_guard.json"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verify_equal(name: str, observed: str | None, expected: str) -> dict[str, Any]:
    passed = observed == expected
    return {
        "pass": passed,
        "observed": observed,
        "expected": expected,
        "failure": None if passed else f"{name} mismatch",
    }


def _state_encoder_feature_names_hash() -> tuple[str, int]:
    from pareto.rl.state_encoder import ParetoStateEncoder

    snapshot = SimpleNamespace(
        current_phase=0,
        time_this_phase=0,
        mean_speed=0.0,
        state_detail={},
        state_incoming={},
        dic_feature={},
    )
    _features, names, debug = ParetoStateEncoder("hybrid_v1").encode_snapshot(snapshot)
    return str(debug["feature_names_hash"]), len(names)


def _verify_model_file(path: Path, expected_hash: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "pass": False,
            "path": str(path),
            "observed": None,
            "expected": expected_hash,
            "failure": f"missing model file: {path}",
        }
    check = _verify_equal(path.name, sha256_file(path), expected_hash)
    check["path"] = str(path)
    return check


def _verify_objective_normalizer(path: Path, expected_hash: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "pass": False,
            "path": str(path),
            "internal_hash": None,
            "file_sha256": None,
            "expected": expected_hash,
            "failure": f"missing objective normalizer: {path}",
        }
    payload = _read_json(path)
    internal_hash = payload.get("hash")
    check = _verify_equal("objective_normalizer_hash", str(internal_hash) if internal_hash is not None else None, expected_hash)
    check.update(
        {
            "path": str(path),
            "internal_hash": internal_hash,
            "file_sha256": sha256_file(path),
        }
    )
    return check


def _verify_state_encoder(expected_hash: str) -> dict[str, Any]:
    observed, feature_count = _state_encoder_feature_names_hash()
    check = _verify_equal("state_encoder_hash", observed, expected_hash)
    check.update(
        {
            "encoder_id": "hybrid_v1",
            "feature_count": feature_count,
            "hash_kind": "feature_names_hash",
        }
    )
    return check


def _verify_guard_packet(path: Path, expected_file_sha256: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        return {"pass": False, "path": str(path), "failure": f"missing guard packet: {path}"}
    payload = _read_json(path)
    file_sha256 = sha256_file(path)
    semantic_passed = (
        payload.get("packet_type") == "formal_jinan_3seed_execution_guard"
        and payload.get("overall_pass") is True
        and payload.get("formal_execution_allowed_now") is False
        and payload.get("formal_experiment_execution_in_this_packet") is False
    )
    sha_passed = expected_file_sha256 is None or file_sha256 == expected_file_sha256
    if not semantic_passed:
        failure = "guard packet is not a passing guard-only packet"
    elif not sha_passed:
        failure = "guard packet file_sha256 mismatch"
    else:
        failure = None
    return {
        "pass": semantic_passed and sha_passed,
        "path": str(path),
        "packet_type": payload.get("packet_type"),
        "overall_pass": payload.get("overall_pass"),
        "formal_execution_allowed_now": payload.get("formal_execution_allowed_now"),
        "formal_experiment_execution_in_this_packet": payload.get("formal_experiment_execution_in_this_packet"),
        "file_sha256": file_sha256,
        "expected_file_sha256": expected_file_sha256,
        "failure": failure,
    }


def build_artifact_hash_verification_packet(
    *,
    out_dir: str | Path,
    guard_packet: str | Path = DEFAULT_GUARD_PACKET,
    guard_packet_hash: str | None = None,
    vector_model_dir: str | Path = DEFAULT_VECTOR_MODEL_DIR,
    vector_model_hash: str = VECTORQ_MODEL_HASH,
    film_model_dir: str | Path = DEFAULT_FILM_MODEL_DIR,
    film_model_hash: str = FILM_MODEL_HASH,
    objective_normalizer: str | Path = DEFAULT_OBJECTIVE_NORMALIZER,
    objective_normalizer_hash: str = OBJECTIVE_NORMALIZER_HASH,
    state_encoder_hash: str = STATE_ENCODER_HASH,
    guard_build_commit: str = "unknown",
    verification_commit: str = "unknown",
) -> dict[str, Any]:
    checks = {
        "guard_packet": _verify_guard_packet(Path(guard_packet), guard_packet_hash),
        "state_encoder_feature_schema": _verify_state_encoder(state_encoder_hash),
        "objective_normalizer_artifact": _verify_objective_normalizer(Path(objective_normalizer), objective_normalizer_hash),
        "vector_model_artifact": _verify_model_file(Path(vector_model_dir) / "model.pt", vector_model_hash),
        "film_model_artifact": _verify_model_file(Path(film_model_dir) / "model.pt", film_model_hash),
    }
    failures = [
        f"{name}: {result.get('failure')}"
        for name, result in checks.items()
        if not result.get("pass")
    ]
    packet = {
        "packet_type": "formal_jinan_3seed_real_artifact_hash_verification",
        "formal_experiment_execution_in_this_packet": False,
        "formal_execution_allowed_now": False,
        "cityflow_run_in_this_packet": False,
        "ppo_training_run_in_this_packet": False,
        "traffic_value_reading_in_this_packet": False,
        "numeric_aggregation_in_this_packet": False,
        "method_ranking_in_this_packet": False,
        "performance_table_in_this_packet": False,
        "paper_result_claim_in_this_packet": False,
        "provenance": {
            "guard_build_commit": guard_build_commit,
            "verification_commit": verification_commit,
            "guard_packet": str(guard_packet),
        },
        "artifact_checks": checks,
        "overall_pass": not failures,
        "failures": failures,
        "next_gate": {
            "required_exact_phrase": FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE,
            "execution_request_packet_required": True,
            "analysis_approval_requested": False,
            "ranking_approval_requested": False,
            "traffic_value_reading_approval_requested": False,
        },
        "forbidden_until_later_gate": [
            "formal_execution_without_exact_phrase",
            "traffic_value_reading",
            "numeric_aggregation",
            "method_ranking",
            "performance_table",
            "best_method_claim",
            "traffic_improvement_claim",
            "paper_ready_claim",
            "seed_expansion_beyond_0_1_2",
            "city_expansion",
        ],
    }
    output = Path(out_dir)
    _write_json(output / "formal_jinan_3seed_real_artifact_hash_verification.json", packet)
    write_markdown(packet, output / "formal_jinan_3seed_real_artifact_hash_verification.md")
    return packet


def write_markdown(packet: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# Formal Jinan 3-Seed Real Artifact Hash Verification",
        "",
        f"- overall_pass: `{packet['overall_pass']}`",
        f"- formal execution allowed now: `{packet['formal_execution_allowed_now']}`",
        f"- formal experiment executed in this packet: `{packet['formal_experiment_execution_in_this_packet']}`",
        f"- next required exact phrase: `{packet['next_gate']['required_exact_phrase']}`",
        "",
        "## Artifact Checks",
    ]
    for name, result in packet["artifact_checks"].items():
        lines.append(f"- {name}: `{'PASS' if result.get('pass') else 'FAIL'}`")
        if result.get("path"):
            lines.append(f"  - path: `{result['path']}`")
        if result.get("observed") is not None:
            lines.append(f"  - observed: `{result['observed']}`")
        if result.get("expected") is not None:
            lines.append(f"  - expected: `{result['expected']}`")
        if result.get("file_sha256") is not None:
            lines.append(f"  - file_sha256: `{result['file_sha256']}`")
        if result.get("expected_file_sha256") is not None:
            lines.append(f"  - expected_file_sha256: `{result['expected_file_sha256']}`")
    lines.extend(
        [
            "",
            "This packet only verifies locked artifact hashes. It does not run CityFlow, PPO, formal execution, traffic-value reading, aggregation, ranking, or performance-table generation.",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--guard_packet", default=DEFAULT_GUARD_PACKET)
    parser.add_argument("--guard_packet_hash")
    parser.add_argument("--vector_model_dir", default=DEFAULT_VECTOR_MODEL_DIR)
    parser.add_argument("--vector_model_hash", default=VECTORQ_MODEL_HASH)
    parser.add_argument("--film_model_dir", default=DEFAULT_FILM_MODEL_DIR)
    parser.add_argument("--film_model_hash", default=FILM_MODEL_HASH)
    parser.add_argument("--objective_normalizer", default=DEFAULT_OBJECTIVE_NORMALIZER)
    parser.add_argument("--objective_normalizer_hash", default=OBJECTIVE_NORMALIZER_HASH)
    parser.add_argument("--state_encoder_hash", default=STATE_ENCODER_HASH)
    parser.add_argument("--guard_build_commit", default="unknown")
    parser.add_argument("--verification_commit", default="unknown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet = build_artifact_hash_verification_packet(
        out_dir=args.out_dir,
        guard_packet=args.guard_packet,
        guard_packet_hash=args.guard_packet_hash,
        vector_model_dir=args.vector_model_dir,
        vector_model_hash=args.vector_model_hash,
        film_model_dir=args.film_model_dir,
        film_model_hash=args.film_model_hash,
        objective_normalizer=args.objective_normalizer,
        objective_normalizer_hash=args.objective_normalizer_hash,
        state_encoder_hash=args.state_encoder_hash,
        guard_build_commit=args.guard_build_commit,
        verification_commit=args.verification_commit,
    )
    print(json.dumps({"overall_pass": packet["overall_pass"], "failure_count": len(packet["failures"])}, sort_keys=True))
    if not packet["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
