from __future__ import annotations

import json
from pathlib import Path

from pareto.rl.formal_jinan_3seed_artifact_hash_verification import sha256_file
from pareto.rl.formal_jinan_3seed_execution_request import build_execution_request_packet


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_guard(path: Path) -> str:
    _write_json(
        path,
        {
            "packet_type": "formal_jinan_3seed_execution_guard",
            "overall_pass": True,
            "formal_execution_allowed_now": False,
            "formal_experiment_execution_in_this_packet": False,
            "run_manifest": [{"method": "vector_quality_potential"} for _ in range(12)],
        },
    )
    return sha256_file(path)


def _write_verification(path: Path, guard_hash: str) -> str:
    _write_json(
        path,
        {
            "packet_type": "formal_jinan_3seed_real_artifact_hash_verification",
            "overall_pass": True,
            "formal_execution_allowed_now": False,
            "formal_experiment_execution_in_this_packet": False,
            "artifact_checks": {
                "guard_packet": {
                    "file_sha256": guard_hash,
                    "expected_file_sha256": guard_hash,
                    "pass": True,
                }
            },
        },
    )
    return sha256_file(path)


def test_execution_request_packet_pins_guard_and_verification_hashes(tmp_path: Path):
    guard_path = tmp_path / "guard.json"
    guard_hash = _write_guard(guard_path)
    verification_path = tmp_path / "verification.json"
    verification_hash = _write_verification(verification_path, guard_hash)

    packet = build_execution_request_packet(
        out_dir=tmp_path / "request",
        guard_packet=guard_path,
        verification_packet=verification_path,
        verification_packet_commit="verification_packet_commit",
        request_commit="request_commit",
    )

    assert packet["overall_pass"] is True
    assert packet["formal_execution_allowed_now"] is False
    assert packet["formal_experiment_execution_in_this_packet"] is False
    assert packet["cityflow_run_in_this_packet"] is False
    assert packet["traffic_value_reading_in_this_packet"] is False
    assert packet["provenance"]["guard_packet_sha256"] == guard_hash
    assert packet["provenance"]["verification_packet_sha256"] == verification_hash
    assert packet["request_checks"]["real_artifact_hash_verification_packet"]["pass"] is True
    assert len(packet["run_manifest"]) == 12
    assert (tmp_path / "request" / "formal_jinan_3seed_execution_request.json").exists()
    assert (tmp_path / "request" / "formal_jinan_3seed_execution_request.md").exists()


def test_execution_request_packet_fails_if_verification_anchors_wrong_guard_hash(tmp_path: Path):
    guard_path = tmp_path / "guard.json"
    _write_guard(guard_path)
    verification_path = tmp_path / "verification.json"
    _write_verification(verification_path, "wrong")

    packet = build_execution_request_packet(
        out_dir=tmp_path / "request",
        guard_packet=guard_path,
        verification_packet=verification_path,
    )

    assert packet["overall_pass"] is False
    assert packet["request_checks"]["real_artifact_hash_verification_packet"]["pass"] is False
    assert "verification packet" in packet["failures"][0]
