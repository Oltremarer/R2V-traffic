from __future__ import annotations

import json
from pathlib import Path

from pareto.rl.formal_jinan_3seed_artifact_hash_verification import (
    build_artifact_hash_verification_packet,
    sha256_file,
)


def _write_guard(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "packet_type": "formal_jinan_3seed_execution_guard",
                "overall_pass": True,
                "formal_execution_allowed_now": False,
                "formal_experiment_execution_in_this_packet": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_artifact_hash_packet_verifies_real_files_without_execution(tmp_path: Path):
    guard_path = tmp_path / "guard.json"
    _write_guard(guard_path)
    vector_dir = tmp_path / "vector"
    film_dir = tmp_path / "film"
    vector_dir.mkdir()
    film_dir.mkdir()
    (vector_dir / "model.pt").write_bytes(b"vector-weights")
    (film_dir / "model.pt").write_bytes(b"film-weights")
    vector_hash = sha256_file(vector_dir / "model.pt")
    film_hash = sha256_file(film_dir / "model.pt")
    normalizer = tmp_path / "objective_norm.json"
    normalizer.write_text(json.dumps({"hash": "norm_hash"}) + "\n", encoding="utf-8")

    packet = build_artifact_hash_verification_packet(
        out_dir=tmp_path / "packet",
        guard_packet=guard_path,
        guard_packet_hash=sha256_file(guard_path),
        vector_model_dir=vector_dir,
        vector_model_hash=vector_hash,
        film_model_dir=film_dir,
        film_model_hash=film_hash,
        objective_normalizer=normalizer,
        objective_normalizer_hash="norm_hash",
        state_encoder_hash="4d1c2b4e276043ac",
        guard_build_commit="guard_commit",
        verification_commit="verification_commit",
    )

    assert packet["overall_pass"] is True
    assert packet["formal_experiment_execution_in_this_packet"] is False
    assert packet["cityflow_run_in_this_packet"] is False
    assert packet["ppo_training_run_in_this_packet"] is False
    assert packet["traffic_value_reading_in_this_packet"] is False
    assert packet["numeric_aggregation_in_this_packet"] is False
    assert packet["artifact_checks"]["vector_model_artifact"]["observed"] == vector_hash
    assert packet["artifact_checks"]["guard_packet"]["file_sha256"] == sha256_file(guard_path)
    assert packet["artifact_checks"]["guard_packet"]["expected_file_sha256"] == sha256_file(guard_path)
    assert packet["artifact_checks"]["film_model_artifact"]["observed"] == film_hash
    assert packet["artifact_checks"]["objective_normalizer_artifact"]["internal_hash"] == "norm_hash"
    assert packet["artifact_checks"]["state_encoder_feature_schema"]["observed"] == "4d1c2b4e276043ac"
    assert (tmp_path / "packet" / "formal_jinan_3seed_real_artifact_hash_verification.json").exists()
    assert (tmp_path / "packet" / "formal_jinan_3seed_real_artifact_hash_verification.md").exists()


def test_artifact_hash_packet_fails_on_model_hash_mismatch(tmp_path: Path):
    guard_path = tmp_path / "guard.json"
    _write_guard(guard_path)
    vector_dir = tmp_path / "vector"
    film_dir = tmp_path / "film"
    vector_dir.mkdir()
    film_dir.mkdir()
    (vector_dir / "model.pt").write_bytes(b"vector-weights")
    (film_dir / "model.pt").write_bytes(b"film-weights")
    normalizer = tmp_path / "objective_norm.json"
    normalizer.write_text(json.dumps({"hash": "norm_hash"}) + "\n", encoding="utf-8")

    packet = build_artifact_hash_verification_packet(
        out_dir=tmp_path / "packet",
        guard_packet=guard_path,
        vector_model_dir=vector_dir,
        vector_model_hash="wrong",
        film_model_dir=film_dir,
        film_model_hash=sha256_file(film_dir / "model.pt"),
        objective_normalizer=normalizer,
        objective_normalizer_hash="norm_hash",
        state_encoder_hash="4d1c2b4e276043ac",
    )

    assert packet["overall_pass"] is False
    assert packet["artifact_checks"]["vector_model_artifact"]["pass"] is False
    assert "vector_model_artifact" in packet["failures"][0]


def test_artifact_hash_packet_rejects_non_guard_packet(tmp_path: Path):
    guard_path = tmp_path / "guard.json"
    guard_path.write_text(json.dumps({"packet_type": "not_a_guard", "overall_pass": True}) + "\n", encoding="utf-8")
    vector_dir = tmp_path / "vector"
    film_dir = tmp_path / "film"
    vector_dir.mkdir()
    film_dir.mkdir()
    (vector_dir / "model.pt").write_bytes(b"vector-weights")
    (film_dir / "model.pt").write_bytes(b"film-weights")
    normalizer = tmp_path / "objective_norm.json"
    normalizer.write_text(json.dumps({"hash": "norm_hash"}) + "\n", encoding="utf-8")

    packet = build_artifact_hash_verification_packet(
        out_dir=tmp_path / "packet",
        guard_packet=guard_path,
        vector_model_dir=vector_dir,
        vector_model_hash=sha256_file(vector_dir / "model.pt"),
        film_model_dir=film_dir,
        film_model_hash=sha256_file(film_dir / "model.pt"),
        objective_normalizer=normalizer,
        objective_normalizer_hash="norm_hash",
        state_encoder_hash="4d1c2b4e276043ac",
    )

    assert packet["overall_pass"] is False
    assert packet["artifact_checks"]["guard_packet"]["pass"] is False


def test_artifact_hash_packet_fails_on_guard_file_sha_mismatch(tmp_path: Path):
    guard_path = tmp_path / "guard.json"
    _write_guard(guard_path)
    vector_dir = tmp_path / "vector"
    film_dir = tmp_path / "film"
    vector_dir.mkdir()
    film_dir.mkdir()
    (vector_dir / "model.pt").write_bytes(b"vector-weights")
    (film_dir / "model.pt").write_bytes(b"film-weights")
    normalizer = tmp_path / "objective_norm.json"
    normalizer.write_text(json.dumps({"hash": "norm_hash"}) + "\n", encoding="utf-8")

    packet = build_artifact_hash_verification_packet(
        out_dir=tmp_path / "packet",
        guard_packet=guard_path,
        guard_packet_hash="wrong",
        vector_model_dir=vector_dir,
        vector_model_hash=sha256_file(vector_dir / "model.pt"),
        film_model_dir=film_dir,
        film_model_hash=sha256_file(film_dir / "model.pt"),
        objective_normalizer=normalizer,
        objective_normalizer_hash="norm_hash",
        state_encoder_hash="4d1c2b4e276043ac",
    )

    assert packet["overall_pass"] is False
    assert packet["artifact_checks"]["guard_packet"]["pass"] is False
    assert packet["artifact_checks"]["guard_packet"]["failure"] == "guard packet file_sha256 mismatch"
