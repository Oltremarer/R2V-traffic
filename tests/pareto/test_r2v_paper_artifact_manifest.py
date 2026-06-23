from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from pareto.r2v.paper_artifact_manifest import (
    ALLOWED_ARTIFACT_TYPES,
    build_paper_artifact_manifest,
    parse_artifact_spec,
)
from pareto.r2v.result_aggregation import aggregate_r2v_results
from pareto.r2v.traffic_artifact_schema import upgrade_weighted_row_to_v2_metadata


def _legacy_weighted_row(sample_id: str = "s0", transition_id: str = "t0") -> dict:
    return {
        "sample_id": sample_id,
        "transition_id": transition_id,
        "metadata": {
            "r2v_schema_version": "r2v-tsc-weighted-transition-v1",
            "r2v_sample_weight": 2.5,
            "r2v_admitted": True,
            "r2v_gates": {
                "rare": True,
                "value": True,
                "support": True,
                "safety": True,
            },
        },
    }


def test_manifest_records_hashes_and_keeps_performance_integrity_roles_separate(tmp_path: Path):
    performance_path = tmp_path / "r2v_performance_rows.jsonl"
    performance_path.write_text(
        json.dumps(
            {
                "method": "r2v_diffusion_not_rare_to_val_full",
                "seed": 0,
                "average_travel_time": 90.0,
                "queue_length": 8.0,
                "delay": 2.5,
                "throughput": 55.0,
                "reward": -15.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    integrity_path = tmp_path / "r2v_summary.json"
    integrity_path.write_text(
        json.dumps(
            {
                "schema_version": "r2v-tsc-candidate-summary-v1",
                "candidate_count": 4,
                "admitted_count": 2,
                "gate_counts": {"rare": 4, "value": 3, "support": 4, "safety": 2},
            }
        ),
        encoding="utf-8",
    )

    manifest = build_paper_artifact_manifest(
        [
            {"artifact_type": "performance", "name": "main_performance_rows", "path": performance_path},
            {"artifact_type": "integrity", "name": "seed0_r2v_summary", "path": integrity_path},
        ]
    )

    entries_by_name = {entry["name"]: entry for entry in manifest["entries"]}

    assert manifest["schema_version"] == "r2v-traffic-paper-artifact-manifest-v1"
    assert manifest["status"] == "READY"
    assert manifest["artifact_type_counts"] == {"integrity": 1, "performance": 1}
    assert manifest["claim_boundary"] == "performance artifacts and integrity/status artifacts are tracked as separate artifact types"
    assert entries_by_name["main_performance_rows"]["artifact_type"] == "performance"
    assert entries_by_name["main_performance_rows"]["json_format"] == "jsonl"
    assert entries_by_name["main_performance_rows"]["line_count"] == 1
    assert entries_by_name["main_performance_rows"]["sha256"] == hashlib.sha256(
        performance_path.read_bytes()
    ).hexdigest()
    assert entries_by_name["seed0_r2v_summary"]["artifact_type"] == "integrity"
    assert entries_by_name["seed0_r2v_summary"]["json_format"] == "json"
    assert entries_by_name["seed0_r2v_summary"]["schema_version"] == "r2v-tsc-candidate-summary-v1"
    assert entries_by_name["seed0_r2v_summary"]["integrity_candidate_count"] == 4
    assert entries_by_name["seed0_r2v_summary"]["integrity_admitted_count"] == 2
    assert entries_by_name["seed0_r2v_summary"]["integrity_gate_counts"]["value"] == 3


def test_manifest_blocks_missing_artifact_without_hashing(tmp_path: Path):
    missing_path = tmp_path / "missing_scores.jsonl"

    manifest = build_paper_artifact_manifest(
        [{"artifact_type": "diffusion_score", "name": "seed0_diffusion_scores", "path": missing_path}]
    )

    assert manifest["status"] == "BLOCKED"
    assert manifest["failed_count"] == 1
    assert manifest["entries"][0]["status"] == "missing"
    assert manifest["entries"][0]["sha256"] is None


def test_manifest_blocks_status_only_file_labeled_as_integrity(tmp_path: Path):
    integrity_path = tmp_path / "status_only_summary.json"
    integrity_path.write_text(json.dumps({"status": "DONE"}), encoding="utf-8")

    manifest = build_paper_artifact_manifest(
        [{"artifact_type": "integrity", "name": "bad_summary", "path": integrity_path}]
    )

    assert manifest["status"] == "BLOCKED"
    entry = manifest["entries"][0]
    assert entry["status"] == "invalid_content"
    assert "candidate summary schema" in entry["message"]


def test_manifest_blocks_status_only_file_labeled_as_performance(tmp_path: Path):
    status_only_path = tmp_path / "status_only.jsonl"
    status_only_path.write_text(json.dumps({"method": "r2v", "seed": 0, "status": "DONE"}) + "\n")

    manifest = build_paper_artifact_manifest(
        [{"artifact_type": "performance", "name": "bad_performance", "path": status_only_path}]
    )

    assert manifest["status"] == "BLOCKED"
    assert manifest["failed_count"] == 1
    entry = manifest["entries"][0]
    assert entry["status"] == "invalid_content"
    assert entry["performance_metric_count"] == 0
    assert "no performance metrics found" in entry["message"]


def test_manifest_blocks_performance_artifact_missing_required_traffic_metrics(tmp_path: Path):
    incomplete_path = tmp_path / "incomplete_performance.jsonl"
    incomplete_path.write_text(
        json.dumps(
            {
                "method": "r2v_diffusion_not_rare_to_val_full",
                "seed": 0,
                "average_travel_time": 90.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = build_paper_artifact_manifest(
        [{"artifact_type": "performance", "name": "incomplete_performance", "path": incomplete_path}]
    )

    assert manifest["status"] == "BLOCKED"
    entry = manifest["entries"][0]
    assert entry["status"] == "invalid_content"
    assert entry["performance_missing_metrics"] == ["queue_length", "delay", "throughput", "reward"]
    assert "missing required traffic metrics" in entry["message"]


def test_manifest_accepts_result_aggregation_artifact(tmp_path: Path):
    perf = tmp_path / "eval_metrics.jsonl"
    perf.write_text(
        json.dumps(
            {
                "method": "r2v_diffusion_not_rare_to_val_full",
                "seed": 0,
                "average_travel_time": 90.0,
                "queue_length": 8.0,
                "delay": 2.5,
                "throughput": 55.0,
                "reward": -15.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    integrity = tmp_path / "r2v_summary.json"
    integrity.write_text(
        json.dumps(
            {
                "schema_version": "r2v-tsc-candidate-summary-v1",
                "candidate_count": 4,
                "admitted_count": 2,
                "gate_counts": {"rare": 4, "value": 3, "support": 4, "safety": 2},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    aggregation_path = tmp_path / "r2v_result_aggregation.json"
    aggregation_path.write_text(
        json.dumps(aggregate_r2v_results(performance_paths=[perf], integrity_paths=[integrity])),
        encoding="utf-8",
    )

    manifest = build_paper_artifact_manifest(
        [
            {"artifact_type": "performance", "name": "main_performance", "path": perf},
            {"artifact_type": "integrity", "name": "seed0_summary", "path": integrity},
            {"artifact_type": "aggregation", "name": "main_result_aggregation", "path": aggregation_path},
        ]
    )

    assert manifest["status"] == "READY"
    entry = {entry["name"]: entry for entry in manifest["entries"]}["main_result_aggregation"]
    assert entry["aggregation_schema_version"] == "r2v-traffic-result-aggregation-v1"
    assert entry["aggregation_performance_row_count"] == 1
    assert entry["aggregation_metric_value_count"] == 5
    assert set(entry["aggregation_metrics"]) == {
        "average_travel_time",
        "queue_length",
        "delay",
        "throughput",
        "reward",
    }
    assert entry["aggregation_input_artifact_counts"] == {"integrity": 1, "performance": 1}
    assert entry["aggregation_input_artifact_hash_count"] == 2


def test_manifest_blocks_result_aggregation_when_input_hashes_do_not_match_bundle(tmp_path: Path):
    perf = tmp_path / "eval_metrics.jsonl"
    perf.write_text(
        json.dumps(
            {
                "method": "r2v_diffusion_not_rare_to_val_full",
                "seed": 0,
                "average_travel_time": 90.0,
                "queue_length": 8.0,
                "delay": 2.5,
                "throughput": 55.0,
                "reward": -15.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    integrity = tmp_path / "r2v_summary.json"
    integrity.write_text(
        json.dumps(
            {
                "schema_version": "r2v-tsc-candidate-summary-v1",
                "candidate_count": 4,
                "admitted_count": 2,
                "gate_counts": {"rare": 4, "value": 3, "support": 4, "safety": 2},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    aggregation_path = tmp_path / "r2v_result_aggregation.json"
    aggregation_path.write_text(
        json.dumps(aggregate_r2v_results(performance_paths=[perf], integrity_paths=[integrity])),
        encoding="utf-8",
    )
    perf.write_text(
        json.dumps(
            {
                "method": "r2v_diffusion_not_rare_to_val_full",
                "seed": 0,
                "average_travel_time": 91.0,
                "queue_length": 8.0,
                "delay": 2.5,
                "throughput": 55.0,
                "reward": -15.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = build_paper_artifact_manifest(
        [
            {"artifact_type": "performance", "name": "main_performance", "path": perf},
            {"artifact_type": "integrity", "name": "seed0_summary", "path": integrity},
            {"artifact_type": "aggregation", "name": "main_result_aggregation", "path": aggregation_path},
        ]
    )

    assert manifest["status"] == "BLOCKED"
    entry = {entry["name"]: entry for entry in manifest["entries"]}["main_result_aggregation"]
    assert entry["status"] == "invalid_content"
    assert "do not match bundled artifacts" in entry["message"]
    assert list(entry["aggregation_unmatched_input_artifact_hashes"]) == ["performance"]


def test_manifest_blocks_result_aggregation_without_input_artifact_hashes(tmp_path: Path):
    aggregation_path = tmp_path / "r2v_result_aggregation.json"
    aggregation_path.write_text(
        json.dumps(
            {
                "schema_version": "r2v-traffic-result-aggregation-v1",
                "performance": {
                    "metrics": {
                        "average_travel_time": "average_travel_time",
                        "queue_length": "queue_length",
                        "delay": "delay",
                        "throughput": "throughput",
                        "reward": "reward",
                    },
                    "by_method": {"r2v": {"reward": {"count": 1, "mean": -15.0}}},
                    "row_count": 1,
                    "metric_value_count": 5,
                },
                "integrity": {"artifact_count": 1},
                "claim_boundary": "performance metrics are aggregated separately from R2V integrity/status artifacts",
            }
        ),
        encoding="utf-8",
    )

    manifest = build_paper_artifact_manifest(
        [{"artifact_type": "aggregation", "name": "main_result_aggregation", "path": aggregation_path}]
    )

    assert manifest["status"] == "BLOCKED"
    entry = manifest["entries"][0]
    assert entry["status"] == "invalid_content"
    assert "input artifact hashes" in entry["message"]


def test_manifest_blocks_status_only_file_labeled_as_aggregation(tmp_path: Path):
    aggregation_path = tmp_path / "bad_aggregation.json"
    aggregation_path.write_text(json.dumps({"status": "DONE", "row_count": 1}), encoding="utf-8")

    manifest = build_paper_artifact_manifest(
        [{"artifact_type": "aggregation", "name": "bad_aggregation", "path": aggregation_path}]
    )

    assert manifest["status"] == "BLOCKED"
    entry = manifest["entries"][0]
    assert entry["status"] == "invalid_content"
    assert "result aggregation schema" in entry["message"]


def test_manifest_accepts_paper_eligible_diffusion_score_artifact(tmp_path: Path):
    score_path = tmp_path / "diffusion_scores.jsonl"
    score_path.write_text(
        json.dumps(
            {
                "transition_id": "t0",
                "rarity_score": 1.0,
                "support_score": 0.5,
                "paper_claim_eligible": True,
                "adapter": "trained_diffusion_detector",
                "model_checkpoint": "checkpoints/r2v_diffusion_seed0.pt",
                "config_hash": "abc123",
                "normalization_id": "jinan_norm_v1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = build_paper_artifact_manifest(
        [{"artifact_type": "diffusion_score", "name": "seed0_scores", "path": score_path}]
    )

    assert manifest["status"] == "READY"
    entry = manifest["entries"][0]
    assert entry["diffusion_score_row_count"] == 1
    assert entry["paper_claim_eligible"] is True
    assert entry["paper_claim_provenance_missing_count"] == 0
    assert entry["paper_claim_proxy_adapter_count"] == 0


def test_manifest_blocks_diffusion_repair_payload_missing_sample_id(tmp_path: Path):
    score_path = tmp_path / "diffusion_scores.jsonl"
    score_path.write_text(
        json.dumps(
            {
                "transition_id": "t0",
                "rarity_score": 1.0,
                "support_score": 0.5,
                "paper_claim_eligible": True,
                "adapter": "trained_diffusion_detector",
                "model_checkpoint": "checkpoints/r2v_diffusion_seed0.pt",
                "config_hash": "abc123",
                "normalization_id": "jinan_norm_v1",
                "repaired_transition": {
                    "transition_id": "t0_repaired",
                    "obs_features": [1.0],
                    "next_obs_features": [1.5],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = build_paper_artifact_manifest(
        [{"artifact_type": "diffusion_score", "name": "seed0_scores", "path": score_path}]
    )

    assert manifest["status"] == "BLOCKED"
    entry = manifest["entries"][0]
    assert entry["status"] == "invalid_content"
    assert "repaired_transition missing sample_id" in entry["message"]


def test_manifest_blocks_proxy_diffusion_score_artifact(tmp_path: Path):
    score_path = tmp_path / "diffusion_scores.jsonl"
    score_path.write_text(
        json.dumps(
            {
                "transition_id": "t0",
                "rarity_score": 1.0,
                "support_score": 0.5,
                "paper_claim_eligible": False,
                "adapter": "traffic_feature_density_proxy",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = build_paper_artifact_manifest(
        [{"artifact_type": "diffusion_score", "name": "seed0_scores", "path": score_path}]
    )

    assert manifest["status"] == "BLOCKED"
    entry = manifest["entries"][0]
    assert entry["status"] == "invalid_content"
    assert entry["paper_claim_eligible"] is False
    assert entry["paper_claim_ineligible_count"] == 1
    assert entry["paper_claim_proxy_adapter_count"] == 1
    assert "not paper-claim eligible" in entry["message"]


def test_manifest_blocks_diffusion_score_artifact_missing_provenance(tmp_path: Path):
    score_path = tmp_path / "diffusion_scores.jsonl"
    score_path.write_text(
        json.dumps(
            {
                "transition_id": "t0",
                "rarity_score": 1.0,
                "support_score": 0.5,
                "paper_claim_eligible": True,
                "adapter": "trained_diffusion_detector",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = build_paper_artifact_manifest(
        [{"artifact_type": "diffusion_score", "name": "seed0_scores", "path": score_path}]
    )

    assert manifest["status"] == "BLOCKED"
    entry = manifest["entries"][0]
    assert entry["status"] == "invalid_content"
    assert entry["paper_claim_eligible"] is True
    assert entry["paper_claim_provenance_missing_count"] == 1
    assert entry["paper_claim_provenance_missing_transition_ids"] == ["t0"]
    assert "missing paper diffusion provenance" in entry["message"]


def test_manifest_accepts_v2_weighted_transition_artifact(tmp_path: Path):
    weighted_path = tmp_path / "r2v_weighted_transitions.jsonl"
    rows = [
        upgrade_weighted_row_to_v2_metadata(_legacy_weighted_row("s0", "t0"), generative_backend="feature_density_proxy"),
        upgrade_weighted_row_to_v2_metadata(_legacy_weighted_row("s1", "t1"), generative_backend="feature_density_proxy"),
    ]
    weighted_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    manifest = build_paper_artifact_manifest(
        [{"artifact_type": "weighted_transitions", "name": "seed0_weighted", "path": weighted_path}]
    )

    assert manifest["status"] == "READY"
    entry = manifest["entries"][0]
    assert entry["weighted_transition_row_count"] == 2
    assert entry["weighted_transition_admitted_count"] == 2
    assert entry["weighted_transition_gate_variants"] == ["full"]
    assert entry["weighted_transition_generative_backends"] == ["feature_density_proxy"]
    assert entry["weighted_transition_admission_modes"] == ["weights_only"]
    assert entry["weighted_transition_row_roles"] == ["source"]
    assert entry["weighted_transition_score_artifact_paths"] == []


def test_manifest_accepts_diffusion_weighted_artifact_when_score_artifact_matches(tmp_path: Path):
    score_path = tmp_path / "diffusion_scores.jsonl"
    score_path.write_text(
        json.dumps(
            {
                "transition_id": "t0",
                "rarity_score": 1.0,
                "support_score": 0.5,
                "paper_claim_eligible": True,
                "adapter": "trained_diffusion_detector",
                "model_checkpoint": "checkpoints/r2v_diffusion_seed0.pt",
                "config_hash": "abc123",
                "normalization_id": "jinan_norm_v1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    weighted_path = tmp_path / "r2v_weighted_transitions.jsonl"
    row = upgrade_weighted_row_to_v2_metadata(_legacy_weighted_row("s0", "t0"), generative_backend="diffusion")
    row["metadata"]["r2v_score_artifact_path"] = str(score_path)
    weighted_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    manifest = build_paper_artifact_manifest(
        [
            {"artifact_type": "diffusion_score", "name": "seed0_scores", "path": score_path},
            {"artifact_type": "weighted_transitions", "name": "seed0_weighted", "path": weighted_path},
        ]
    )

    assert manifest["status"] == "READY"
    entry = next(row for row in manifest["entries"] if row["artifact_type"] == "weighted_transitions")
    assert entry["weighted_transition_generative_backends"] == ["diffusion"]
    assert entry["weighted_transition_score_artifact_paths"] == [str(score_path)]


def test_manifest_blocks_diffusion_weighted_artifact_without_matching_score_artifact(tmp_path: Path):
    score_path = tmp_path / "diffusion_scores.jsonl"
    score_path.write_text(
        json.dumps(
            {
                "transition_id": "t0",
                "rarity_score": 1.0,
                "support_score": 0.5,
                "paper_claim_eligible": True,
                "adapter": "trained_diffusion_detector",
                "model_checkpoint": "checkpoints/r2v_diffusion_seed0.pt",
                "config_hash": "abc123",
                "normalization_id": "jinan_norm_v1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    weighted_path = tmp_path / "r2v_weighted_transitions.jsonl"
    row = upgrade_weighted_row_to_v2_metadata(_legacy_weighted_row("s0", "t0"), generative_backend="diffusion")
    row["metadata"]["r2v_score_artifact_path"] = str(tmp_path / "different_scores.jsonl")
    weighted_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    manifest = build_paper_artifact_manifest(
        [
            {"artifact_type": "diffusion_score", "name": "seed0_scores", "path": score_path},
            {"artifact_type": "weighted_transitions", "name": "seed0_weighted", "path": weighted_path},
        ]
    )

    assert manifest["status"] == "BLOCKED"
    entry = next(row for row in manifest["entries"] if row["artifact_type"] == "weighted_transitions")
    assert entry["status"] == "invalid_content"
    assert "do not match bundled diffusion score artifacts" in entry["message"]


def test_manifest_blocks_weighted_transition_artifact_missing_traffic_schema(tmp_path: Path):
    weighted_path = tmp_path / "r2v_weighted_transitions.jsonl"
    weighted_path.write_text(json.dumps(_legacy_weighted_row()) + "\n", encoding="utf-8")

    manifest = build_paper_artifact_manifest(
        [{"artifact_type": "weighted_transitions", "name": "bad_weighted", "path": weighted_path}]
    )

    assert manifest["status"] == "BLOCKED"
    entry = manifest["entries"][0]
    assert entry["status"] == "invalid_content"
    assert "r2v_traffic_schema_version" in entry["message"]


def test_manifest_accepts_ready_readiness_artifact(tmp_path: Path):
    readiness_path = tmp_path / "r2v_readiness.json"
    readiness_path.write_text(
        json.dumps(
            {
                "schema_version": "r2v-traffic-readiness-v1",
                "status": "READY",
                "failed_count": 0,
                "checks": [{"name": "diffusion_score_artifact", "status": "pass"}],
            }
        ),
        encoding="utf-8",
    )

    manifest = build_paper_artifact_manifest(
        [{"artifact_type": "readiness", "name": "seed0_readiness", "path": readiness_path}]
    )

    assert manifest["status"] == "READY"
    entry = manifest["entries"][0]
    assert entry["readiness_status"] == "READY"
    assert entry["readiness_failed_count"] == 0
    assert entry["readiness_check_count"] == 1


def test_manifest_blocks_blocked_readiness_artifact(tmp_path: Path):
    readiness_path = tmp_path / "r2v_readiness.json"
    readiness_path.write_text(
        json.dumps(
            {
                "schema_version": "r2v-traffic-readiness-v1",
                "status": "BLOCKED",
                "failed_count": 1,
                "failed_checks": [{"name": "diffusion_score_artifact", "status": "fail"}],
            }
        ),
        encoding="utf-8",
    )

    manifest = build_paper_artifact_manifest(
        [{"artifact_type": "readiness", "name": "seed0_readiness", "path": readiness_path}]
    )

    assert manifest["status"] == "BLOCKED"
    entry = manifest["entries"][0]
    assert entry["status"] == "invalid_content"
    assert entry["readiness_status"] == "BLOCKED"
    assert entry["readiness_failed_count"] == 1
    assert "readiness artifact is not READY" in entry["message"]


def test_manifest_rejects_unknown_artifact_type(tmp_path: Path):
    path = tmp_path / "artifact.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown artifact_type"):
        build_paper_artifact_manifest(
            [{"artifact_type": "status_as_performance", "name": "bad", "path": path}]
        )


def test_parse_artifact_spec_splits_type_name_and_path():
    artifact_type, name, path = parse_artifact_spec("performance:main:/tmp/perf.jsonl")

    assert artifact_type == "performance"
    assert name == "main"
    assert path == "/tmp/perf.jsonl"
    assert "performance" in ALLOWED_ARTIFACT_TYPES


def test_paper_artifact_manifest_cli_writes_json(tmp_path: Path):
    performance_path = tmp_path / "perf.jsonl"
    performance_path.write_text(
        json.dumps(
            {
                "method": "baseline",
                "average_travel_time": 100.0,
                "queue_length": 10.0,
                "delay": 3.0,
                "throughput": 50.0,
                "reward": -20.0,
            }
        )
        + "\n"
    )
    output_path = tmp_path / "manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pareto.r2v.paper_artifact_manifest",
            "--artifact",
            f"performance:main_performance:{performance_path}",
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "READY"
    assert payload["entries"][0]["name"] == "main_performance"
    assert payload["entries"][0]["artifact_type"] == "performance"


def test_paper_artifact_manifest_cli_blocks_status_only_performance_artifact(tmp_path: Path):
    status_only_path = tmp_path / "status_only.jsonl"
    status_only_path.write_text(json.dumps({"method": "r2v", "seed": 0, "status": "DONE"}) + "\n")
    output_path = tmp_path / "manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pareto.r2v.paper_artifact_manifest",
            "--artifact",
            f"performance:bad_performance:{status_only_path}",
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKED"
    assert payload["failed_entries"][0]["status"] == "invalid_content"


def test_paper_artifact_manifest_cli_blocks_proxy_diffusion_score_artifact(tmp_path: Path):
    score_path = tmp_path / "proxy_scores.jsonl"
    score_path.write_text(
        json.dumps(
            {
                "transition_id": "t0",
                "rarity_score": 1.0,
                "support_score": 0.5,
                "paper_claim_eligible": False,
                "adapter": "traffic_feature_density_proxy",
            }
        )
        + "\n"
    )
    output_path = tmp_path / "manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pareto.r2v.paper_artifact_manifest",
            "--artifact",
            f"diffusion_score:seed0_scores:{score_path}",
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKED"
    assert payload["failed_entries"][0]["paper_claim_proxy_adapter_count"] == 1
