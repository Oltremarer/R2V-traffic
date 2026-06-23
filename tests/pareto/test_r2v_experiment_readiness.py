from __future__ import annotations

import json
from pathlib import Path

import pytest

from pareto.r2v.experiment_readiness import check_r2v_traffic_readiness, parse_args, parse_seed_artifact


def _write(path: Path, text: str = "{}") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_readiness_reports_missing_cityflow_data_and_transitions(tmp_path: Path):
    report = check_r2v_traffic_readiness(
        root=tmp_path,
        scenario="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        transition_glob="records/jinan/seed{seed}/transitions_raw.jsonl",
        seeds=(0,),
    )

    assert report["status"] == "BLOCKED"
    failed = {(item["name"], item.get("seed")) for item in report["failed_checks"]}
    assert ("traffic_file", None) in failed
    assert ("roadnet_file", None) in failed
    assert ("transition_inputs", 0) in failed


def test_readiness_passes_when_cityflow_data_transitions_and_diffusion_artifacts_exist(tmp_path: Path):
    _write(tmp_path / "data/Jinan/3_4/anon_3_4_jinan_real.json")
    _write(tmp_path / "data/Jinan/3_4/roadnet_3_4.json")
    _write(tmp_path / "records/jinan/seed0/transitions_raw.jsonl", json.dumps({"transition_id": "t0"}) + "\n")
    _write(
        tmp_path / "artifacts/seed0/diffusion_scores.jsonl",
        json.dumps({"transition_id": "t0", "rarity_score": 1.0, "support_score": 0.5}) + "\n",
    )

    report = check_r2v_traffic_readiness(
        root=tmp_path,
        scenario="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        transition_glob="records/jinan/seed{seed}/transitions_raw.jsonl",
        seeds=(0,),
        diffusion_artifacts={0: "artifacts/seed0/diffusion_scores.jsonl"},
        require_diffusion_artifacts=True,
    )

    assert report["status"] == "READY"
    assert report["failed_count"] == 0


def test_readiness_rejects_proxy_diffusion_artifact_for_paper_claims(tmp_path: Path):
    _write(tmp_path / "data/Jinan/3_4/anon_3_4_jinan_real.json")
    _write(tmp_path / "data/Jinan/3_4/roadnet_3_4.json")
    _write(tmp_path / "records/jinan/seed0/transitions_raw.jsonl", json.dumps({"transition_id": "t0"}) + "\n")
    _write(
        tmp_path / "artifacts/seed0/diffusion_scores.jsonl",
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
    )

    report = check_r2v_traffic_readiness(
        root=tmp_path,
        scenario="jinan",
        transition_glob="records/jinan/seed{seed}/transitions_raw.jsonl",
        seeds=(0,),
        diffusion_artifacts={0: "artifacts/seed0/diffusion_scores.jsonl"},
        require_diffusion_artifacts=True,
        require_paper_claim_eligible_diffusion=True,
    )

    assert report["status"] == "BLOCKED"
    failure = next(item for item in report["failed_checks"] if item["name"] == "diffusion_score_artifact")
    assert failure["paper_claim_eligible"] is False
    assert "not paper-claim eligible" in failure["message"]


def test_readiness_accepts_paper_eligible_diffusion_artifact_for_paper_claims(tmp_path: Path):
    _write(tmp_path / "data/Jinan/3_4/anon_3_4_jinan_real.json")
    _write(tmp_path / "data/Jinan/3_4/roadnet_3_4.json")
    _write(tmp_path / "records/jinan/seed0/transitions_raw.jsonl", json.dumps({"transition_id": "t0"}) + "\n")
    _write(
        tmp_path / "artifacts/seed0/diffusion_scores.jsonl",
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
    )

    report = check_r2v_traffic_readiness(
        root=tmp_path,
        scenario="jinan",
        transition_glob="records/jinan/seed{seed}/transitions_raw.jsonl",
        seeds=(0,),
        diffusion_artifacts={0: "artifacts/seed0/diffusion_scores.jsonl"},
        require_diffusion_artifacts=True,
        require_paper_claim_eligible_diffusion=True,
    )

    assert report["status"] == "READY"


def test_readiness_rejects_diffusion_repair_payload_missing_sample_id(tmp_path: Path):
    _write(tmp_path / "data/Jinan/3_4/anon_3_4_jinan_real.json")
    _write(tmp_path / "data/Jinan/3_4/roadnet_3_4.json")
    _write(tmp_path / "records/jinan/seed0/transitions_raw.jsonl", json.dumps({"transition_id": "t0"}) + "\n")
    _write(
        tmp_path / "artifacts/seed0/diffusion_scores.jsonl",
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
    )

    report = check_r2v_traffic_readiness(
        root=tmp_path,
        scenario="jinan",
        transition_glob="records/jinan/seed{seed}/transitions_raw.jsonl",
        seeds=(0,),
        diffusion_artifacts={0: "artifacts/seed0/diffusion_scores.jsonl"},
        require_diffusion_artifacts=True,
        require_paper_claim_eligible_diffusion=True,
    )

    assert report["status"] == "BLOCKED"
    failure = next(item for item in report["failed_checks"] if item["name"] == "diffusion_score_artifact")
    assert "repaired_transition missing sample_id" in failure["message"]


def test_readiness_rejects_paper_eligible_diffusion_artifact_missing_provenance(tmp_path: Path):
    _write(tmp_path / "data/Jinan/3_4/anon_3_4_jinan_real.json")
    _write(tmp_path / "data/Jinan/3_4/roadnet_3_4.json")
    _write(tmp_path / "records/jinan/seed0/transitions_raw.jsonl", json.dumps({"transition_id": "t0"}) + "\n")
    _write(
        tmp_path / "artifacts/seed0/diffusion_scores.jsonl",
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
    )

    report = check_r2v_traffic_readiness(
        root=tmp_path,
        scenario="jinan",
        transition_glob="records/jinan/seed{seed}/transitions_raw.jsonl",
        seeds=(0,),
        diffusion_artifacts={0: "artifacts/seed0/diffusion_scores.jsonl"},
        require_diffusion_artifacts=True,
        require_paper_claim_eligible_diffusion=True,
    )

    assert report["status"] == "BLOCKED"
    failure = next(item for item in report["failed_checks"] if item["name"] == "diffusion_score_artifact")
    assert failure["paper_claim_provenance_missing_count"] == 1
    assert failure["paper_claim_provenance_missing_transition_ids"] == ["t0"]
    assert "missing paper diffusion provenance" in failure["message"]


def test_readiness_rejects_proxy_adapter_even_if_paper_claim_flag_is_true(tmp_path: Path):
    _write(tmp_path / "data/Jinan/3_4/anon_3_4_jinan_real.json")
    _write(tmp_path / "data/Jinan/3_4/roadnet_3_4.json")
    _write(tmp_path / "records/jinan/seed0/transitions_raw.jsonl", json.dumps({"transition_id": "t0"}) + "\n")
    _write(
        tmp_path / "artifacts/seed0/diffusion_scores.jsonl",
        json.dumps(
            {
                "transition_id": "t0",
                "rarity_score": 1.0,
                "support_score": 0.5,
                "paper_claim_eligible": True,
                "adapter": "traffic_feature_density_proxy",
                "model_checkpoint": "checkpoints/r2v_diffusion_seed0.pt",
                "config_hash": "abc123",
                "normalization_id": "jinan_norm_v1",
            }
        )
        + "\n",
    )

    report = check_r2v_traffic_readiness(
        root=tmp_path,
        scenario="jinan",
        transition_glob="records/jinan/seed{seed}/transitions_raw.jsonl",
        seeds=(0,),
        diffusion_artifacts={0: "artifacts/seed0/diffusion_scores.jsonl"},
        require_diffusion_artifacts=True,
        require_paper_claim_eligible_diffusion=True,
    )

    assert report["status"] == "BLOCKED"
    failure = next(item for item in report["failed_checks"] if item["name"] == "diffusion_score_artifact")
    assert failure["paper_claim_proxy_adapter_count"] == 1
    assert failure["paper_claim_proxy_adapter_transition_ids"] == ["t0"]
    assert "uses proxy adapter" in failure["message"]


def test_readiness_rejects_malformed_diffusion_artifact(tmp_path: Path):
    _write(tmp_path / "data/Jinan/3_4/anon_3_4_jinan_real.json")
    _write(tmp_path / "data/Jinan/3_4/roadnet_3_4.json")
    _write(tmp_path / "records/jinan/seed0/transitions_raw.jsonl", json.dumps({"transition_id": "t0"}) + "\n")
    _write(tmp_path / "artifacts/seed0/diffusion_scores.jsonl", json.dumps({"transition_id": "t0"}) + "\n")

    report = check_r2v_traffic_readiness(
        root=tmp_path,
        scenario="jinan",
        transition_glob="records/jinan/seed{seed}/transitions_raw.jsonl",
        seeds=(0,),
        diffusion_artifacts={0: "artifacts/seed0/diffusion_scores.jsonl"},
        require_diffusion_artifacts=True,
    )

    assert report["status"] == "BLOCKED"
    failure = next(item for item in report["failed_checks"] if item["name"] == "diffusion_score_artifact")
    assert "missing rarity_score" in failure["message"]


def test_readiness_rejects_diffusion_artifact_that_does_not_cover_transitions(tmp_path: Path):
    _write(tmp_path / "data/Jinan/3_4/anon_3_4_jinan_real.json")
    _write(tmp_path / "data/Jinan/3_4/roadnet_3_4.json")
    _write(
        tmp_path / "records/jinan/seed0/transitions_raw.jsonl",
        "\n".join(
            [
                json.dumps({"transition_id": "t0"}),
                json.dumps({"transition_id": "missing_from_scores"}),
            ]
        )
        + "\n",
    )
    _write(
        tmp_path / "artifacts/seed0/diffusion_scores.jsonl",
        json.dumps({"transition_id": "t0", "rarity_score": 1.0, "support_score": 0.5}) + "\n",
    )

    report = check_r2v_traffic_readiness(
        root=tmp_path,
        scenario="jinan",
        transition_glob="records/jinan/seed{seed}/transitions_raw.jsonl",
        seeds=(0,),
        diffusion_artifacts={0: "artifacts/seed0/diffusion_scores.jsonl"},
        require_diffusion_artifacts=True,
    )

    assert report["status"] == "BLOCKED"
    failure = next(item for item in report["failed_checks"] if item["name"] == "diffusion_score_artifact")
    assert failure["missing_transition_count"] == 1
    assert failure["missing_transition_ids"] == ["missing_from_scores"]


def test_readiness_requires_diffusion_artifact_mapping_per_seed(tmp_path: Path):
    _write(tmp_path / "data/Jinan/3_4/anon_3_4_jinan_real.json")
    _write(tmp_path / "data/Jinan/3_4/roadnet_3_4.json")
    _write(tmp_path / "records/jinan/seed0/transitions_raw.jsonl", "{}\n")

    report = check_r2v_traffic_readiness(
        root=tmp_path,
        scenario="jinan",
        transition_glob="records/jinan/seed{seed}/transitions_raw.jsonl",
        seeds=(0,),
        require_diffusion_artifacts=True,
    )

    assert report["status"] == "BLOCKED"
    assert any(item["name"] == "diffusion_score_artifact" for item in report["failed_checks"])


def test_readiness_rejects_performance_rows_missing_throughput(tmp_path: Path):
    perf = _write(
        tmp_path / "metrics.jsonl",
        json.dumps(
            {
                "method": "r2v",
                "average_travel_time": 10.0,
                "queue_length": 2.0,
                "delay": 1.0,
                "reward": -1.0,
                "status": "DONE",
            }
        )
        + "\n",
    )

    report = check_r2v_traffic_readiness(
        root=tmp_path,
        require_cityflow_data=False,
        performance_paths=[perf],
        require_performance_metrics=True,
    )

    assert report["status"] == "BLOCKED"
    metric_failure = next(item for item in report["failed_checks"] if item["name"] == "performance_metrics")
    assert metric_failure["missing_rows"][0]["missing"] == ["throughput"]


def test_readiness_accepts_legacy_metric_aliases_when_throughput_is_present(tmp_path: Path):
    perf = _write(
        tmp_path / "metrics.jsonl",
        json.dumps(
            {
                "method": "baseline",
                "test_avg_travel_time_over": 10.0,
                "test_avg_queue_len_over": 2.0,
                "test_avg_waiting_time_over": 1.0,
                "throughput": 5.0,
                "test_reward_over": -1.0,
            }
        )
        + "\n",
    )

    report = check_r2v_traffic_readiness(
        root=tmp_path,
        require_cityflow_data=False,
        performance_paths=[perf],
        require_performance_metrics=True,
    )

    assert report["status"] == "READY"


def _performance_row(method: str, seed: int, *, status: str = "DONE") -> dict:
    return {
        "method": method,
        "seed": seed,
        "average_travel_time": 10.0 + seed,
        "queue_length": 2.0,
        "delay": 1.0,
        "throughput": 5.0,
        "reward": -1.0,
        "status": status,
    }


def test_readiness_rejects_incomplete_main_performance_method_seed_grid(tmp_path: Path):
    perf = _write(
        tmp_path / "metrics.jsonl",
        "\n".join(
            json.dumps(row)
            for row in [
                _performance_row("baseline_uniform", 0),
                _performance_row("r2v_diffusion_not_rare_to_val_full", 0),
                _performance_row("baseline_uniform", 1),
            ]
        )
        + "\n",
    )

    report = check_r2v_traffic_readiness(
        root=tmp_path,
        require_cityflow_data=False,
        performance_paths=[perf],
        require_performance_metrics=True,
        expected_performance_methods=("baseline_uniform", "r2v_diffusion_not_rare_to_val_full"),
        expected_performance_seeds=(0, 1),
        require_completed_performance_status=True,
    )

    assert report["status"] == "BLOCKED"
    coverage_failure = next(item for item in report["failed_checks"] if item["name"] == "performance_coverage")
    assert coverage_failure["missing_method_seed_pairs"] == [
        {"method": "r2v_diffusion_not_rare_to_val_full", "seed": 1}
    ]


def test_readiness_rejects_unfinished_main_performance_rows(tmp_path: Path):
    perf = _write(
        tmp_path / "metrics.jsonl",
        "\n".join(
            json.dumps(row)
            for row in [
                _performance_row("baseline_uniform", 0, status="DONE"),
                _performance_row("r2v_diffusion_not_rare_to_val_full", 0, status="RUNNING"),
            ]
        )
        + "\n",
    )

    report = check_r2v_traffic_readiness(
        root=tmp_path,
        require_cityflow_data=False,
        performance_paths=[perf],
        require_performance_metrics=True,
        expected_performance_methods=("baseline_uniform", "r2v_diffusion_not_rare_to_val_full"),
        expected_performance_seeds=(0,),
        require_completed_performance_status=True,
    )

    assert report["status"] == "BLOCKED"
    coverage_failure = next(item for item in report["failed_checks"] if item["name"] == "performance_coverage")
    assert coverage_failure["unfinished_rows"] == [{"method": "r2v_diffusion_not_rare_to_val_full", "seed": 0, "status": "RUNNING"}]


def test_readiness_accepts_complete_main_performance_grid(tmp_path: Path):
    perf = _write(
        tmp_path / "metrics.jsonl",
        "\n".join(
            json.dumps(row)
            for seed in (0, 1, 2)
            for row in [
                _performance_row("baseline_uniform", seed),
                _performance_row("r2v_diffusion_not_rare_to_val_full", seed),
            ]
        )
        + "\n",
    )

    report = check_r2v_traffic_readiness(
        root=tmp_path,
        require_cityflow_data=False,
        performance_paths=[perf],
        require_performance_metrics=True,
        expected_performance_methods=("baseline_uniform", "r2v_diffusion_not_rare_to_val_full"),
        expected_performance_seeds=(0, 1, 2),
        require_completed_performance_status=True,
    )

    assert report["status"] == "READY"
    coverage = next(item for item in report["checks"] if item["name"] == "performance_coverage")
    assert coverage["expected_row_count"] == 6


def test_readiness_rejects_proxy_repair_metadata_policy_for_strict_paper_runs(tmp_path: Path):
    report = check_r2v_traffic_readiness(
        root=tmp_path,
        require_cityflow_data=False,
        repair_metadata_policy="metadata_or_proxy",
        require_strict_repair_metadata_policy=True,
    )

    assert report["status"] == "BLOCKED"
    failure = next(item for item in report["failed_checks"] if item["name"] == "repair_metadata_policy")
    assert failure["repair_metadata_policy"] == "metadata_or_proxy"
    assert "require_metadata" in failure["message"]


def test_readiness_accepts_require_metadata_policy_for_strict_paper_runs(tmp_path: Path):
    report = check_r2v_traffic_readiness(
        root=tmp_path,
        require_cityflow_data=False,
        repair_metadata_policy="require_metadata",
        require_strict_repair_metadata_policy=True,
    )

    assert report["status"] == "READY"
    check = next(item for item in report["checks"] if item["name"] == "repair_metadata_policy")
    assert check["status"] == "pass"


def test_parse_args_accepts_performance_grid_flags(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "experiment_readiness.py",
            "--no-require_cityflow_data",
            "--performance_path",
            "metrics.jsonl",
            "--require_performance_metrics",
            "--expected_performance_method",
            "baseline_uniform",
            "--expected_performance_method",
            "r2v_diffusion_not_rare_to_val_full",
            "--expected_performance_seed",
            "0",
            "--expected_performance_seed",
            "1",
            "--expected_performance_seed",
            "2",
            "--require_completed_performance_status",
        ],
    )

    args = parse_args()

    assert args.expected_performance_method == ["baseline_uniform", "r2v_diffusion_not_rare_to_val_full"]
    assert args.expected_performance_seed == [0, 1, 2]
    assert args.require_completed_performance_status is True


def test_parse_args_accepts_strict_repair_metadata_policy_flags(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "experiment_readiness.py",
            "--no-require_cityflow_data",
            "--repair_metadata_policy",
            "require_metadata",
            "--require_strict_repair_metadata_policy",
        ],
    )

    args = parse_args()

    assert args.repair_metadata_policy == "require_metadata"
    assert args.require_strict_repair_metadata_policy is True


def test_parse_seed_artifact_requires_seed_path_pairs():
    assert parse_seed_artifact(["0:a.jsonl", "2:b.jsonl"]) == {0: "a.jsonl", 2: "b.jsonl"}
    with pytest.raises(ValueError, match="seed:path"):
        parse_seed_artifact(["missing_separator"])
