from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pareto.r2v.build_generative_score_artifact import build_score_artifact_from_files
from pareto.r2v.experiment_readiness import check_r2v_traffic_readiness
from pareto.r2v.generative_scorer import load_generative_score_artifact


def _transition(transition_id: str, obs: list[float], next_obs: list[float]) -> dict:
    return {
        "schema_version": "pareto-transition-v1",
        "transition_id": transition_id,
        "sample_id": transition_id,
        "obs_features": obs,
        "next_obs_features": next_obs,
        "metadata": {},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_build_proxy_score_artifact_writes_loader_compatible_diffusion_rows(tmp_path: Path):
    transitions = _write_jsonl(
        tmp_path / "transitions.jsonl",
        [
            _transition("t0", [0.0, 0.0], [0.0, 0.0]),
            _transition("t1", [1.0, 1.0], [1.0, 1.0]),
            _transition("t2", [10.0, 10.0], [10.0, 10.0]),
        ],
    )
    output = tmp_path / "scores.jsonl"
    summary_output = tmp_path / "scores_summary.json"

    report = build_score_artifact_from_files(
        transitions=[transitions],
        output=output,
        summary_output=summary_output,
        backend="diffusion",
        adapter="traffic_feature_density_proxy",
    )

    rows = _read_jsonl(output)
    loaded, loaded_summary = load_generative_score_artifact(output, backend="diffusion")
    summary = json.loads(summary_output.read_text(encoding="utf-8"))

    assert report["score_count"] == 3
    assert loaded_summary["backend"] == "diffusion"
    assert set(loaded) == {"t0", "t1", "t2"}
    assert all(math.isfinite(row["rarity_score"]) for row in rows)
    assert all(row["support_score"] > 0.0 for row in rows)
    assert all(row["backend"] == "diffusion" for row in rows)
    assert all(row["adapter"] == "traffic_feature_density_proxy" for row in rows)
    assert all(row["paper_claim_eligible"] is False for row in rows)
    assert summary["adapter"] == "traffic_feature_density_proxy"
    assert summary["paper_claim_eligible"] is False
    assert summary["rare_is_not_value_boundary"] is True


def test_loader_preserves_repaired_transition_payload(tmp_path: Path):
    score_path = _write_jsonl(
        tmp_path / "diffusion_repair_scores.jsonl",
        [
            {
                "transition_id": "t0",
                "rarity_score": 1.0,
                "support_score": 0.8,
                "repaired_transition": {
                    "transition_id": "t0_repaired",
                    "sample_id": "t0_repaired",
                    "obs_features": [1.0],
                    "next_obs_features": [1.5],
                    "metadata": {},
                },
            }
        ],
    )

    loaded, _summary = load_generative_score_artifact(score_path, backend="diffusion")

    assert loaded["t0"].repaired_transition["transition_id"] == "t0_repaired"


def test_loader_rejects_non_object_repaired_transition_payload(tmp_path: Path):
    score_path = _write_jsonl(
        tmp_path / "bad_diffusion_repair_scores.jsonl",
        [
            {
                "transition_id": "t0",
                "rarity_score": 1.0,
                "support_score": 0.8,
                "repaired_transition": "not-an-object",
            }
        ],
    )

    with pytest.raises(ValueError, match="repaired_transition must be an object"):
        load_generative_score_artifact(score_path, backend="diffusion")


def test_build_proxy_score_artifact_rejects_duplicate_transition_ids(tmp_path: Path):
    transitions = _write_jsonl(
        tmp_path / "transitions.jsonl",
        [
            _transition("duplicate", [0.0], [0.0]),
            _transition("duplicate", [1.0], [1.0]),
        ],
    )

    with pytest.raises(ValueError, match="duplicate"):
        build_score_artifact_from_files(
            transitions=[transitions],
            output=tmp_path / "scores.jsonl",
            summary_output=tmp_path / "scores_summary.json",
            backend="diffusion",
            adapter="traffic_feature_density_proxy",
        )


def test_generated_proxy_score_artifact_satisfies_readiness_diffusion_check(tmp_path: Path):
    (tmp_path / "data/Jinan/3_4").mkdir(parents=True)
    (tmp_path / "data/Jinan/3_4/anon_3_4_jinan_real.json").write_text("{}", encoding="utf-8")
    (tmp_path / "data/Jinan/3_4/roadnet_3_4.json").write_text("{}", encoding="utf-8")
    transitions = _write_jsonl(
        tmp_path / "records/jinan/seed0/transitions_raw.jsonl",
        [
            _transition("t0", [0.0, 0.0], [0.0, 0.0]),
            _transition("t1", [2.0, 2.0], [2.0, 2.0]),
        ],
    )
    build_score_artifact_from_files(
        transitions=[transitions],
        output=tmp_path / "records/r2v_traffic/diffusion_seed0_scores.jsonl",
        summary_output=tmp_path / "records/r2v_traffic/diffusion_seed0_scores_summary.json",
        backend="diffusion",
        adapter="traffic_feature_density_proxy",
    )

    report = check_r2v_traffic_readiness(
        root=tmp_path,
        scenario="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        transition_glob="records/jinan/seed{seed}/transitions_raw.jsonl",
        seeds=(0,),
        diffusion_artifacts={0: "records/r2v_traffic/diffusion_seed0_scores.jsonl"},
        require_diffusion_artifacts=True,
    )

    assert report["status"] == "READY"


def test_generated_proxy_score_artifact_fails_strict_paper_readiness(tmp_path: Path):
    (tmp_path / "data/Jinan/3_4").mkdir(parents=True)
    (tmp_path / "data/Jinan/3_4/anon_3_4_jinan_real.json").write_text("{}", encoding="utf-8")
    (tmp_path / "data/Jinan/3_4/roadnet_3_4.json").write_text("{}", encoding="utf-8")
    transitions = _write_jsonl(
        tmp_path / "records/jinan/seed0/transitions_raw.jsonl",
        [_transition("t0", [0.0, 0.0], [0.0, 0.0])],
    )
    build_score_artifact_from_files(
        transitions=[transitions],
        output=tmp_path / "records/r2v_traffic/diffusion_seed0_scores.jsonl",
        summary_output=tmp_path / "records/r2v_traffic/diffusion_seed0_scores_summary.json",
        backend="diffusion",
        adapter="traffic_feature_density_proxy",
    )

    report = check_r2v_traffic_readiness(
        root=tmp_path,
        scenario="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        transition_glob="records/jinan/seed{seed}/transitions_raw.jsonl",
        seeds=(0,),
        diffusion_artifacts={0: "records/r2v_traffic/diffusion_seed0_scores.jsonl"},
        require_diffusion_artifacts=True,
        require_paper_claim_eligible_diffusion=True,
    )

    assert report["status"] == "BLOCKED"
    failure = next(item for item in report["failed_checks"] if item["name"] == "diffusion_score_artifact")
    assert failure["paper_claim_eligible"] is False
