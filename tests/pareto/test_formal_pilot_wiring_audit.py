from __future__ import annotations

import json
from pathlib import Path

import pytest

from pareto.rl.formal_pilot_wiring_audit import build_formal_pilot_wiring_guard_audit


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / "metadata.json").write_text(
        json.dumps(
            {
                "formal_pilot_wiring_dry_run": True,
                "real_env_rollout": True,
                "cityflow_step_called": True,
                "real_ppo_update": True,
                "formal_experiment": False,
                "performance_claim": False,
                "traffic_result_value_reading_executed": False,
                "method_ranking_executed": False,
                "paper_result_claim": False,
                "seed_expansion_allowed": False,
                "city_expansion_allowed": False,
                "formal_experiment_requires_new_pro_approval": True,
                "reward_adapter": "vector_quality_potential",
                "representation_run_id": "v3_rev15_m02_iso3_c15_u03",
                "vector_model_dir": "model_weights/pareto_quality/jinan/eval_consistency_remediation_v3/v3_rev15_m02_iso3_c15_u03",
                "vector_model_hash": "abc",
                "vector_model_hash_expected": "abc",
                "vector_model_hash_verified": True,
                "vector_model_loaded": True,
                "objective_normalizer_hash_verified": True,
                "state_encoder_hash": "4d1c2b4e276043ac",
                "obs_dim": 193,
                "checkpoint_load_verified": True,
                "checkpoint_valid": True,
                "policy_update_count": 1,
                "action_guard": {
                    "unique_actions_used": 4,
                    "global_single_action_rate": 0.25,
                    "max_single_action_rate_allowed": 0.95,
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "status.json").write_text(
        json.dumps(
            {
                "status": "FORMAL_PILOT_WIRING_DRY_RUN_DONE",
                "reward_row_count": 2,
                "policy_update_count": 1,
                "checkpoint_load_verified": True,
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(run / "reward_components.jsonl", [{"total_reward": 1.0}, {"total_reward": -0.5}])
    _write_jsonl(run / "loss_debug.jsonl", [{"total_loss": 0.3}, {"policy_loss": 0.1, "value_loss": 0.2}])
    _write_jsonl(run / "train_metrics.jsonl", [{"reward_finite": True}])
    (run / "checkpoint_last.pt").write_bytes(b"checkpoint")
    (run / "training_checkpoint_last.pt").write_bytes(b"training")
    return run


def test_formal_pilot_wiring_audit_passes_guard_only_run(tmp_path: Path):
    run = _run_dir(tmp_path)

    audit = build_formal_pilot_wiring_guard_audit(
        run,
        guard_commit="3cdb59d",
        representation_commit="a872e68",
        execution_commit="local",
    )

    assert audit["overall_pass"] is True
    assert audit["closed_loop_executed"]["pass"] is True
    assert audit["vectorq_wiring"]["pass"] is True
    assert audit["reward_finite_nonzero"]["pass"] is True
    assert audit["loss_finite"]["pass"] is True
    assert audit["action_non_collapse"]["pass"] is True
    assert audit["checkpoint_roundtrip"]["pass"] is True
    assert audit["scope_flags"]["pass"] is True
    assert audit["forbidden_artifact_scan"]["pass"] is True


def test_formal_pilot_wiring_audit_rejects_unverified_vector_hash(tmp_path: Path):
    run = _run_dir(tmp_path)
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    metadata["vector_model_hash_verified"] = False
    (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    audit = build_formal_pilot_wiring_guard_audit(
        run,
        guard_commit="3cdb59d",
        representation_commit="a872e68",
        execution_commit="local",
    )

    assert audit["overall_pass"] is False
    assert audit["vectorq_wiring"]["pass"] is False


def test_formal_pilot_wiring_audit_rejects_forbidden_performance_artifact(tmp_path: Path):
    run = _run_dir(tmp_path)
    (run / "performance_table.csv").write_text("forbidden\n", encoding="utf-8")

    audit = build_formal_pilot_wiring_guard_audit(
        run,
        guard_commit="3cdb59d",
        representation_commit="a872e68",
        execution_commit="local",
    )

    assert audit["overall_pass"] is False
    assert audit["forbidden_artifact_scan"]["pass"] is False
    assert any("performance_table.csv" in item for item in audit["forbidden_artifact_scan"]["forbidden"])
