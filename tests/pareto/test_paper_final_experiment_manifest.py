from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pareto.rl.paper_final_experiment_manifest import (
    PAPER_FINAL_SEEDS,
    REQUIRED_CITY_TRAFFIC,
    REQUIRED_PAPER_BASELINES,
    assert_ready_for_final_execution,
    manifest_with_metric_source_policy,
    manifest_with_baseline_registry,
    load_paper_final_manifest,
    paper_final_blockers,
    validate_paper_final_manifest,
)
from pareto.rl.paper_baseline_registry import BASELINE_REGISTRY, registry_with_learned_artifact_inventory


def _manifest() -> dict:
    return {
        "packet_type": "paper_final_experiment_manifest",
        "execution_allowed_now": False,
        "cities": [
            {"scenario": "jinan", "traffic_file": "anon_3_4_jinan_real.json"},
            {"scenario": "hangzhou", "traffic_file": "anon_4_4_hangzhou_real.json"},
            {"scenario": "newyork_28x7", "traffic_file": "anon_28_7_newyork_real_double.json"},
        ],
        "main_seed_ids": [0, 1, 2, 3, 4],
        "seed_binding": {
            "cityflow_seed": "seed_id",
            "policy_seed": "seed_id",
            "model_seed": "seed_id",
            "reference_policy_seed": "seed_id",
        },
        "preference_templates": {
            "efficiency_focused": [0.7, 0.1, 0.1, 0.1],
            "safety_focused": [0.1, 0.7, 0.1, 0.1],
            "fairness_focused": [0.1, 0.1, 0.7, 0.1],
            "stability_focused": [0.1, 0.1, 0.1, 0.7],
            "balanced": [0.25, 0.25, 0.25, 0.25],
        },
        "model": {"hidden_dim": 256, "preference_dim": 4, "action_dim": 4},
        "ppo": {
            "algorithm_label": "PPO",
            "hidden_dim": 256,
            "lr": 0.0003,
            "minibatch_size": 2048,
            "update_epochs": 10,
            "clip_eps": 0.2,
            "rollout_horizon_sim_steps": 3600,
            "total_env_steps_per_seed": 1_000_000,
            "shaping_warmup_start": 0.0,
            "shaping_warmup_end": 0.4,
            "shaping_warmup_fraction": 0.3,
        },
        "baselines": [
            {
                "name": name,
                "status": "implemented",
                "command_family": f"{name.lower()}_command",
                "city_support": ["jinan", "hangzhou", "newyork_28x7"],
            }
            for name in REQUIRED_PAPER_BASELINES
        ],
        "metric_families": {
            "efficiency": {"status": "implemented", "metrics": ["average_travel_time", "average_waiting_time", "average_queue_length"]},
            "safety": {"status": "implemented", "metrics": ["ttc_p10", "ttc_p50", "ttc_violation_rate", "harsh_brake_rate"]},
            "fairness": {"status": "implemented", "metrics": ["waiting_time_imbalance"]},
            "stability": {"status": "implemented", "metrics": ["phase_switch_rate", "oscillation_index"]},
            "representation": {"status": "implemented", "metrics": ["obj_acc", "pref_acc", "rev_acc", "dpr"]},
            "pareto": {"status": "implemented", "metrics": ["hypervolume", "coverage", "dominance_violation", "utility", "alignment"]},
            "controllability": {"status": "implemented", "metrics": ["monotonicity", "smoothness", "calibration_error"]},
            "generalization": {"status": "implemented", "metrics": ["heldout_preference_utility", "shifted_traffic_att"]},
        },
        "output_roots": {
            "train": "records/paper_final/train_20260602_v1",
            "eval": "records/paper_final/eval_20260602_v1",
            "diagnostics": "records/paper_final/diagnostics_20260602_v1",
            "preflight": "records/paper_final/preflight_20260602_v1",
        },
        "forbidden_actions": ["ranking", "plot", "paper_table", "paper_result_text"],
    }


def _complete_learned_artifact_inventory() -> dict:
    rows = []
    for baseline, family in (("Cond-Scalar-RL", "cond_scalar"), ("VectorQ-PPO", "pareto_quality")):
        for city in ("jinan", "hangzhou", "newyork_28x7"):
            rows.append(
                {
                    "baseline": baseline,
                    "city": city,
                    "status": "implemented_guarded_preview",
                    "model_path": f"model_weights/{family}/{city}/paper_final/run/model.pt",
                    "model_hash": "a" * 64,
                    "objective_normalizer_path": f"model_weights/{family}/{city}/paper_final/run/objective_normalizer.json",
                    "objective_normalizer_hash": "b" * 64,
                    "executes_training_now": False,
                }
            )
    return {
        "packet_type": "paper_learned_artifact_inventory",
        "coverage_status": "complete",
        "rows": rows,
        "executes_training_now": False,
    }


def test_manifest_accepts_paper_complete_scope():
    packet = validate_paper_final_manifest(_manifest())

    assert PAPER_FINAL_SEEDS == (0, 1, 2, 3, 4)
    assert set(REQUIRED_CITY_TRAFFIC) == {"jinan", "hangzhou", "newyork_28x7"}
    assert paper_final_blockers(packet) == []


def test_manifest_rejects_missing_hangzhou_or_newyork():
    packet = _manifest()
    packet["cities"] = [{"scenario": "jinan", "traffic_file": "anon_3_4_jinan_real.json"}]

    with pytest.raises(ValueError, match="missing required city"):
        validate_paper_final_manifest(packet)


def test_manifest_rejects_stage_a_scale_ppo():
    packet = _manifest()
    packet["model"]["hidden_dim"] = 64
    packet["ppo"]["minibatch_size"] = 12

    with pytest.raises(ValueError, match="paper-scale PPO"):
        validate_paper_final_manifest(packet)


def test_ready_for_final_execution_rejects_missing_blockers():
    packet = _manifest()
    packet["baselines"][0] = copy.deepcopy(packet["baselines"][0])
    packet["baselines"][0]["status"] = "missing_blocker"
    packet["baselines"][0]["blocker"] = "no importable implementation"

    validated = validate_paper_final_manifest(packet)
    blockers = paper_final_blockers(validated)

    assert blockers
    with pytest.raises(ValueError, match="final execution blocked"):
        assert_ready_for_final_execution(validated)


def test_load_manifest_from_config_file(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")

    assert load_paper_final_manifest(path)["packet_type"] == "paper_final_experiment_manifest"


def test_manifest_with_dynamic_registry_keeps_c2t_and_weighted_blockers_only_for_baselines():
    registry = registry_with_learned_artifact_inventory(
        BASELINE_REGISTRY,
        _complete_learned_artifact_inventory(),
    )

    manifest = manifest_with_baseline_registry(_manifest(), registry)
    blockers = paper_final_blockers(manifest)

    assert not any("Cond-Scalar-RL" in item for item in blockers)
    assert not any("VectorQ-PPO" in item for item in blockers)
    assert any("C2T-scalar" in item for item in blockers)
    assert any("Weighted-RL" in item for item in blockers)


def test_manifest_with_metric_source_policy_clears_representation_source_blocker_only():
    manifest = load_paper_final_manifest("configs/formal/paper_final_experiment_manifest_2026-06-02.json")
    policy = {
        "obj_acc": {"status": "implemented", "allowed_sources": ["representation_formal_gate_packet.json"]},
        "pref_acc": {"status": "implemented", "allowed_sources": ["representation_formal_gate_packet.json"]},
        "rev_acc": {"status": "implemented", "allowed_sources": ["representation_formal_gate_packet.json"]},
        "dpr": {"status": "implemented", "allowed_sources": ["representation_formal_gate_packet.json"]},
    }

    updated = manifest_with_metric_source_policy(manifest, policy)
    blockers = paper_final_blockers(updated)

    assert not any("metric family representation" in item for item in blockers)
    assert any("baseline C2T-scalar" in item for item in blockers)
    assert any("baseline Weighted-RL" in item for item in blockers)
