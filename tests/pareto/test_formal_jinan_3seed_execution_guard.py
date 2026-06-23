from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pareto.rl.formal_jinan_3seed_execution_guard import (
    APPROVED_FORMAL_JINAN_PPO_METHODS,
    FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE,
    build_execution_guard_packet,
    build_formal_jinan_3seed_execution_manifest,
    load_formal_experiment_preregistration,
    validate_formal_experiment_preregistration,
)


PREREG_PATH = Path(
    "docs/pro_reviews/pareto_ppo_formal_experiment_preregistration_2026-06-01/"
    "formal_experiment_preregistration.json"
)
FORMAL_EXECUTION_SPEC_PATH = Path("configs/formal/jinan_3seed_ppo_formal_execution_locked.json")
WIRING_DRYRUN_SPEC_PATH = Path("configs/formal/jinan_seed0_vectorq_formal_pilot_wiring_dryrun.json")


def _packet() -> dict:
    return load_formal_experiment_preregistration(PREREG_PATH)


def test_execution_guard_builds_locked_four_method_three_seed_manifest():
    packet = _packet()

    manifest = build_formal_jinan_3seed_execution_manifest(
        packet,
        base_spec_path=str(FORMAL_EXECUTION_SPEC_PATH),
        out_root="records/formal_jinan_3seed_guarded_20260601",
    )

    assert len(manifest) == 12
    assert {entry["seed_id"] for entry in manifest} == {0, 1, 2}
    assert {entry["method"] for entry in manifest} == set(APPROVED_FORMAL_JINAN_PPO_METHODS)

    for entry in manifest:
        assert entry["scenario"] == "jinan"
        assert entry["traffic_file"] == "anon_3_4_jinan_real.json"
        assert entry["cityflow_seed"] == entry["seed_id"]
        assert entry["policy_seed"] == entry["seed_id"]
        assert entry["model_seed"] == entry["seed_id"]
        assert entry["episodes"] == 5
        assert entry["max_decision_steps_per_episode"] == 120
        assert entry["rollout_steps"] == 120
        assert entry["base_spec_rollout_steps"] == 120
        assert entry["min_action_time"] == 30
        assert entry["sim_seconds_per_method_seed"] == 3600
        assert entry["objective_normalizer_hash"] == "b2c55e7d2c42856a"
        assert entry["state_encoder_hash"] == "4d1c2b4e276043ac"
        assert entry["formal_execution_allowed_now"] is False
        assert entry["approval_phrase_required"] == FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE
        assert entry["command_kind"] == "template_only_not_executed"
        assert "--formal_jinan_3seed_execution" in entry["command_preview"]
        assert "--rollout_steps 120" in entry["command_preview"]

    vector_entries = [entry for entry in manifest if entry["method"] == "vector_quality_potential"]
    assert vector_entries
    assert all(entry["vector_model_hash"] for entry in vector_entries)
    assert all("--vector_model_hash" in entry["command_preview"] for entry in vector_entries)

    film_entries = [entry for entry in manifest if entry["method"] == "film_scalar_potential"]
    assert film_entries
    assert all(entry["film_model_hash"] for entry in film_entries)
    assert all("--film_model_hash" in entry["command_preview"] for entry in film_entries)


def test_preregistration_rejects_execution_or_analysis_permission():
    packet = _packet()
    packet["approval"]["formal_experiment_allowed_now"] = True

    with pytest.raises(ValueError, match="formal_experiment_allowed_now"):
        validate_formal_experiment_preregistration(packet)

    packet = _packet()
    packet["packet_scope"]["traffic_value_reading_in_this_packet"] = True

    with pytest.raises(ValueError, match="traffic_value_reading_in_this_packet"):
        validate_formal_experiment_preregistration(packet)


def test_manifest_rejects_method_or_budget_drift():
    packet = _packet()
    packet["methods"] = copy.deepcopy(packet["methods"])
    packet["methods"]["ppo_methods"] = [m for m in packet["methods"]["ppo_methods"] if m["method_id"] != "weighted_proxy"]

    with pytest.raises(ValueError, match="approved PPO methods"):
        build_formal_jinan_3seed_execution_manifest(
            packet,
            base_spec_path=str(FORMAL_EXECUTION_SPEC_PATH),
            out_root="records/formal_jinan_3seed_guarded_20260601",
        )

    packet = _packet()
    packet["budget"]["episodes_per_method_seed"] = 4

    with pytest.raises(ValueError, match="episodes_per_method_seed"):
        build_formal_jinan_3seed_execution_manifest(
            packet,
            base_spec_path=str(FORMAL_EXECUTION_SPEC_PATH),
            out_root="records/formal_jinan_3seed_guarded_20260601",
        )


def test_manifest_rejects_base_spec_rollout_drift():
    packet = _packet()

    with pytest.raises(ValueError, match="base_spec ppo.rollout_steps"):
        build_formal_jinan_3seed_execution_manifest(
            packet,
            base_spec_path=str(WIRING_DRYRUN_SPEC_PATH),
            out_root="records/formal_jinan_3seed_guarded_20260601",
        )


def test_execution_guard_packet_is_guard_only_and_machine_readable(tmp_path: Path):
    packet = _packet()
    out_dir = tmp_path / "guard_packet"

    guard = build_execution_guard_packet(
        packet,
        out_dir=out_dir,
        base_spec_path=str(FORMAL_EXECUTION_SPEC_PATH),
        out_root="records/formal_jinan_3seed_guarded_20260601",
        preregistration_commit="f0dbdcb",
        guard_build_commit="local",
        preregistration_packet_path=str(PREREG_PATH),
    )

    assert guard["overall_pass"] is True
    assert guard["formal_experiment_execution_in_this_packet"] is False
    assert guard["formal_execution_allowed_now"] is False
    assert guard["next_gate"]["required_exact_phrase"] == FORMAL_JINAN_3SEED_EXECUTION_APPROVAL_PHRASE
    assert len(guard["run_manifest"]) == 12
    assert (out_dir / "formal_jinan_3seed_execution_guard.json").exists()
    assert (out_dir / "formal_jinan_3seed_execution_guard.md").exists()

    loaded = json.loads((out_dir / "formal_jinan_3seed_execution_guard.json").read_text(encoding="utf-8"))
    assert loaded["overall_pass"] is True
    assert loaded["execution_guard_checks"]["manifest_shape"]["pass"] is True
    assert loaded["execution_guard_checks"]["base_spec_budget_consistency"]["pass"] is True
    assert loaded["execution_guard_checks"]["forbidden_actions_blocked"]["pass"] is True
    assert "formal_experiment_allowed_now" not in loaded["execution_guard_checks"]["forbidden_actions_blocked"]["lock_checks"]
    assert loaded["provenance"]["preregistration_packet"] == str(PREREG_PATH)
