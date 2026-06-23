from __future__ import annotations

import json
from pathlib import Path

import pytest

from pareto.rl.formal_jinan_3seed_artifact_hash_verification import sha256_file
from pareto.rl.formal_jinan_3seed_reference_eval_guard import (
    build_reference_eval_guard_packet,
    validate_reference_eval_execution_request,
    validate_reference_eval_output_allowlist,
)
from pareto.rl.formal_jinan_3seed_reference_eval_proposal import (
    FUTURE_REFERENCE_EVAL_APPROVAL_PHRASE,
    build_reference_eval_proposal_packet,
)


PPO_METHODS = ("vector_quality_potential", "film_scalar_potential", "weighted_proxy", "env_reward")
SEEDS = (0, 1, 2)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_analysis_packet(path: Path) -> None:
    _write_json(
        path,
        {
            "packet_type": "formal_jinan_3seed_descriptive_analysis",
            "analysis_status": "PASS",
            "permissions": {
                "method_ranking_executed": False,
                "performance_table_generated": False,
                "best_method_claim_generated": False,
                "traffic_improvement_claim_generated": False,
                "paper_ready_claim_generated": False,
                "comparison_requires_new_gate": True,
                "seed_expansion_executed": False,
                "city_expansion_executed": False,
                "not_for_main_results": True,
                "exclude_from_paper": True,
            },
            "provenance": {
                "guard_packet_sha256": "guard-sha",
                "verification_packet_sha256": "verification-sha",
                "request_packet_sha256": "request-sha",
                "execution_audit_packet_sha256": "execution-audit-sha",
                "execution_audit_commit": "d08ecc9",
            },
        },
    )


def _write_proposal(tmp_path: Path) -> Path:
    analysis = tmp_path / "analysis.json"
    _write_analysis_packet(analysis)
    proposal_dir = tmp_path / "proposal"
    build_reference_eval_proposal_packet(out_dir=proposal_dir, analysis_packet=analysis)
    return proposal_dir / "formal_jinan_3seed_reference_eval_proposal.json"


def _write_train_runs(root: Path) -> None:
    for seed in SEEDS:
        for method in PPO_METHODS:
            run_dir = root / f"seed{seed}" / method
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "checkpoint_last.pt").write_bytes(f"{method}-{seed}-policy".encode())
            (run_dir / "training_checkpoint_last.pt").write_bytes(f"{method}-{seed}-training".encode())
            _write_json(
                run_dir / "metadata.json",
                {
                    "formal_jinan_3seed_execution": True,
                    "scenario": "jinan",
                    "traffic_file": "anon_3_4_jinan_real.json",
                    "method": method,
                    "cityflow_seed": seed,
                    "policy_seed": seed,
                    "model_seed": seed,
                    "performance_claim": False,
                    "method_ranking_allowed": False,
                    "performance_table_allowed": False,
                    "traffic_result_value_reading_executed": False,
                },
            )


def test_reference_eval_guard_packet_locks_deterministic_policy_and_checkpoint_hashes(tmp_path: Path):
    proposal = _write_proposal(tmp_path)
    train_root = tmp_path / "records" / "formal_jinan_3seed_guarded_20260601"
    eval_root = tmp_path / "records" / "formal_jinan_3seed_eval_20260601"
    _write_train_runs(train_root)

    packet = build_reference_eval_guard_packet(
        out_dir=tmp_path / "guard",
        proposal_packet=proposal,
        train_run_root=train_root,
        eval_output_root=eval_root,
    )

    assert packet["overall_pass"] is True
    assert packet["formal_reference_eval_execution_allowed_now"] is False
    assert packet["reference_eval_run_in_this_packet"] is False
    assert packet["cityflow_run_in_this_packet"] is False
    assert packet["model_rollout_in_this_packet"] is False
    assert packet["traffic_result_value_reading_in_this_packet"] is False
    assert packet["numeric_traffic_aggregation_in_this_packet"] is False
    assert packet["method_ranking_in_this_packet"] is False

    deterministic = packet["runtime_eval_policy"]
    assert deterministic["ppo_action_selection"] == "argmax_deterministic"
    assert deterministic["reference_action_selection"] == "deterministic_policy"
    assert deterministic["stochastic_sampling_allowed"] is False
    assert deterministic["exploration_noise_allowed"] is False
    assert deterministic["temperature"] == 0.0
    assert deterministic["cityflow_seed_binding"] == "seed_id"

    runs = packet["future_eval_manifest"]
    assert len(runs) == 18
    ppo_runs = [run for run in runs if run["run_kind"] == "ppo_checkpoint_eval"]
    ref_runs = [run for run in runs if run["run_kind"] == "reference_policy_eval"]
    assert len(ppo_runs) == 12
    assert len(ref_runs) == 6
    for run in ppo_runs:
        checkpoint = train_root / f"seed{run['seed']}" / run["method"] / "checkpoint_last.pt"
        training_checkpoint = train_root / f"seed{run['seed']}" / run["method"] / "training_checkpoint_last.pt"
        assert run["checkpoint_last_sha256"] == sha256_file(checkpoint)
        assert run["training_checkpoint_last_sha256"] == sha256_file(training_checkpoint)
        assert run["checkpoint_load_required_before_eval"] is True

    assert packet["future_eval_artifact_policy"]["eval_output_root"] == str(eval_root)
    assert packet["provenance"]["proposal_packet_sha256"] == sha256_file(proposal)
    assert (tmp_path / "guard" / "formal_jinan_3seed_reference_eval_guard.json").exists()
    assert (tmp_path / "guard" / "formal_jinan_3seed_reference_eval_guard.md").exists()


def test_reference_eval_execution_request_rejects_wrong_phrase_and_non_deterministic_policy(tmp_path: Path):
    proposal = _write_proposal(tmp_path)
    train_root = tmp_path / "records" / "formal_jinan_3seed_guarded_20260601"
    eval_root = tmp_path / "records" / "formal_jinan_3seed_eval_20260601"
    _write_train_runs(train_root)
    packet = build_reference_eval_guard_packet(
        out_dir=tmp_path / "guard",
        proposal_packet=proposal,
        train_run_root=train_root,
        eval_output_root=eval_root,
    )

    request = {
        "approval_phrase": FUTURE_REFERENCE_EVAL_APPROVAL_PHRASE,
        "method": "vector_quality_potential",
        "seed_id": 0,
        "run_kind": "ppo_checkpoint_eval",
        "eval_output_dir": str(eval_root / "seed0" / "vector_quality_potential"),
        "ppo_action_selection": "argmax_deterministic",
        "stochastic_sampling_allowed": False,
        "exploration_noise_allowed": False,
        "temperature": 0.0,
        "cityflow_seed": 0,
        "policy_seed": 0,
        "model_seed": 0,
        "min_action_time": 30,
        "episodes": 5,
        "max_decision_steps_per_episode": 120,
    }
    validated = validate_reference_eval_execution_request(request, packet)
    assert validated["method"] == "vector_quality_potential"
    assert validated["seed_id"] == 0

    bad = dict(request)
    bad["approval_phrase"] = "wrong"
    with pytest.raises(ValueError, match="exact reference-eval approval phrase"):
        validate_reference_eval_execution_request(bad, packet)

    bad = dict(request)
    bad["ppo_action_selection"] = "sample"
    with pytest.raises(ValueError, match="argmax_deterministic"):
        validate_reference_eval_execution_request(bad, packet)

    bad = dict(request)
    bad["eval_output_dir"] = str(train_root / "seed0" / "vector_quality_potential")
    with pytest.raises(ValueError, match="eval_output_dir must be under eval_output_root"):
        validate_reference_eval_execution_request(bad, packet)

    for key in ("cityflow_seed", "policy_seed", "model_seed"):
        bad = dict(request)
        bad[key] = 1
        with pytest.raises(ValueError, match=f"{key} must equal seed_id"):
            validate_reference_eval_execution_request(bad, packet)

    ref_request = dict(request)
    ref_request.update(
        {
            "method": "MaxPressure",
            "seed_id": 1,
            "run_kind": "reference_policy_eval",
            "eval_output_dir": str(eval_root / "seed1" / "MaxPressure"),
            "cityflow_seed": 1,
            "policy_seed": 1,
            "model_seed": 1,
            "reference_action_selection": "deterministic_policy",
            "reference_policy_seed": 0,
        }
    )
    ref_request.pop("ppo_action_selection")
    with pytest.raises(ValueError, match="reference_policy_seed must equal seed_id"):
        validate_reference_eval_execution_request(ref_request, packet)


def test_reference_eval_output_allowlist_rejects_forbidden_or_extra_artifacts(tmp_path: Path):
    eval_root = tmp_path / "records" / "formal_jinan_3seed_eval_20260601"
    run_dir = eval_root / "seed0" / "vector_quality_potential"
    run_dir.mkdir(parents=True)
    for name in ("metadata.json", "status.json", "eval_metrics.jsonl", "eval_guard_report.json"):
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    validate_reference_eval_output_allowlist(run_dir, eval_root=eval_root)

    (run_dir / "ranking.csv").write_text("method,value\n", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden reference-eval artifacts"):
        validate_reference_eval_output_allowlist(run_dir, eval_root=eval_root)
    (run_dir / "ranking.csv").unlink()

    (run_dir / "action_debug.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-allowlisted reference-eval artifacts"):
        validate_reference_eval_output_allowlist(run_dir, eval_root=eval_root)


def test_reference_eval_output_allowlist_rejects_nested_and_requires_exact_root_set(tmp_path: Path):
    eval_root = tmp_path / "records" / "formal_jinan_3seed_eval_20260601"
    run_dir = eval_root / "seed0" / "vector_quality_potential"
    run_dir.mkdir(parents=True)
    required = (
        "metadata.json",
        "status.json",
        "eval_metrics.json",
        "eval_metrics.jsonl",
        "eval_guard_report.json",
    )
    for name in required:
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    validate_reference_eval_output_allowlist(run_dir, eval_root=eval_root, required_outputs=required)

    nested = run_dir / "nested" / "metadata.json"
    nested.parent.mkdir()
    nested.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nested reference-eval artifacts"):
        validate_reference_eval_output_allowlist(run_dir, eval_root=eval_root, required_outputs=required)
    nested.unlink()
    nested.parent.rmdir()

    (run_dir / "eval_guard_report.json").unlink()
    with pytest.raises(ValueError, match="exact required root-level set"):
        validate_reference_eval_output_allowlist(run_dir, eval_root=eval_root, required_outputs=required)
