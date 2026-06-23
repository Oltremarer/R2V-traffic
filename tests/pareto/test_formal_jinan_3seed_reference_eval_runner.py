from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pareto.rl.formal_jinan_3seed_artifact_hash_verification import sha256_file
from pareto.rl.formal_jinan_3seed_reference_eval_guard import build_reference_eval_guard_packet
from pareto.rl.formal_jinan_3seed_reference_eval_proposal import (
    FUTURE_REFERENCE_EVAL_APPROVAL_PHRASE,
    build_reference_eval_proposal_packet,
)
from pareto.rl.formal_jinan_3seed_reference_eval_runner import (
    build_reference_eval_runner_packet,
    execute_guarded_reference_eval_request,
    recheck_checkpoint_sha_binding,
)
from pareto.rl.ppo_actor_critic import (
    PreferenceConditionedActorCritic,
    save_actor_critic_checkpoint,
    save_training_checkpoint,
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
            metadata = _training_metadata(method=method, seed=seed)
            model = PreferenceConditionedActorCritic(obs_dim=5, preference_dim=4, action_dim=4, hidden_dim=8)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            save_actor_critic_checkpoint(run_dir / "checkpoint_last.pt", model, metadata)
            save_training_checkpoint(
                run_dir / "training_checkpoint_last.pt",
                model,
                optimizer,
                metadata,
                step=1,
                episode=1,
                global_update=1,
            )
            _write_json(run_dir / "metadata.json", metadata)


def _training_metadata(*, method: str, seed: int) -> dict:
    return {
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
    }


def _manifest_row(packet: dict, *, method: str, seed: int) -> dict:
    for row in packet["future_eval_manifest"]:
        if row["method"] == method and int(row["seed"]) == int(seed):
            return row
    raise AssertionError("missing manifest row")


def _rewrite_checkpoint_with_metadata(path: Path, *, training: bool, metadata: dict) -> None:
    model = PreferenceConditionedActorCritic(obs_dim=5, preference_dim=4, action_dim=4, hidden_dim=8)
    if training:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        save_training_checkpoint(path, model, optimizer, metadata, step=1, episode=1, global_update=1)
    else:
        save_actor_critic_checkpoint(path, model, metadata)


def _guard_packet(tmp_path: Path) -> tuple[dict, Path, Path]:
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
    return packet, train_root, eval_root


def _request(eval_root: Path, *, method: str = "vector_quality_potential", seed_id: int = 0) -> dict:
    run_kind = "reference_policy_eval" if method in {"MaxPressure", "AdvancedMaxPressure"} else "ppo_checkpoint_eval"
    request = {
        "approval_phrase": FUTURE_REFERENCE_EVAL_APPROVAL_PHRASE,
        "method": method,
        "seed_id": seed_id,
        "run_kind": run_kind,
        "eval_output_dir": str(eval_root / f"seed{seed_id}" / method),
        "stochastic_sampling_allowed": False,
        "exploration_noise_allowed": False,
        "temperature": 0.0,
        "cityflow_seed": seed_id,
        "policy_seed": seed_id,
        "model_seed": seed_id,
        "min_action_time": 30,
        "episodes": 5,
        "max_decision_steps_per_episode": 120,
    }
    if run_kind == "ppo_checkpoint_eval":
        request["ppo_action_selection"] = "argmax_deterministic"
    else:
        request["reference_action_selection"] = "deterministic_policy"
        request["reference_policy_seed"] = seed_id
    return request


def _debug(*, completed: int = 2, incomplete: int = 1, observed: int = 3) -> dict:
    return {
        "att_definition": "completed_vehicle_mean_finite_leave_minus_enter",
        "completed_vehicle_count": completed,
        "incomplete_vehicle_count": incomplete,
        "observed_vehicle_rows": observed,
    }


def test_runner_packet_is_guard_only_and_pins_guard_packet(tmp_path: Path):
    guard_packet, _train_root, _eval_root = _guard_packet(tmp_path)
    guard_path = tmp_path / "guard" / "formal_jinan_3seed_reference_eval_guard.json"

    packet = build_reference_eval_runner_packet(
        out_dir=tmp_path / "runner",
        guard_packet_path=guard_path,
        guard_packet=guard_packet,
    )

    assert packet["packet_type"] == "formal_jinan_3seed_reference_eval_runner"
    assert packet["runner_status"] == "PASS"
    assert packet["reference_eval_run_in_this_packet"] is False
    assert packet["cityflow_run_in_this_packet"] is False
    assert packet["model_rollout_in_this_packet"] is False
    assert packet["traffic_result_value_reading_in_this_packet"] is False
    assert packet["runner_entrypoint"] == "execute_guarded_reference_eval_request"
    assert packet["runtime_guard_hooks"] == [
        "validate_reference_eval_execution_request",
        "recheck_checkpoint_sha_binding",
        "preflight_reference_eval_output_dir",
        "validate_common_metric_debug",
        "validate_reference_eval_output_allowlist",
    ]
    assert packet["provenance"]["guard_packet_sha256"] == sha256_file(guard_path)
    assert (tmp_path / "runner" / "formal_jinan_3seed_reference_eval_runner.json").exists()
    assert (tmp_path / "runner" / "formal_jinan_3seed_reference_eval_runner.md").exists()


def test_execute_guarded_reference_eval_request_rechecks_sha_and_writes_only_allowed_outputs(tmp_path: Path):
    guard_packet, _train_root, eval_root = _guard_packet(tmp_path)
    calls: list[dict] = []

    def fake_evaluator(request: dict, manifest_row: dict) -> dict:
        calls.append({"request": request, "manifest_row": manifest_row})
        return {
            "common_metrics": {
                "average_travel_time": 12.5,
                "throughput": 42.0,
                "mean_queue_length": 3.25,
            },
            "common_metric_debug": _debug(),
        }

    result = execute_guarded_reference_eval_request(
        _request(eval_root),
        guard_packet,
        evaluator=fake_evaluator,
    )

    run_dir = eval_root / "seed0" / "vector_quality_potential"
    assert result["status"] == "PASS"
    assert result["checkpoint_sha_rechecked"] is True
    assert len(calls) == 1
    assert calls[0]["manifest_row"]["checkpoint_last_sha256"]
    assert sorted(path.name for path in run_dir.iterdir()) == [
        "eval_guard_report.json",
        "eval_metrics.json",
        "eval_metrics.jsonl",
        "metadata.json",
        "status.json",
    ]
    metrics = json.loads((run_dir / "eval_metrics.json").read_text(encoding="utf-8"))
    assert metrics["average_travel_time"] == 12.5
    assert metrics["metric_interpretation"] == "raw_common_metric_no_ranking_no_claim"
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    guard_report = json.loads((run_dir / "eval_guard_report.json").read_text(encoding="utf-8"))
    assert metadata["common_metric_debug"]["incomplete_vehicle_count"] == 1
    assert guard_report["common_metric_debug"]["completed_vehicle_count"] == 2
    assert status["common_metric_debug_recorded"] is True
    assert metadata["cityflow_seed"] == 0
    assert metadata["policy_seed"] == 0
    assert metadata["model_seed"] == 0
    assert metadata["reference_policy_seed"] is None
    assert metadata["seed_binding"]["seed_binding"] == "cityflow_seed=policy_seed=model_seed=seed_id"
    assert metadata["eval_preference_policy"] == "balanced_primary_v1"
    assert metadata["eval_preference"] == [0.25, 0.25, 0.25, 0.25]
    assert metrics["eval_preference_policy"] == "balanced_primary_v1"
    assert metrics["eval_preference"] == [0.25, 0.25, 0.25, 0.25]
    assert guard_report["eval_preference_policy"] == "balanced_primary_v1"
    assert guard_report["eval_preference"] == [0.25, 0.25, 0.25, 0.25]


def test_execute_guarded_reference_eval_request_rejects_checkpoint_sha_drift(tmp_path: Path):
    guard_packet, train_root, eval_root = _guard_packet(tmp_path)
    (train_root / "seed0" / "vector_quality_potential" / "checkpoint_last.pt").write_bytes(b"drifted")

    with pytest.raises(ValueError, match="checkpoint_last.pt sha256 mismatch"):
        recheck_checkpoint_sha_binding(_request(eval_root), guard_packet)


def test_execute_guarded_reference_eval_request_rejects_training_metadata_drift(tmp_path: Path):
    guard_packet, train_root, eval_root = _guard_packet(tmp_path)
    _write_json(
        train_root / "seed0" / "vector_quality_potential" / "metadata.json",
        {
            "formal_jinan_3seed_execution": True,
            "scenario": "jinan",
            "traffic_file": "anon_3_4_jinan_real.json",
            "method": "weighted_proxy",
            "cityflow_seed": 0,
            "policy_seed": 0,
            "model_seed": 0,
            "performance_claim": False,
            "method_ranking_allowed": False,
            "performance_table_allowed": False,
            "traffic_result_value_reading_executed": False,
        },
    )

    with pytest.raises(ValueError, match="metadata.json sha256 mismatch"):
        recheck_checkpoint_sha_binding(_request(eval_root), guard_packet)


@pytest.mark.parametrize(
    ("path_key", "sha_key", "training", "label"),
    [
        ("checkpoint_last_path", "checkpoint_last_sha256", False, "checkpoint_last.pt"),
        ("training_checkpoint_last_path", "training_checkpoint_last_sha256", True, "training_checkpoint_last.pt"),
    ],
)
def test_recheck_rejects_checkpoint_payload_metadata_mismatch_even_when_sha_matches(
    tmp_path: Path,
    path_key: str,
    sha_key: str,
    training: bool,
    label: str,
):
    guard_packet, _train_root, eval_root = _guard_packet(tmp_path)
    row = _manifest_row(guard_packet, method="vector_quality_potential", seed=0)
    path = Path(row[path_key])
    bad_metadata = _training_metadata(method="weighted_proxy", seed=0)
    _rewrite_checkpoint_with_metadata(path, training=training, metadata=bad_metadata)
    row[sha_key] = sha256_file(path)

    with pytest.raises(ValueError, match=rf"{label} metadata.method mismatch"):
        recheck_checkpoint_sha_binding(_request(eval_root), guard_packet)


def test_reference_policy_request_uses_same_guard_path_without_checkpoint_load(tmp_path: Path):
    guard_packet, _train_root, eval_root = _guard_packet(tmp_path)

    def fake_reference_evaluator(request: dict, manifest_row: dict) -> dict:
        assert request["run_kind"] == "reference_policy_eval"
        assert request["reference_action_selection"] == "deterministic_policy"
        assert manifest_row["checkpoint_last_path"] is None
        return {
            "common_metrics": {
                "average_travel_time": 10.0,
                "throughput": 20.0,
                "mean_queue_length": 1.0,
            },
            "common_metric_debug": _debug(),
        }

    result = execute_guarded_reference_eval_request(
        _request(eval_root, method="MaxPressure", seed_id=1),
        guard_packet,
        evaluator=fake_reference_evaluator,
    )

    assert result["status"] == "PASS"
    assert result["checkpoint_sha_rechecked"] is False
    run_dir = eval_root / "seed1" / "MaxPressure"
    assert (run_dir / "eval_metrics.json").exists()
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["reference_policy_seed"] == 1
    assert metadata["seed_binding"]["reference_policy_seed_binding"] == "reference_policy_seed=seed_id"
    assert metadata["eval_preference_policy"] == "not_applicable_reference_policy"
    assert metadata["eval_preference"] is None


def test_execute_guarded_reference_eval_request_rejects_ranking_like_metrics(tmp_path: Path):
    guard_packet, _train_root, eval_root = _guard_packet(tmp_path)

    def bad_evaluator(_request: dict, _manifest_row: dict) -> dict:
        return {
            "common_metrics": {
                "average_travel_time": 12.5,
                "throughput": 42.0,
                "mean_queue_length": 3.25,
                "rank": 1,
            },
            "common_metric_debug": _debug(),
        }

    with pytest.raises(ValueError, match="forbidden metric field"):
        execute_guarded_reference_eval_request(
            _request(eval_root),
            guard_packet,
            evaluator=bad_evaluator,
        )


@pytest.mark.parametrize(
    ("artifact_rel", "content"),
    [
        ("metadata.json", "{}\n"),
        ("status.json", "{}\n"),
        ("nested/metadata.json", "{}\n"),
        ("ranking.csv", "method,value\n"),
    ],
)
def test_execute_guarded_reference_eval_request_rejects_nonempty_run_dir_before_evaluator(
    tmp_path: Path,
    artifact_rel: str,
    content: str,
):
    guard_packet, _train_root, eval_root = _guard_packet(tmp_path)
    run_dir = eval_root / "seed0" / "vector_quality_potential"
    stale = run_dir / artifact_rel
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text(content, encoding="utf-8")
    calls: list[dict] = []

    def fake_evaluator(request: dict, manifest_row: dict) -> dict:
        calls.append({"request": request, "manifest_row": manifest_row})
        return {
            "common_metrics": {
                "average_travel_time": 12.5,
                "throughput": 42.0,
                "mean_queue_length": 3.25,
            },
            "common_metric_debug": _debug(),
        }

    with pytest.raises(ValueError, match="output dir must be empty before execution"):
        execute_guarded_reference_eval_request(
            _request(eval_root),
            guard_packet,
            evaluator=fake_evaluator,
        )

    assert calls == []
    assert not (run_dir / "eval_metrics.json").exists()


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (
            {
                "common_metrics": {
                    "average_travel_time": 12.5,
                    "throughput": 42.0,
                    "mean_queue_length": 3.25,
                }
            },
            "common_metric_debug must be a nonempty object",
        ),
        (
            {
                "common_metrics": {
                    "average_travel_time": 12.5,
                    "throughput": 42.0,
                    "mean_queue_length": 3.25,
                },
                "common_metric_debug": {},
            },
            "common_metric_debug must be a nonempty object",
        ),
        (
            {
                "common_metrics": {
                    "average_travel_time": 12.5,
                    "throughput": 42.0,
                    "mean_queue_length": 3.25,
                },
                "common_metric_debug": {
                    "att_definition": "completed_vehicle_mean_finite_leave_minus_enter",
                    "completed_vehicle_count": 2,
                },
            },
            "missing common_metric_debug keys",
        ),
        (
            {
                "common_metrics": {
                    "average_travel_time": 12.5,
                    "throughput": 42.0,
                    "mean_queue_length": 3.25,
                },
                "common_metric_debug": _debug(completed=0, incomplete=2, observed=2),
            },
            "completed_vehicle_count must be positive",
        ),
    ],
)
def test_execute_guarded_reference_eval_request_requires_common_metric_debug(
    tmp_path: Path,
    payload: dict,
    match: str,
):
    guard_packet, _train_root, eval_root = _guard_packet(tmp_path)

    def bad_evaluator(_request: dict, _manifest_row: dict) -> dict:
        return payload

    with pytest.raises(ValueError, match=match):
        execute_guarded_reference_eval_request(
            _request(eval_root),
            guard_packet,
            evaluator=bad_evaluator,
        )

    run_dir = eval_root / "seed0" / "vector_quality_potential"
    assert not (run_dir / "eval_metrics.json").exists()
