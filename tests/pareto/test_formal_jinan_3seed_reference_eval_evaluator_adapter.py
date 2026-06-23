from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pareto.rl.formal_jinan_3seed_reference_eval_evaluator_adapter import (
    build_guarded_reference_eval_evaluator,
    build_reference_eval_evaluator_adapter_packet,
)
from pareto.rl.formal_jinan_3seed_reference_eval_guard import build_reference_eval_guard_packet
from pareto.rl.formal_jinan_3seed_reference_eval_proposal import (
    FUTURE_REFERENCE_EVAL_APPROVAL_PHRASE,
    build_reference_eval_proposal_packet,
)
from pareto.rl.formal_jinan_3seed_reference_eval_runner import (
    build_reference_eval_runner_packet,
    execute_guarded_reference_eval_request,
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


def _packets(tmp_path: Path) -> tuple[dict, Path, Path, Path]:
    proposal = _write_proposal(tmp_path)
    train_root = tmp_path / "records" / "formal_jinan_3seed_guarded_20260601"
    eval_root = tmp_path / "records" / "formal_jinan_3seed_eval_20260601"
    _write_train_runs(train_root)
    guard_packet = build_reference_eval_guard_packet(
        out_dir=tmp_path / "guard",
        proposal_packet=proposal,
        train_run_root=train_root,
        eval_output_root=eval_root,
    )
    guard_path = tmp_path / "guard" / "formal_jinan_3seed_reference_eval_guard.json"
    build_reference_eval_runner_packet(
        out_dir=tmp_path / "runner",
        guard_packet_path=guard_path,
        guard_packet=guard_packet,
    )
    runner_path = tmp_path / "runner" / "formal_jinan_3seed_reference_eval_runner.json"
    return guard_packet, runner_path, train_root, eval_root


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


class MockPolicy:
    def __init__(self, action: int):
        self.action = action
        self.calls: list[dict] = []

    def select_action(self, observation: dict, **kwargs) -> int:
        self.calls.append({"observation": observation, **kwargs})
        return self.action


class MockEnv:
    def __init__(self, **kwargs):
        self.config = kwargs
        self.reset_calls: list[dict] = []
        self.step_calls: list[dict] = []

    def reset(self, *, episode: int, seed: int) -> dict:
        self.reset_calls.append({"episode": episode, "seed": seed})
        return {"episode": episode, "step": 0}

    def step(self, action: int, *, min_action_time: int) -> dict:
        self.step_calls.append({"action": action, "min_action_time": min_action_time})
        return {"episode": len(self.reset_calls) - 1, "step": len(self.step_calls)}

    def common_metrics(self) -> dict:
        return {
            "average_travel_time": 12.0,
            "throughput": float(len(self.step_calls)),
            "mean_queue_length": 1.5,
        }

    def common_metric_debug(self) -> dict:
        return {
            "att_definition": "completed_vehicle_mean_finite_leave_minus_enter",
            "completed_vehicle_count": 2,
            "incomplete_vehicle_count": 1,
            "observed_vehicle_rows": 3,
        }


def test_evaluator_adapter_uses_pinned_checkpoint_and_deterministic_policy(tmp_path: Path):
    guard_packet, _runner_path, _train_root, eval_root = _packets(tmp_path)
    loaded: list[dict] = []
    policies: list[MockPolicy] = []
    envs: list[MockEnv] = []

    def ppo_loader(*, checkpoint_path: Path, request: dict, manifest_row: dict) -> MockPolicy:
        loaded.append(
            {
                "checkpoint_path": checkpoint_path,
                "request": dict(request),
                "manifest_row": dict(manifest_row),
            }
        )
        policy = MockPolicy(action=2)
        policies.append(policy)
        return policy

    def env_factory(**kwargs) -> MockEnv:
        env = MockEnv(**kwargs)
        envs.append(env)
        return env

    evaluator = build_guarded_reference_eval_evaluator(
        guard_packet,
        ppo_policy_loader=ppo_loader,
        reference_policy_factory=None,
        env_factory=env_factory,
    )

    result = execute_guarded_reference_eval_request(
        _request(eval_root),
        guard_packet,
        evaluator=evaluator,
    )

    assert result["status"] == "PASS"
    assert len(loaded) == 1
    assert loaded[0]["checkpoint_path"] == Path(guard_packet["future_eval_manifest"][0]["checkpoint_last_path"])
    assert loaded[0]["manifest_row"]["checkpoint_last_sha256"]
    assert len(envs) == 1
    assert envs[0].config["scenario"] == "jinan"
    assert envs[0].config["traffic_file"] == "anon_3_4_jinan_real.json"
    assert envs[0].config["cityflow_seed"] == 0
    assert envs[0].config["episodes"] == 5
    assert envs[0].config["max_decision_steps_per_episode"] == 120
    assert envs[0].config["stochastic_sampling_allowed"] is False
    assert envs[0].config["exploration_noise_allowed"] is False
    assert envs[0].config["temperature"] == 0.0
    assert len(envs[0].reset_calls) == 5
    assert len(envs[0].step_calls) == 5 * 120
    assert len(policies[0].calls) == 5 * 120
    assert all(call["deterministic"] is True for call in policies[0].calls)
    assert all(call["temperature"] == 0.0 for call in policies[0].calls)
    assert all(call["stochastic_sampling_allowed"] is False for call in policies[0].calls)
    assert all(call["exploration_noise_allowed"] is False for call in policies[0].calls)


def test_evaluator_adapter_rejects_sha_drift_before_policy_loader(tmp_path: Path):
    guard_packet, _runner_path, train_root, eval_root = _packets(tmp_path)
    loader_called = False
    (train_root / "seed0" / "vector_quality_potential" / "checkpoint_last.pt").write_bytes(b"drift")

    def ppo_loader(**_kwargs):
        nonlocal loader_called
        loader_called = True
        return MockPolicy(action=0)

    evaluator = build_guarded_reference_eval_evaluator(
        guard_packet,
        ppo_policy_loader=ppo_loader,
        reference_policy_factory=None,
        env_factory=lambda **kwargs: MockEnv(**kwargs),
    )

    with pytest.raises(ValueError, match="checkpoint_last.pt sha256 mismatch"):
        evaluator(_request(eval_root), guard_packet["future_eval_manifest"][0])
    assert loader_called is False


def test_reference_policy_adapter_uses_deterministic_reference_policy_without_checkpoint(tmp_path: Path):
    guard_packet, _runner_path, _train_root, eval_root = _packets(tmp_path)
    ppo_loader_called = False
    reference_calls: list[dict] = []
    reference_policies: list[MockPolicy] = []

    def ppo_loader(**_kwargs):
        nonlocal ppo_loader_called
        ppo_loader_called = True
        return MockPolicy(action=0)

    def reference_policy_factory(*, method: str, request: dict, manifest_row: dict) -> MockPolicy:
        reference_calls.append({"method": method, "request": dict(request), "manifest_row": dict(manifest_row)})
        policy = MockPolicy(action=1)
        reference_policies.append(policy)
        return policy

    evaluator = build_guarded_reference_eval_evaluator(
        guard_packet,
        ppo_policy_loader=ppo_loader,
        reference_policy_factory=reference_policy_factory,
        env_factory=lambda **kwargs: MockEnv(**kwargs),
    )

    result = execute_guarded_reference_eval_request(
        _request(eval_root, method="AdvancedMaxPressure", seed_id=2),
        guard_packet,
        evaluator=evaluator,
    )

    assert result["status"] == "PASS"
    assert ppo_loader_called is False
    assert reference_calls[0]["method"] == "AdvancedMaxPressure"
    assert reference_calls[0]["manifest_row"]["checkpoint_last_path"] is None
    assert len(reference_policies[0].calls) == 5 * 120
    assert all(call["deterministic"] is True for call in reference_policies[0].calls)
    assert all(call["temperature"] == 0.0 for call in reference_policies[0].calls)
    assert all(call["stochastic_sampling_allowed"] is False for call in reference_policies[0].calls)
    assert all(call["exploration_noise_allowed"] is False for call in reference_policies[0].calls)


def test_evaluator_adapter_packet_is_guard_only_and_pins_runner_packet(tmp_path: Path):
    _guard_packet, runner_path, _train_root, _eval_root = _packets(tmp_path)

    packet = build_reference_eval_evaluator_adapter_packet(
        out_dir=tmp_path / "adapter",
        runner_packet_path=runner_path,
    )

    assert packet["packet_type"] == "formal_jinan_3seed_reference_eval_evaluator_adapter"
    assert packet["overall_pass"] is True
    assert packet["cityflow_run_in_this_packet"] is False
    assert packet["model_rollout_in_this_packet"] is False
    assert packet["traffic_result_value_reading_in_this_packet"] is False
    assert packet["adapter_contract"]["real_cityflow_evaluator_executed_in_this_packet"] is False
    assert packet["adapter_contract"]["mock_only_tests_required"] is True
    assert packet["provenance"]["runner_packet_sha256"]
    assert (tmp_path / "adapter" / "formal_jinan_3seed_reference_eval_evaluator_adapter.json").exists()
    assert (tmp_path / "adapter" / "formal_jinan_3seed_reference_eval_evaluator_adapter.md").exists()
