from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pareto.rl.formal_jinan_3seed_reference_eval_evaluator_adapter import (
    build_guarded_reference_eval_evaluator,
)
from pareto.rl.formal_jinan_3seed_reference_eval_evaluator_binding import (
    ReferenceEvalCityFlowEnvWrapper,
    build_formal_reference_eval_env_factory,
    build_formal_reference_eval_ppo_policy_loader,
    build_formal_reference_eval_reference_policy_factory,
    build_reference_eval_evaluator_binding_packet,
)
from pareto.rl.formal_jinan_3seed_reference_eval_evaluator_adapter import (
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
from pareto.rl.state_encoder import feature_values_hash


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
            model = PreferenceConditionedActorCritic(obs_dim=3, preference_dim=4, action_dim=4, hidden_dim=8)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            for parameter in model.parameters():
                parameter.data.zero_()
            model.actor.bias.data[:] = model.actor.bias.new_tensor([0.1, 1.5, -0.3, 0.9])
            metadata = _training_metadata(method=method, seed=seed)
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


def _packets(tmp_path: Path) -> tuple[dict, Path, Path, Path, Path]:
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
    build_reference_eval_evaluator_adapter_packet(
        out_dir=tmp_path / "adapter",
        runner_packet_path=runner_path,
    )
    adapter_path = tmp_path / "adapter" / "formal_jinan_3seed_reference_eval_evaluator_adapter.json"
    return guard_packet, runner_path, adapter_path, train_root, eval_root


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


def test_ppo_policy_loader_uses_argmax_and_rejects_nondeterministic_flags(tmp_path: Path):
    guard_packet, _runner_path, _adapter_path, _train_root, _eval_root = _packets(tmp_path)
    row = guard_packet["future_eval_manifest"][0]
    loader = build_formal_reference_eval_ppo_policy_loader(
        determinism_probe_observation={"obs_features": [0.0, 0.0, 0.0], "w": [0.25, 0.25, 0.25, 0.25]},
    )

    policy = loader(
        checkpoint_path=Path(row["checkpoint_last_path"]),
        request={"seed_id": 0},
        manifest_row=row,
    )

    obs = {"obs_features": [2.0, -1.0, 0.5], "w": [0.25, 0.25, 0.25, 0.25]}
    assert policy.select_action(
        obs,
        deterministic=True,
        temperature=0.0,
        stochastic_sampling_allowed=False,
        exploration_noise_allowed=False,
    ) == 1
    assert policy.select_action(
        obs,
        deterministic=True,
        temperature=0.0,
        stochastic_sampling_allowed=False,
        exploration_noise_allowed=False,
    ) == 1
    with pytest.raises(ValueError, match="deterministic=True"):
        policy.select_action(
            obs,
            deterministic=False,
            temperature=0.0,
            stochastic_sampling_allowed=False,
            exploration_noise_allowed=False,
        )


def test_ppo_policy_loader_supports_batched_2d_observations_and_preferences(tmp_path: Path):
    guard_packet, _runner_path, _adapter_path, _train_root, _eval_root = _packets(tmp_path)
    row = guard_packet["future_eval_manifest"][0]
    loader = build_formal_reference_eval_ppo_policy_loader(
        determinism_probe_observation={"obs_features": [0.0, 0.0, 0.0], "w": [0.25, 0.25, 0.25, 0.25]},
    )

    policy = loader(
        checkpoint_path=Path(row["checkpoint_last_path"]),
        request={"seed_id": 0},
        manifest_row=row,
    )

    actions = policy.select_action(
        {
            "obs_features": [[0.0, 0.0, 0.0], [3.0, -1.0, 0.5]],
            "w": [[1.0, 1.0, 1.0, 1.0], [0.7, 0.1, 0.1, 0.1]],
        },
        deterministic=True,
        temperature=0.0,
        stochastic_sampling_allowed=False,
        exploration_noise_allowed=False,
    )

    assert actions == [1, 1]


def test_reference_policy_factory_uses_deterministic_choose_action_without_kwargs_swallowing(tmp_path: Path):
    _guard_packet, _runner_path, _adapter_path, _train_root, _eval_root = _packets(tmp_path)
    agent_calls: list[dict] = []

    class FakeMaxPressureAgent:
        def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, intersection_id):
            self.intersection_id = intersection_id

        def choose_action(self, count, state):
            agent_calls.append({"count": count, "state": state, "intersection_id": self.intersection_id})
            return int(state["cur_phase"][0])

    def config_builder(**_kwargs):
        return (
            {"FIXED_TIME": [30, 30, 30, 30]},
            {"NUM_INTERSECTIONS": 2, "PHASE": {1: [], 2: [], 3: [], 4: []}},
            {"PATH_TO_WORK_DIRECTORY": "unused"},
        )

    factory = build_formal_reference_eval_reference_policy_factory(
        config_builder=config_builder,
        agent_class_map={"MaxPressure": FakeMaxPressureAgent},
    )
    policy = factory(method="MaxPressure", request={"seed_id": 0}, manifest_row={"method": "MaxPressure"})
    action = policy.select_action(
        {"step": 7, "states": [{"cur_phase": [2]}, {"cur_phase": [3]}]},
        deterministic=True,
        temperature=0.0,
        stochastic_sampling_allowed=False,
        exploration_noise_allowed=False,
    )

    assert action == [2, 3]
    assert [call["count"] for call in agent_calls] == [7, 7]
    with pytest.raises(ValueError, match="temperature=0.0"):
        policy.select_action(
            {"step": 7, "states": [{"cur_phase": [2]}, {"cur_phase": [3]}]},
            deterministic=True,
            temperature=0.2,
            stochastic_sampling_allowed=False,
            exploration_noise_allowed=False,
        )


def test_reference_policy_factory_does_not_require_encoder_or_obs_feature_schema(tmp_path: Path):
    _guard_packet, _runner_path, _adapter_path, _train_root, _eval_root = _packets(tmp_path)

    class FakeMaxPressureAgent:
        def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, intersection_id):
            self.intersection_id = intersection_id

        def choose_action(self, count, state):
            return int(state["cur_phase"][0])

    def config_builder(**_kwargs):
        return (
            {"FIXED_TIME": [30, 30, 30, 30]},
            {"NUM_INTERSECTIONS": 1, "PHASE": {1: [], 2: [], 3: [], 4: []}},
            {"PATH_TO_WORK_DIRECTORY": "unused"},
        )

    factory = build_formal_reference_eval_reference_policy_factory(
        config_builder=config_builder,
        agent_class_map={"MaxPressure": FakeMaxPressureAgent},
    )
    policy = factory(method="MaxPressure", request={"seed_id": 0}, manifest_row={"method": "MaxPressure"})

    assert policy.select_action(
        {"step": 0, "states": [{"cur_phase": [3]}]},
        deterministic=True,
        temperature=0.0,
        stochastic_sampling_allowed=False,
        exploration_noise_allowed=False,
    ) == [3]


def test_env_factory_wrapper_exposes_budget_bound_interface_and_common_metrics(tmp_path: Path):
    created: list[dict] = []

    class FakeIntersection:
        def __init__(self, queue):
            self.dic_feature = {"lane_num_waiting_vehicle_in": queue}
            self.dic_vehicle_arrive_leave_time = {
                f"veh_{idx}": {"enter_time": 0.0, "leave_time": float(idx + 1)}
                for idx in range(len(queue))
            }

        def get_dic_vehicle_arrive_leave_time(self):
            return self.dic_vehicle_arrive_leave_time

    class FakeCityFlowEnv:
        def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf, dic_path):
            created.append(
                {
                    "path_to_log": path_to_log,
                    "path_to_work_directory": path_to_work_directory,
                    "dic_traffic_env_conf": dict(dic_traffic_env_conf),
                    "dic_path": dict(dic_path),
                }
            )
            self.list_intersection = [FakeIntersection([1, 2]), FakeIntersection([3, 4])]
            self.actions: list[list[int]] = []
            self.step_index = 0

        def reset(self):
            self.step_index = 0
            return [{"cur_phase": [0]}, {"cur_phase": [1]}]

        def step(self, action):
            self.actions.append(list(action))
            self.step_index += 1
            return ([{"cur_phase": [1]}, {"cur_phase": [2]}], [0.0, 0.0], False, [0.0, 0.0])

    def config_builder(**kwargs):
        return (
            {"FIXED_TIME": [30, 30, 30, 30]},
            {
                "NUM_INTERSECTIONS": 2,
                "PHASE": {1: [], 2: [], 3: [], 4: []},
                "CITYFLOW_SEED": kwargs["seed"],
                "MIN_ACTION_TIME": kwargs["min_action_time"],
            },
            {
                "PATH_TO_WORK_DIRECTORY": str(tmp_path / "work"),
                "PATH_TO_MODEL": str(tmp_path / "model"),
                "PATH_TO_DATA": "data/template/Jinan/3_4",
            },
        )

    env_factory = build_formal_reference_eval_env_factory(
        config_builder=config_builder,
        env_class=FakeCityFlowEnv,
    )
    env = env_factory(
        scenario="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        method="MaxPressure",
        seed_id=2,
        cityflow_seed=2,
        min_action_time=30,
        episodes=5,
        max_decision_steps_per_episode=120,
        stochastic_sampling_allowed=False,
        exploration_noise_allowed=False,
        temperature=0.0,
    )

    obs0 = env.reset(episode=0, seed=2)
    assert obs0["step"] == 0
    assert len(obs0["states"]) == 2
    obs1 = env.step([1, 2], min_action_time=30)
    assert obs1["step"] == 1
    assert created[0]["dic_traffic_env_conf"]["CITYFLOW_SEED"] == 2
    assert env._env.actions == [[1, 2]]
    metrics = env.common_metrics()
    assert set(metrics) == {"average_travel_time", "throughput", "mean_queue_length"}
    assert metrics["throughput"] > 0


def test_env_wrapper_rejects_wrong_joint_action_length():
    class FakeEnv:
        def __init__(self):
            self.list_intersection = [object(), object()]

        def reset(self):
            return [{"cur_phase": [0]}, {"cur_phase": [1]}]

        def step(self, action):
            return ([{"cur_phase": [0]}, {"cur_phase": [1]}], [0.0, 0.0], False, {})

    env = ReferenceEvalCityFlowEnvWrapper(
        FakeEnv(),
        dic_traffic_env_conf={"NUM_INTERSECTIONS": 2, "PHASE": {1: [], 2: [], 3: [], 4: []}},
        min_action_time=30,
    )
    env.reset(episode=0, seed=0)

    with pytest.raises(ValueError, match="joint action length"):
        env.step([1], min_action_time=30)


def test_env_wrapper_rejects_invalid_action_ids_and_bool_actions():
    class FakeEnv:
        def __init__(self):
            self.list_intersection = [object()]

        def reset(self):
            return [{"cur_phase": [0]}]

        def step(self, action):
            return ([{"cur_phase": [0]}], [0.0], False, {})

    env = ReferenceEvalCityFlowEnvWrapper(
        FakeEnv(),
        dic_traffic_env_conf={"NUM_INTERSECTIONS": 1, "PHASE": {1: [], 2: [], 3: [], 4: []}},
        min_action_time=30,
    )
    env.reset(episode=0, seed=0)

    with pytest.raises(ValueError, match="invalid action id"):
        env.step([4], min_action_time=30)
    with pytest.raises(ValueError, match="deterministic policy action"):
        env.step([True], min_action_time=30)


def test_env_wrapper_fails_closed_when_feature_encoder_raises(monkeypatch):
    import pareto.data.snapshot as snapshot_module

    class FakeEnv:
        list_intersection = [object()]

        def reset(self):
            return [{"cur_phase": [0]}]

    class FailingEncoder:
        def encode_snapshot(self, _snapshot):
            raise RuntimeError("encoder exploded")

    monkeypatch.setattr(snapshot_module, "capture_snapshot", lambda _env, _idx: object())

    env = ReferenceEvalCityFlowEnvWrapper(
        FakeEnv(),
        dic_traffic_env_conf={"MIN_ACTION_TIME": 30},
        min_action_time=30,
        encoder=FailingEncoder(),
    )

    with pytest.raises(ValueError, match="failed to encode obs_features"):
        env.reset(episode=0, seed=0)


def test_env_wrapper_rejects_empty_obs_features_on_step(monkeypatch):
    import pareto.data.snapshot as snapshot_module

    class FakeIntersection:
        dic_feature = {"lane_num_waiting_vehicle_in": [1.0]}

    class FakeEnv:
        list_intersection = [FakeIntersection()]

        def reset(self):
            return [{"cur_phase": [0]}]

        def step(self, _action):
            return ([{"cur_phase": [1]}], [0.0], False, [0.0])

    class Encoder:
        def __init__(self):
            self.calls = 0

        def encode_snapshot(self, _snapshot):
            self.calls += 1
            if self.calls == 1:
                return [1.0], ["feature_0"], {"feature_names_hash": "hash_a"}
            return [], [], {"feature_names_hash": "hash_a"}

    monkeypatch.setattr(snapshot_module, "capture_snapshot", lambda _env, _idx: object())

    env = ReferenceEvalCityFlowEnvWrapper(
        FakeEnv(),
        dic_traffic_env_conf={"MIN_ACTION_TIME": 30},
        min_action_time=30,
        encoder=Encoder(),
    )
    env.reset(episode=0, seed=0)

    with pytest.raises(ValueError, match="empty obs_features"):
        env.step([0], min_action_time=30)


def test_env_wrapper_rejects_nonfinite_obs_features_on_step(monkeypatch):
    import pareto.data.snapshot as snapshot_module

    class FakeIntersection:
        dic_feature = {"lane_num_waiting_vehicle_in": [1.0]}

    class FakeEnv:
        list_intersection = [FakeIntersection()]

        def reset(self):
            return [{"cur_phase": [0]}]

        def step(self, _action):
            return ([{"cur_phase": [1]}], [0.0], False, [0.0])

    class Encoder:
        def __init__(self):
            self.calls = 0

        def encode_snapshot(self, _snapshot):
            self.calls += 1
            if self.calls == 1:
                return [1.0], ["feature_0"], {"feature_names_hash": "hash_a"}
            return [float("nan")], ["feature_0"], {"feature_names_hash": "hash_a"}

    monkeypatch.setattr(snapshot_module, "capture_snapshot", lambda _env, _idx: object())

    env = ReferenceEvalCityFlowEnvWrapper(
        FakeEnv(),
        dic_traffic_env_conf={"MIN_ACTION_TIME": 30},
        min_action_time=30,
        encoder=Encoder(),
    )
    env.reset(episode=0, seed=0)

    with pytest.raises(ValueError, match="non-finite obs_features"):
        env.step([0], min_action_time=30)


def test_env_wrapper_rejects_feature_schema_hash_drift(monkeypatch):
    import pareto.data.snapshot as snapshot_module

    class FakeIntersection:
        dic_feature = {"lane_num_waiting_vehicle_in": [1.0]}

    class FakeEnv:
        list_intersection = [FakeIntersection()]

        def reset(self):
            return [{"cur_phase": [0]}]

        def step(self, _action):
            return ([{"cur_phase": [1]}], [0.0], False, [0.0])

    class Encoder:
        def __init__(self):
            self.calls = 0

        def encode_snapshot(self, _snapshot):
            self.calls += 1
            if self.calls == 1:
                return [1.0], ["feature_0"], {"feature_names_hash": "hash_a"}
            return [2.0], ["feature_0"], {"feature_names_hash": "hash_b"}

    monkeypatch.setattr(snapshot_module, "capture_snapshot", lambda _env, _idx: object())

    env = ReferenceEvalCityFlowEnvWrapper(
        FakeEnv(),
        dic_traffic_env_conf={"MIN_ACTION_TIME": 30},
        min_action_time=30,
        encoder=Encoder(),
    )
    env.reset(episode=0, seed=0)

    with pytest.raises(ValueError, match="obs_feature schema hash drift"):
        env.step([0], min_action_time=30)


def test_env_wrapper_records_obs_feature_integrity_debug(monkeypatch):
    import pareto.data.snapshot as snapshot_module

    class FakeIntersection:
        dic_feature = {"lane_num_waiting_vehicle_in": [1.0]}

        def get_dic_vehicle_arrive_leave_time(self):
            return {"veh_0": {"enter_time": 0.0, "leave_time": 5.0}}

    class FakeEnv:
        list_intersection = [FakeIntersection()]

        def reset(self):
            return [{"cur_phase": [0]}]

        def step(self, _action):
            self.list_intersection[0].dic_feature = {"lane_num_waiting_vehicle_in": [2.0]}
            return ([{"cur_phase": [1]}], [0.0], False, [0.0])

    class Encoder:
        def __init__(self):
            self.calls = 0

        def encode_snapshot(self, _snapshot):
            self.calls += 1
            return [float(self.calls)], ["feature_0"], {"feature_names_hash": "hash_a"}

    monkeypatch.setattr(snapshot_module, "capture_snapshot", lambda _env, _idx: object())

    env = ReferenceEvalCityFlowEnvWrapper(
        FakeEnv(),
        dic_traffic_env_conf={"MIN_ACTION_TIME": 30},
        min_action_time=30,
        encoder=Encoder(),
    )
    env.reset(episode=0, seed=0)
    env.step([0], min_action_time=30)
    env.common_metrics()

    integrity = env.common_metric_debug()["obs_feature_integrity"]
    assert integrity["feature_integrity_pass"] is True
    assert integrity["feature_row_count"] == 2
    assert integrity["first_obs_feature_sha256"] == feature_values_hash([1.0])
    assert integrity["final_obs_feature_sha256"] == feature_values_hash([2.0])


def test_env_wrapper_rejects_constant_obs_feature_sequence_in_common_metrics(monkeypatch):
    import pareto.data.snapshot as snapshot_module

    class FakeIntersection:
        dic_feature = {"lane_num_waiting_vehicle_in": [1.0]}

        def get_dic_vehicle_arrive_leave_time(self):
            return {"veh_0": {"enter_time": 0.0, "leave_time": 5.0}}

    class FakeEnv:
        list_intersection = [FakeIntersection()]

        def reset(self):
            return [{"cur_phase": [0]}]

        def step(self, _action):
            return ([{"cur_phase": [0]}], [0.0], False, [0.0])

    class Encoder:
        def encode_snapshot(self, _snapshot):
            return [1.0], ["feature_0"], {"feature_names_hash": "hash_a"}

    monkeypatch.setattr(snapshot_module, "capture_snapshot", lambda _env, _idx: object())

    env = ReferenceEvalCityFlowEnvWrapper(
        FakeEnv(),
        dic_traffic_env_conf={"MIN_ACTION_TIME": 30},
        min_action_time=30,
        encoder=Encoder(),
    )
    env.reset(episode=0, seed=0)
    env.step([0], min_action_time=30)

    with pytest.raises(ValueError, match="constant observation feature sequence"):
        env.common_metrics()


def test_env_wrapper_rejects_nonfinite_travel_time_metric():
    class FakeIntersection:
        dic_feature = {"lane_num_waiting_vehicle_in": [1.0]}

        def get_dic_vehicle_arrive_leave_time(self):
            return {"veh_0": {"enter_time": 0.0, "leave_time": float("inf")}}

    class FakeEnv:
        list_intersection = [FakeIntersection()]

    env = ReferenceEvalCityFlowEnvWrapper(
        FakeEnv(),
        dic_traffic_env_conf={"MIN_ACTION_TIME": 30},
        min_action_time=30,
    )

    with pytest.raises(ValueError, match="non-finite travel time"):
        env.common_metrics()


def test_env_wrapper_skips_incomplete_nan_leave_times_and_records_count():
    class FakeIntersection:
        dic_feature = {"lane_num_waiting_vehicle_in": [1.0]}

        def get_dic_vehicle_arrive_leave_time(self):
            return {
                "completed": {"enter_time": 2.0, "leave_time": 8.0},
                "incomplete": {"enter_time": 3.0, "leave_time": float("nan")},
            }

    class FakeEnv:
        list_intersection = [FakeIntersection()]

    env = ReferenceEvalCityFlowEnvWrapper(
        FakeEnv(),
        dic_traffic_env_conf={"MIN_ACTION_TIME": 30},
        min_action_time=30,
    )

    metrics = env.common_metrics()

    assert metrics["average_travel_time"] == 6.0
    assert metrics["throughput"] == 1.0
    assert env.common_metric_debug()["completed_vehicle_count"] == 1
    assert env.common_metric_debug()["incomplete_vehicle_count"] == 1
    assert env.common_metric_debug()["att_definition"] == "completed_vehicle_mean_finite_leave_minus_enter"


def test_env_wrapper_fails_when_no_completed_vehicle_travel_times():
    class FakeIntersection:
        dic_feature = {"lane_num_waiting_vehicle_in": [1.0]}

        def get_dic_vehicle_arrive_leave_time(self):
            return {
                "incomplete_0": {"enter_time": 2.0, "leave_time": float("nan")},
                "incomplete_1": {"enter_time": 3.0, "leave_time": float("nan")},
            }

    class FakeEnv:
        list_intersection = [FakeIntersection()]

    env = ReferenceEvalCityFlowEnvWrapper(
        FakeEnv(),
        dic_traffic_env_conf={"MIN_ACTION_TIME": 30},
        min_action_time=30,
    )

    with pytest.raises(ValueError, match="undefined without completed vehicle"):
        env.common_metrics()
    assert env.common_metric_debug()["completed_vehicle_count"] == 0
    assert env.common_metric_debug()["incomplete_vehicle_count"] == 2


def test_env_wrapper_reset_clears_queue_observations_between_episodes():
    class FakeIntersection:
        def __init__(self):
            self.dic_feature = {"lane_num_waiting_vehicle_in": [10.0]}

        def get_dic_vehicle_arrive_leave_time(self):
            return {"veh_0": {"enter_time": 0.0, "leave_time": 4.0}}

    class FakeEnv:
        def __init__(self):
            self.list_intersection = [FakeIntersection()]
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1
            queue = 10.0 if self.reset_count == 1 else 2.0
            self.list_intersection[0].dic_feature = {"lane_num_waiting_vehicle_in": [queue]}
            return [{"cur_phase": [0]}]

    env = ReferenceEvalCityFlowEnvWrapper(
        FakeEnv(),
        dic_traffic_env_conf={"MIN_ACTION_TIME": 30},
        min_action_time=30,
    )

    env.reset(episode=0, seed=0)
    assert env.common_metrics()["mean_queue_length"] == 10.0
    env.reset(episode=1, seed=0)

    assert env.common_metrics()["mean_queue_length"] == 2.0


def test_env_wrapper_rejects_nonfinite_queue_observation():
    class FakeIntersection:
        dic_feature = {"lane_num_waiting_vehicle_in": [float("inf")]}

    class FakeEnv:
        list_intersection = [FakeIntersection()]

        def reset(self):
            return [{"cur_phase": [0]}]

    env = ReferenceEvalCityFlowEnvWrapper(
        FakeEnv(),
        dic_traffic_env_conf={"MIN_ACTION_TIME": 30},
        min_action_time=30,
    )

    with pytest.raises(ValueError, match="non-finite queue observation"):
        env.reset(episode=0, seed=0)


def test_binding_factories_connect_through_guarded_adapter_with_joint_actions(tmp_path: Path):
    guard_packet, _runner_path, _adapter_path, _train_root, eval_root = _packets(tmp_path)

    class FakeEnv:
        def __init__(self, **_kwargs):
            self.actions: list[list[int]] = []

        def reset(self, *, episode: int, seed: int):
            return {"step": 0, "states": [{"cur_phase": [0]}, {"cur_phase": [1]}]}

        def step(self, action, *, min_action_time: int):
            self.actions.append(list(action))
            return {"step": len(self.actions), "states": [{"cur_phase": [1]}, {"cur_phase": [2]}]}

        def common_metrics(self):
            return {"average_travel_time": 1.0, "throughput": 2.0, "mean_queue_length": 3.0}

        def common_metric_debug(self):
            return {
                "att_definition": "completed_vehicle_mean_finite_leave_minus_enter",
                "completed_vehicle_count": 2,
                "incomplete_vehicle_count": 1,
                "observed_vehicle_rows": 3,
            }

    class FakePolicy:
        def select_action(
            self,
            observation,
            *,
            deterministic: bool,
            temperature: float,
            stochastic_sampling_allowed: bool,
            exploration_noise_allowed: bool,
        ):
            assert deterministic is True
            assert temperature == 0.0
            assert stochastic_sampling_allowed is False
            assert exploration_noise_allowed is False
            return [1, 2]

    evaluator = build_guarded_reference_eval_evaluator(
        guard_packet,
        ppo_policy_loader=lambda **_kwargs: FakePolicy(),
        reference_policy_factory=None,
        env_factory=lambda **kwargs: FakeEnv(**kwargs),
    )

    result = execute_guarded_reference_eval_request(_request(eval_root), guard_packet, evaluator=evaluator)

    assert result["status"] == "PASS"
    metadata = json.loads((Path(result["run_dir"]) / "metadata.json").read_text(encoding="utf-8"))
    guard_report = json.loads((Path(result["run_dir"]) / "eval_guard_report.json").read_text(encoding="utf-8"))
    assert metadata["common_metric_debug"]["incomplete_vehicle_count"] == 1
    assert guard_report["common_metric_debug"]["completed_vehicle_count"] == 2


def test_binding_packet_is_guard_only_and_pins_adapter_packet(tmp_path: Path):
    _guard_packet, _runner_path, adapter_path, _train_root, _eval_root = _packets(tmp_path)

    packet = build_reference_eval_evaluator_binding_packet(
        out_dir=tmp_path / "binding",
        adapter_packet_path=adapter_path,
        adapter_commit="e04d7ce",
    )

    assert packet["packet_type"] == "formal_jinan_3seed_reference_eval_evaluator_binding"
    assert packet["overall_pass"] is True
    assert packet["reference_eval_run_in_this_packet"] is False
    assert packet["cityflow_run_in_this_packet"] is False
    assert packet["model_rollout_in_this_packet"] is False
    assert packet["traffic_result_value_reading_in_this_packet"] is False
    assert packet["binding_contract"]["ppo_policy_loader_implemented"] is True
    assert packet["binding_contract"]["reference_policy_factory_implemented"] is True
    assert packet["binding_contract"]["env_factory_implemented"] is True
    assert packet["binding_contract"]["deterministic_probe_required"] is True
    assert packet["binding_contract"]["ppo_default_preference_locked"] is True
    assert packet["binding_contract"]["ppo_default_preference"] == [0.25, 0.25, 0.25, 0.25]
    assert packet["binding_contract"]["reference_policy_requires_encoder"] is False
    assert packet["binding_contract"]["obs_feature_schema_hash_enforced"] is True
    assert packet["binding_contract"]["common_metric_debug_recorded_to_metadata"] is True
    assert packet["provenance"]["adapter_packet_sha256"]
    assert packet["provenance"]["adapter_commit"] == "e04d7ce"
