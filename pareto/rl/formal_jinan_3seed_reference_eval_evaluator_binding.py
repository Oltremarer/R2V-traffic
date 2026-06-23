#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.rl.formal_jinan_3seed_artifact_hash_verification import sha256_file
from pareto.rl.formal_jinan_3seed_reference_eval_evaluator_adapter import (
    DEFAULT_ADAPTER_DIR,
    validate_reference_eval_evaluator_adapter_packet,
)
from pareto.rl.formal_jinan_3seed_reference_eval_runner import REQUIRED_COMMON_METRICS
from pareto.rl.state_encoder import validate_feature_integrity_sequence


DEFAULT_ADAPTER_PACKET = f"{DEFAULT_ADAPTER_DIR}/formal_jinan_3seed_reference_eval_evaluator_adapter.json"
DEFAULT_BINDING_DIR = (
    "docs/pro_reviews/pareto_ppo_formal_jinan_3seed_reference_eval_evaluator_binding_2026-06-01"
)
BINDING_PACKET_TYPE = "formal_jinan_3seed_reference_eval_evaluator_binding"
FORBIDDEN_BINDING_TRUE_FLAGS = (
    "reference_eval_run_in_this_packet",
    "cityflow_run_in_this_packet",
    "model_rollout_in_this_packet",
    "checkpoint_real_inference_in_this_packet",
    "traffic_result_value_reading_in_this_packet",
    "numeric_traffic_aggregation_in_this_packet",
    "method_ranking_in_this_packet",
    "performance_table_in_this_packet",
    "best_method_claim_in_this_packet",
    "traffic_improvement_claim_in_this_packet",
    "paper_result_claim_in_this_packet",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalize_preference(values: Sequence[float]) -> list[float]:
    total = float(sum(float(value) for value in values))
    if total <= 0.0:
        raise ValueError("preference weights must have positive sum")
    return [float(value) / total for value in values]


def _require_deterministic_flags(
    *,
    deterministic: bool,
    temperature: float,
    stochastic_sampling_allowed: bool,
    exploration_noise_allowed: bool,
) -> None:
    if deterministic is not True:
        raise ValueError("deterministic=True is required")
    if float(temperature) != 0.0:
        raise ValueError("temperature=0.0 is required")
    if stochastic_sampling_allowed is not False:
        raise ValueError("stochastic_sampling_allowed=False is required")
    if exploration_noise_allowed is not False:
        raise ValueError("exploration_noise_allowed=False is required")


class DeterministicPPOPolicy:
    def __init__(self, model: Any, *, default_preference: Sequence[float] | None = None) -> None:
        self.model = model
        self.model.eval()
        self.default_preference = _normalize_preference(default_preference or [0.25, 0.25, 0.25, 0.25])

    def _extract_features(self, observation: Any):
        import torch

        if isinstance(observation, dict):
            features = observation.get("obs_features", observation.get("obs"))
        else:
            features = observation
        if features is None:
            raise ValueError("PPO observation must contain obs_features or obs")
        tensor = torch.as_tensor(features, dtype=torch.float32)
        if tensor.ndim not in {1, 2}:
            raise ValueError("PPO observation features must be 1-D or 2-D")
        return tensor

    def _extract_preference(self, observation: Any, *, batch_size: int):
        import torch

        if isinstance(observation, dict):
            preference = observation.get("w", observation.get("preference", self.default_preference))
        else:
            preference = self.default_preference
        tensor = torch.as_tensor(preference, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = torch.as_tensor(_normalize_preference(tensor.tolist()), dtype=torch.float32)
        if tensor.ndim == 2:
            rows = [_normalize_preference(row.tolist()) for row in tensor]
            tensor = torch.as_tensor(rows, dtype=torch.float32)
        if tensor.ndim not in {1, 2}:
            raise ValueError("PPO preference must be 1-D or 2-D")
        if tensor.ndim == 2 and tensor.shape[0] != batch_size:
            raise ValueError("batched preference length must match batched observation length")
        return tensor

    def select_action(
        self,
        observation: Any,
        *,
        deterministic: bool,
        temperature: float,
        stochastic_sampling_allowed: bool,
        exploration_noise_allowed: bool,
    ) -> int | list[int]:
        _require_deterministic_flags(
            deterministic=deterministic,
            temperature=temperature,
            stochastic_sampling_allowed=stochastic_sampling_allowed,
            exploration_noise_allowed=exploration_noise_allowed,
        )
        import torch

        obs = self._extract_features(observation)
        is_batched = obs.ndim == 2
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        preference = self._extract_preference(observation, batch_size=int(obs.shape[0]))
        with torch.no_grad():
            logits, _values = self.model.forward(obs, preference)
            actions = torch.argmax(logits, dim=-1).cpu().tolist()
        return [int(action) for action in actions] if is_batched else int(actions[0])


def build_formal_reference_eval_ppo_policy_loader(
    *,
    determinism_probe_observation: dict[str, Any] | None = None,
    default_preference: Sequence[float] | None = None,
) -> Callable[[Path, dict[str, Any], dict[str, Any]], DeterministicPPOPolicy]:
    def loader(
        *,
        checkpoint_path: Path,
        request: dict[str, Any],
        manifest_row: dict[str, Any],
    ) -> DeterministicPPOPolicy:
        expected = Path(str(manifest_row.get("checkpoint_last_path")))
        if Path(checkpoint_path) != expected:
            raise ValueError("checkpoint_path must match manifest_row checkpoint_last_path")
        from pareto.rl.ppo_actor_critic import load_actor_critic_checkpoint

        model, _payload = load_actor_critic_checkpoint(checkpoint_path)
        policy = DeterministicPPOPolicy(model, default_preference=default_preference)
        probe = dict(determinism_probe_observation or {})
        if "obs_features" not in probe and "obs" not in probe:
            probe["obs_features"] = [0.0] * int(model.obs_dim)
        if "w" not in probe and "preference" not in probe:
            probe["w"] = list(policy.default_preference)
        action_1 = policy.select_action(
            probe,
            deterministic=True,
            temperature=0.0,
            stochastic_sampling_allowed=False,
            exploration_noise_allowed=False,
        )
        action_2 = policy.select_action(
            probe,
            deterministic=True,
            temperature=0.0,
            stochastic_sampling_allowed=False,
            exploration_noise_allowed=False,
        )
        if action_1 != action_2:
            raise ValueError("deterministic PPO argmax probe produced inconsistent actions")
        return policy

    return loader


class DeterministicReferencePolicy:
    def __init__(self, agents: Sequence[Any]) -> None:
        self.agents = list(agents)

    def select_action(
        self,
        observation: dict[str, Any],
        *,
        deterministic: bool,
        temperature: float,
        stochastic_sampling_allowed: bool,
        exploration_noise_allowed: bool,
    ) -> list[int]:
        _require_deterministic_flags(
            deterministic=deterministic,
            temperature=temperature,
            stochastic_sampling_allowed=stochastic_sampling_allowed,
            exploration_noise_allowed=exploration_noise_allowed,
        )
        states = observation.get("states")
        if not isinstance(states, list):
            raise ValueError("reference policy observation must contain states list")
        if len(states) != len(self.agents):
            raise ValueError("states length must match number of reference agents")
        step = int(observation.get("step", 0))
        return [int(agent.choose_action(step, state)) for agent, state in zip(self.agents, states)]


def _default_reference_agent_class_map() -> dict[str, Any]:
    from models.advanced_maxpressure_agent import AdvancedMaxPressureAgent
    from models.maxpressure_agent import MaxPressureAgent

    return {
        "MaxPressure": MaxPressureAgent,
        "AdvancedMaxPressure": AdvancedMaxPressureAgent,
    }


def build_formal_reference_eval_reference_policy_factory(
    *,
    config_builder: Callable[..., tuple[dict, dict, dict]] | None = None,
    agent_class_map: dict[str, Any] | None = None,
) -> Callable[[str, dict[str, Any], dict[str, Any]], DeterministicReferencePolicy]:
    def factory(
        *,
        method: str,
        request: dict[str, Any],
        manifest_row: dict[str, Any],
    ) -> DeterministicReferencePolicy:
        if method not in {"MaxPressure", "AdvancedMaxPressure"}:
            raise ValueError("reference_policy_factory only supports MaxPressure and AdvancedMaxPressure")
        builder = config_builder
        if builder is None:
            from pareto.common.scenario import build_llmlight_env_config

            builder = build_llmlight_env_config
        classes = agent_class_map or _default_reference_agent_class_map()
        if method not in classes:
            raise ValueError(f"missing reference agent class for {method}")
        dic_agent_conf, dic_traffic_env_conf, dic_path = builder(
            scenario="jinan",
            traffic_file="anon_3_4_jinan_real.json",
            seed=int(request.get("cityflow_seed", request.get("seed_id", 0))),
            run_counts=int(request.get("episodes", 5))
            * int(request.get("max_decision_steps_per_episode", 120))
            * int(request.get("min_action_time", 30)),
            min_action_time=int(request.get("min_action_time", 30)),
            model_name=method,
        )
        agent_class = classes[method]
        num_intersections = int(dic_traffic_env_conf.get("NUM_INTERSECTIONS", 1))
        agents = [
            agent_class(
                dic_agent_conf=dic_agent_conf,
                dic_traffic_env_conf=dic_traffic_env_conf,
                dic_path=dic_path,
                cnt_round=0,
                intersection_id=idx,
            )
            for idx in range(num_intersections)
        ]
        return DeterministicReferencePolicy(agents)

    return factory


class ReferenceEvalCityFlowEnvWrapper:
    def __init__(
        self,
        env: Any,
        *,
        dic_traffic_env_conf: dict[str, Any],
        min_action_time: int,
        encoder: Any | None = None,
    ) -> None:
        self._env = env
        self.dic_traffic_env_conf = dict(dic_traffic_env_conf)
        self.min_action_time = int(min_action_time)
        self.encoder = encoder
        self.step_index = 0
        self._queue_observations: list[float] = []
        self._last_common_metric_debug: dict[str, Any] = {}
        self._obs_feature_schema_hash: str | None = None
        self._obs_feature_rows: list[list[float]] = []

    def reset(self, *, episode: int, seed: int) -> dict[str, Any]:
        self.step_index = 0
        self._queue_observations = []
        self._last_common_metric_debug = {}
        self.dic_traffic_env_conf["CITYFLOW_SEED"] = int(seed)
        if hasattr(self._env, "dic_traffic_env_conf"):
            self._env.dic_traffic_env_conf["CITYFLOW_SEED"] = int(seed)
        states = self._env.reset()
        self._record_queue_observation()
        return self._observation(states)

    def step(self, action: int | Sequence[int], *, min_action_time: int) -> dict[str, Any]:
        if int(min_action_time) != self.min_action_time:
            raise ValueError("min_action_time mismatch")
        normalized_action = self._normalize_joint_action(action)
        result = self._env.step(normalized_action)
        self.step_index += 1
        states = result[0] if isinstance(result, tuple) else result
        self._record_queue_observation()
        return self._observation(states)

    def common_metrics(self) -> dict[str, float]:
        travel_times: list[float] = []
        incomplete_vehicle_count = 0
        observed_vehicle_rows = 0
        for inter in getattr(self._env, "list_intersection", []):
            getter = getattr(inter, "get_dic_vehicle_arrive_leave_time", None)
            if getter is None:
                continue
            for row in getter().values():
                if "enter_time" in row and "leave_time" in row:
                    observed_vehicle_rows += 1
                    enter_time = float(row["enter_time"])
                    leave_time = float(row["leave_time"])
                    if not math.isfinite(enter_time):
                        raise ValueError("non-finite travel time endpoint")
                    if math.isnan(leave_time):
                        incomplete_vehicle_count += 1
                        continue
                    if not math.isfinite(leave_time):
                        raise ValueError("non-finite travel time endpoint")
                    duration = leave_time - enter_time
                    if not math.isfinite(duration):
                        raise ValueError("non-finite travel time")
                    if duration < 0.0:
                        raise ValueError("negative travel time")
                    travel_times.append(duration)
        self._last_common_metric_debug = {
            "att_definition": "completed_vehicle_mean_finite_leave_minus_enter",
            "completed_vehicle_count": len(travel_times),
            "incomplete_vehicle_count": incomplete_vehicle_count,
            "observed_vehicle_rows": observed_vehicle_rows,
        }
        if self._obs_feature_rows:
            self._last_common_metric_debug["obs_feature_integrity"] = validate_feature_integrity_sequence(
                self._obs_feature_rows
            )
        if not travel_times:
            raise ValueError("average_travel_time is undefined without completed vehicle travel times")
        mean_queue = sum(self._queue_observations) / len(self._queue_observations) if self._queue_observations else 0.0
        if not math.isfinite(mean_queue):
            raise ValueError("non-finite mean_queue_length")
        metrics = {
            "average_travel_time": sum(travel_times) / len(travel_times),
            "throughput": float(len(travel_times)),
            "mean_queue_length": float(mean_queue),
        }
        if set(metrics) != set(REQUIRED_COMMON_METRICS):
            raise ValueError("common_metrics must return only required common metrics")
        return metrics

    def common_metric_debug(self) -> dict[str, Any]:
        return dict(self._last_common_metric_debug)

    def _observation(self, states: Any) -> dict[str, Any]:
        obs = {"step": int(self.step_index), "states": states}
        features = self._encode_current_features()
        if features:
            obs["obs_features"] = features
            obs["obs_feature_names_hash"] = self._obs_feature_schema_hash
        return obs

    def obs_feature_schema_hash(self) -> str | None:
        return self._obs_feature_schema_hash

    def _encode_current_features(self) -> list[list[float]]:
        if self.encoder is None:
            return []
        from pareto.data.snapshot import capture_snapshot

        intersections = getattr(self._env, "list_intersection", [])
        if not intersections:
            raise ValueError("cannot encode obs_features without intersections")
        encoded: list[list[float]] = []
        for idx in range(len(intersections)):
            try:
                features, names, debug = self.encoder.encode_snapshot(capture_snapshot(self._env, idx))
                values = features.tolist() if hasattr(features, "tolist") else list(features)
                feature_row = [float(value) for value in values]
            except Exception as exc:
                raise ValueError(f"failed to encode obs_features for intersection {idx}: {exc}") from exc
            if not feature_row:
                raise ValueError(f"empty obs_features for intersection {idx}")
            if any(not math.isfinite(value) for value in feature_row):
                raise ValueError(f"non-finite obs_features for intersection {idx}")
            row_hash = _feature_names_hash(names, debug)
            if self._obs_feature_schema_hash is None:
                self._obs_feature_schema_hash = row_hash
            elif row_hash != self._obs_feature_schema_hash:
                raise ValueError("obs_feature schema hash drift")
            encoded.append(feature_row)
        if encoded:
            self._obs_feature_rows.append([value for row in encoded for value in row])
        return encoded

    def _record_queue_observation(self) -> None:
        queues: list[float] = []
        for inter in getattr(self._env, "list_intersection", []):
            feature = getattr(inter, "dic_feature", {})
            value = feature.get("lane_num_waiting_vehicle_in", [])
            if isinstance(value, (list, tuple)):
                queues.extend(float(item) for item in value)
            elif value is not None:
                queues.append(float(value))
        if queues:
            if any(not math.isfinite(value) for value in queues):
                raise ValueError("non-finite queue observation")
            self._queue_observations.append(sum(queues) / len(queues))

    def _num_intersections(self) -> int:
        intersections = getattr(self._env, "list_intersection", None)
        if isinstance(intersections, Sequence) and not isinstance(intersections, (str, bytes)):
            return len(intersections)
        return int(self.dic_traffic_env_conf.get("NUM_INTERSECTIONS", 1))

    def _action_dim(self) -> int | None:
        phase = self.dic_traffic_env_conf.get("PHASE")
        if isinstance(phase, dict):
            return len(phase)
        if isinstance(phase, Sequence) and not isinstance(phase, (str, bytes)):
            return len(phase)
        action_dim = self.dic_traffic_env_conf.get("ACTION_DIM", self.dic_traffic_env_conf.get("NUM_PHASES"))
        if action_dim is None:
            return None
        return int(action_dim)

    def _normalize_joint_action(self, action: int | Sequence[int]) -> list[int]:
        num_intersections = self._num_intersections()
        if isinstance(action, bool):
            raise ValueError("deterministic policy action must be integer action ids, not bool")
        if isinstance(action, int):
            values = [int(action)]
        elif isinstance(action, Sequence) and not isinstance(action, (str, bytes)):
            values = []
            for value in action:
                if isinstance(value, bool):
                    raise ValueError("deterministic policy action must be integer action ids, not bool")
                if not isinstance(value, int):
                    raise ValueError("deterministic policy action must be integer action ids")
                values.append(int(value))
        else:
            raise ValueError("deterministic policy action must be integer action ids")
        if len(values) != num_intersections:
            raise ValueError(f"joint action length {len(values)} must match intersections {num_intersections}")
        action_dim = self._action_dim()
        if action_dim is not None:
            if action_dim <= 0:
                raise ValueError("action dimension must be positive")
            invalid = [value for value in values if value < 0 or value >= action_dim]
            if invalid:
                raise ValueError(f"invalid action id outside [0, {action_dim - 1}]: {invalid}")
        return values


def _feature_names_hash(names: Any, debug: Any) -> str:
    if isinstance(debug, dict) and debug.get("feature_names_hash"):
        return str(debug["feature_names_hash"])
    values = names.tolist() if hasattr(names, "tolist") else list(names or [])
    payload = json.dumps([str(value) for value in values], sort_keys=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_formal_reference_eval_env_factory(
    *,
    config_builder: Callable[..., tuple[dict, dict, dict]] | None = None,
    env_class: Any | None = None,
    encoder_factory: Callable[[], Any] | None = None,
) -> Callable[..., ReferenceEvalCityFlowEnvWrapper]:
    def factory(
        *,
        scenario: str,
        traffic_file: str,
        method: str,
        seed_id: int,
        cityflow_seed: int,
        min_action_time: int,
        episodes: int,
        max_decision_steps_per_episode: int,
        stochastic_sampling_allowed: bool,
        exploration_noise_allowed: bool,
        temperature: float,
    ) -> ReferenceEvalCityFlowEnvWrapper:
        _require_deterministic_flags(
            deterministic=True,
            temperature=temperature,
            stochastic_sampling_allowed=stochastic_sampling_allowed,
            exploration_noise_allowed=exploration_noise_allowed,
        )
        builder = config_builder
        if builder is None:
            from pareto.common.scenario import build_llmlight_env_config

            builder = build_llmlight_env_config
        cls = env_class
        if cls is None:
            from utils.cityflow_env import CityFlowEnv

            cls = CityFlowEnv
        dic_agent_conf, dic_traffic_env_conf, dic_path = builder(
            scenario=scenario,
            traffic_file=traffic_file,
            seed=int(cityflow_seed),
            run_counts=int(episodes) * int(max_decision_steps_per_episode) * int(min_action_time),
            min_action_time=int(min_action_time),
            model_name=method,
        )
        _ = dic_agent_conf
        env = cls(
            path_to_log=dic_path["PATH_TO_WORK_DIRECTORY"],
            path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
        )
        encoder = encoder_factory() if encoder_factory is not None else None
        return ReferenceEvalCityFlowEnvWrapper(
            env,
            dic_traffic_env_conf=dic_traffic_env_conf,
            min_action_time=int(min_action_time),
            encoder=encoder,
        )

    return factory


def validate_reference_eval_evaluator_binding_packet(packet: dict[str, Any]) -> None:
    if packet.get("packet_type") != BINDING_PACKET_TYPE:
        raise ValueError(f"packet_type must be {BINDING_PACKET_TYPE}")
    for key in FORBIDDEN_BINDING_TRUE_FLAGS:
        if packet.get(key) is not False:
            raise ValueError(f"{key} must be false")
    contract = packet.get("binding_contract") or {}
    required_true = (
        "ppo_policy_loader_implemented",
        "reference_policy_factory_implemented",
        "env_factory_implemented",
        "deterministic_probe_required",
        "ppo_argmax_required",
        "ppo_default_preference_locked",
        "obs_feature_schema_hash_enforced",
        "common_metric_debug_recorded_to_metadata",
    )
    for key in required_true:
        if contract.get(key) is not True:
            raise ValueError(f"{key} must be true")
    if float(contract.get("temperature_required", -1.0)) != 0.0:
        raise ValueError("temperature_required must be 0.0")
    if contract.get("common_metrics_only") != list(REQUIRED_COMMON_METRICS):
        raise ValueError("common_metrics_only must match required common metrics")
    if contract.get("ppo_default_preference") != [0.25, 0.25, 0.25, 0.25]:
        raise ValueError("ppo_default_preference must be balanced and locked")
    if contract.get("reference_policy_requires_encoder") is not False:
        raise ValueError("reference_policy_requires_encoder must be false")
    if int(contract.get("episodes", -1)) != 5:
        raise ValueError("episodes must be 5")
    if int(contract.get("max_decision_steps_per_episode", -1)) != 120:
        raise ValueError("max_decision_steps_per_episode must be 120")
    if packet.get("overall_pass") is not (not packet.get("failures")):
        raise ValueError("overall_pass must match failures")


def build_reference_eval_evaluator_binding_packet(
    *,
    out_dir: str | Path = DEFAULT_BINDING_DIR,
    adapter_packet_path: str | Path = DEFAULT_ADAPTER_PACKET,
    adapter_commit: str = "e04d7ce",
) -> dict[str, Any]:
    adapter_path = Path(adapter_packet_path)
    failures: list[str] = []
    if adapter_path.is_file():
        adapter_packet = _read_json(adapter_path)
        try:
            validate_reference_eval_evaluator_adapter_packet(adapter_packet)
        except ValueError as exc:
            failures.append(f"adapter packet: {exc}")
        adapter_sha = sha256_file(adapter_path)
    else:
        adapter_packet = {}
        adapter_sha = None
        failures.append(f"missing adapter packet: {adapter_path}")
    provenance = adapter_packet.get("provenance") or {}
    packet = {
        "packet_type": BINDING_PACKET_TYPE,
        "binding_status": "PASS" if not failures else "FAIL",
        "overall_pass": not failures,
        "failures": failures,
        "reference_eval_run_in_this_packet": False,
        "cityflow_run_in_this_packet": False,
        "model_rollout_in_this_packet": False,
        "checkpoint_real_inference_in_this_packet": False,
        "traffic_result_value_reading_in_this_packet": False,
        "numeric_traffic_aggregation_in_this_packet": False,
        "method_ranking_in_this_packet": False,
        "performance_table_in_this_packet": False,
        "best_method_claim_in_this_packet": False,
        "traffic_improvement_claim_in_this_packet": False,
        "paper_result_claim_in_this_packet": False,
        "binding_entrypoints": {
            "ppo_policy_loader": "build_formal_reference_eval_ppo_policy_loader",
            "reference_policy_factory": "build_formal_reference_eval_reference_policy_factory",
            "env_factory": "build_formal_reference_eval_env_factory",
            "evaluator_adapter": "build_guarded_reference_eval_evaluator",
        },
        "binding_contract": {
            "ppo_policy_loader_implemented": True,
            "reference_policy_factory_implemented": True,
            "env_factory_implemented": True,
            "deterministic_probe_required": True,
            "deterministic_probe_is_synthetic": True,
            "ppo_argmax_required": True,
            "ppo_default_preference_locked": True,
            "ppo_default_preference": [0.25, 0.25, 0.25, 0.25],
            "reference_policy_requires_encoder": False,
            "reference_policy_observation_source": "llmlight_state_list_only",
            "obs_feature_schema_hash_enforced": True,
            "common_metric_debug_recorded_to_metadata": True,
            "temperature_required": 0.0,
            "stochastic_sampling_allowed": False,
            "exploration_noise_allowed": False,
            "real_cityflow_rollout_executed_in_this_packet": False,
            "checkpoint_real_inference_executed_in_this_packet": False,
            "traffic_result_value_reading_in_this_packet": False,
            "episodes": 5,
            "min_action_time": 30,
            "max_decision_steps_per_episode": 120,
            "common_metrics_only": list(REQUIRED_COMMON_METRICS),
        },
        "provenance": {
            "adapter_packet": str(adapter_path),
            "adapter_packet_sha256": adapter_sha,
            "adapter_commit": adapter_commit,
            "runner_packet_sha256": provenance.get("runner_packet_sha256"),
            "runner_commit": provenance.get("runner_commit"),
            "guard_packet_sha256": provenance.get("guard_packet_sha256"),
            "proposal_packet_sha256": provenance.get("proposal_packet_sha256"),
            "analysis_packet_sha256": provenance.get("analysis_packet_sha256"),
            "execution_audit_packet_sha256": provenance.get("execution_audit_packet_sha256"),
        },
    }
    if packet["overall_pass"]:
        validate_reference_eval_evaluator_binding_packet(packet)
    output = Path(out_dir)
    _write_json(output / "formal_jinan_3seed_reference_eval_evaluator_binding.json", packet)
    write_markdown(packet, output / "formal_jinan_3seed_reference_eval_evaluator_binding.md")
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    return packet


def write_markdown(packet: dict[str, Any], path: str | Path) -> None:
    contract = packet["binding_contract"]
    lines = [
        "# Formal Jinan 3-Seed Reference-Eval Evaluator Binding",
        "",
        f"- binding_status: `{packet['binding_status']}`",
        f"- overall_pass: `{packet['overall_pass']}`",
        f"- reference eval run in this packet: `{packet['reference_eval_run_in_this_packet']}`",
        f"- CityFlow run in this packet: `{packet['cityflow_run_in_this_packet']}`",
        f"- model rollout in this packet: `{packet['model_rollout_in_this_packet']}`",
        f"- checkpoint real inference in this packet: `{packet['checkpoint_real_inference_in_this_packet']}`",
        f"- traffic result value reading in this packet: `{packet['traffic_result_value_reading_in_this_packet']}`",
        f"- numeric aggregation in this packet: `{packet['numeric_traffic_aggregation_in_this_packet']}`",
        "",
        "## Binding Contract",
        "",
        f"- PPO policy loader implemented: `{contract['ppo_policy_loader_implemented']}`",
        f"- reference policy factory implemented: `{contract['reference_policy_factory_implemented']}`",
        f"- env factory implemented: `{contract['env_factory_implemented']}`",
        f"- deterministic probe required: `{contract['deterministic_probe_required']}`",
        f"- PPO argmax required: `{contract['ppo_argmax_required']}`",
        f"- PPO default preference locked: `{contract['ppo_default_preference_locked']}`",
        f"- reference policy requires encoder: `{contract['reference_policy_requires_encoder']}`",
        f"- obs feature schema hash enforced: `{contract['obs_feature_schema_hash_enforced']}`",
        f"- common metric debug recorded to metadata: `{contract['common_metric_debug_recorded_to_metadata']}`",
        f"- temperature required: `{contract['temperature_required']}`",
        f"- episodes: `{contract['episodes']}`",
        f"- min action time: `{contract['min_action_time']}`",
        f"- max decision steps per episode: `{contract['max_decision_steps_per_episode']}`",
        f"- common metrics only: `{contract['common_metrics_only']}`",
        "",
        "## Provenance",
        "",
        f"- adapter packet sha256: `{packet['provenance']['adapter_packet_sha256']}`",
        f"- adapter commit: `{packet['provenance']['adapter_commit']}`",
        "",
        "## Failures",
        "",
    ]
    if packet["failures"]:
        lines.extend(f"- {failure}" for failure in packet["failures"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "This packet binds concrete factories to the already guarded evaluator adapter. It does not run CityFlow, execute a model rollout, perform checkpoint inference on real traffic states, read traffic result values, aggregate traffic metrics, rank methods, generate performance tables, or make traffic-improvement or paper-ready claims.",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=DEFAULT_BINDING_DIR)
    parser.add_argument("--adapter_packet", default=DEFAULT_ADAPTER_PACKET)
    parser.add_argument("--adapter_commit", default="e04d7ce")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet = build_reference_eval_evaluator_binding_packet(
        out_dir=args.out_dir,
        adapter_packet_path=args.adapter_packet,
        adapter_commit=args.adapter_commit,
    )
    print(json.dumps({"overall_pass": packet["overall_pass"], "failure_count": len(packet["failures"])}, sort_keys=True))
    if not packet["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
