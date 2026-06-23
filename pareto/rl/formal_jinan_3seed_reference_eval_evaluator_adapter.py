#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.rl.formal_jinan_3seed_artifact_hash_verification import sha256_file
from pareto.rl.formal_jinan_3seed_reference_eval_guard import (
    validate_reference_eval_execution_request,
    validate_reference_eval_guard_packet,
)
from pareto.rl.formal_jinan_3seed_reference_eval_runner import (
    DEFAULT_RUNNER_DIR,
    _manifest_row,
    recheck_checkpoint_sha_binding,
    validate_common_metric_payload,
    validate_reference_eval_runner_packet,
)


DEFAULT_RUNNER_PACKET = f"{DEFAULT_RUNNER_DIR}/formal_jinan_3seed_reference_eval_runner.json"
DEFAULT_ADAPTER_DIR = (
    "docs/pro_reviews/pareto_ppo_formal_jinan_3seed_reference_eval_evaluator_adapter_2026-06-01"
)
ADAPTER_PACKET_TYPE = "formal_jinan_3seed_reference_eval_evaluator_adapter"
FORBIDDEN_ADAPTER_TRUE_FLAGS = (
    "reference_eval_run_in_this_packet",
    "cityflow_run_in_this_packet",
    "model_rollout_in_this_packet",
    "traffic_result_value_reading_in_this_packet",
    "numeric_traffic_aggregation_in_this_packet",
    "method_ranking_in_this_packet",
    "performance_table_in_this_packet",
    "best_method_claim_in_this_packet",
    "traffic_improvement_claim_in_this_packet",
    "paper_result_claim_in_this_packet",
)

PolicyLoader = Callable[..., Any]
ReferencePolicyFactory = Callable[..., Any]
EnvFactory = Callable[..., Any]


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalize_action_payload(action: Any) -> int | list[int]:
    if isinstance(action, bool):
        raise ValueError("deterministic policy action must be an int or list of ints")
    if isinstance(action, int):
        return int(action)
    if isinstance(action, (list, tuple)):
        normalized: list[int] = []
        for item in action:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError("deterministic joint action must contain only ints")
            normalized.append(int(item))
        return normalized
    raise ValueError("deterministic policy action must be an int or list of ints")


def _select_deterministic_action(policy: Any, observation: Any) -> int | list[int]:
    if not hasattr(policy, "select_action"):
        raise ValueError("policy object must expose select_action")
    return _normalize_action_payload(
        policy.select_action(
            observation,
            deterministic=True,
            temperature=0.0,
            stochastic_sampling_allowed=False,
            exploration_noise_allowed=False,
        )
    )


def _run_fixed_budget_rollout(env: Any, policy: Any, request: dict[str, Any]) -> dict[str, float]:
    episodes = int(request["episodes"])
    max_steps = int(request["max_decision_steps_per_episode"])
    min_action_time = int(request["min_action_time"])
    seed_id = int(request["seed_id"])
    for episode in range(episodes):
        observation = env.reset(episode=episode, seed=seed_id)
        for _step in range(max_steps):
            action = _select_deterministic_action(policy, observation)
            observation = env.step(action, min_action_time=min_action_time)
    if not hasattr(env, "common_metrics"):
        raise ValueError("eval environment adapter must expose common_metrics")
    metrics = validate_common_metric_payload(env.common_metrics())
    debug = env.common_metric_debug() if hasattr(env, "common_metric_debug") else {}
    if debug:
        return {"common_metrics": metrics, "common_metric_debug": debug}
    return metrics


def build_guarded_reference_eval_evaluator(
    guard_packet: dict[str, Any],
    *,
    ppo_policy_loader: PolicyLoader | None,
    reference_policy_factory: ReferencePolicyFactory | None,
    env_factory: EnvFactory,
) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, float]]:
    """Build the future evaluator adapter under guard-only constraints.

    This function does not import or run CityFlow by itself. The real evaluator
    must be provided through ``env_factory`` and policy factories after the
    external gate approves execution. Tests use mock factories to prove the
    adapter enforces deterministic policy calls, checkpoint SHA rechecks, and
    fixed budget/seed binding before any future real execution.
    """
    validate_reference_eval_guard_packet(guard_packet)

    def evaluator(request: dict[str, Any], manifest_row: dict[str, Any]) -> dict[str, float]:
        validated = validate_reference_eval_execution_request(request, guard_packet)
        expected_row = _manifest_row(
            guard_packet,
            method=str(validated["method"]),
            seed_id=int(validated["seed_id"]),
        )
        if expected_row.get("run_kind") != manifest_row.get("run_kind"):
            raise ValueError("manifest row run_kind mismatch")
        if expected_row.get("method") != manifest_row.get("method"):
            raise ValueError("manifest row method mismatch")
        if int(expected_row.get("seed", -1)) != int(manifest_row.get("seed", -2)):
            raise ValueError("manifest row seed mismatch")

        run_kind = str(validated["run_kind"])
        if run_kind == "ppo_checkpoint_eval":
            if ppo_policy_loader is None:
                raise ValueError("ppo_policy_loader is required for PPO checkpoint eval")
            # Repeat the SHA check inside the adapter so bypassing the runner's
            # outer check still cannot reach policy loading with drifted files.
            recheck_checkpoint_sha_binding(validated, guard_packet)
            checkpoint_path = Path(str(expected_row["checkpoint_last_path"]))
            policy = ppo_policy_loader(
                checkpoint_path=checkpoint_path,
                request=dict(validated),
                manifest_row=dict(expected_row),
            )
        elif run_kind == "reference_policy_eval":
            if reference_policy_factory is None:
                raise ValueError("reference_policy_factory is required for reference policy eval")
            policy = reference_policy_factory(
                method=str(validated["method"]),
                request=dict(validated),
                manifest_row=dict(expected_row),
            )
        else:
            raise ValueError(f"unsupported run_kind: {run_kind}")

        env = env_factory(
            scenario="jinan",
            traffic_file="anon_3_4_jinan_real.json",
            method=str(validated["method"]),
            seed_id=int(validated["seed_id"]),
            cityflow_seed=int(validated["cityflow_seed"]),
            min_action_time=int(validated["min_action_time"]),
            episodes=int(validated["episodes"]),
            max_decision_steps_per_episode=int(validated["max_decision_steps_per_episode"]),
            stochastic_sampling_allowed=False,
            exploration_noise_allowed=False,
            temperature=0.0,
        )
        return _run_fixed_budget_rollout(env, policy, validated)

    return evaluator


def build_reference_eval_evaluator_adapter_packet(
    *,
    out_dir: str | Path = DEFAULT_ADAPTER_DIR,
    runner_packet_path: str | Path = DEFAULT_RUNNER_PACKET,
    runner_commit: str = "c980523",
) -> dict[str, Any]:
    runner_path = Path(runner_packet_path)
    failures: list[str] = []
    if runner_path.is_file():
        runner_packet = _read_json(runner_path)
        try:
            validate_reference_eval_runner_packet(runner_packet)
        except ValueError as exc:
            failures.append(f"runner packet: {exc}")
        runner_sha = sha256_file(runner_path)
    else:
        runner_packet = {}
        runner_sha = None
        failures.append(f"missing runner packet: {runner_path}")
    provenance = runner_packet.get("provenance") or {}
    packet = {
        "packet_type": ADAPTER_PACKET_TYPE,
        "adapter_status": "PASS" if not failures else "FAIL",
        "overall_pass": not failures,
        "failures": failures,
        "reference_eval_run_in_this_packet": False,
        "cityflow_run_in_this_packet": False,
        "model_rollout_in_this_packet": False,
        "traffic_result_value_reading_in_this_packet": False,
        "numeric_traffic_aggregation_in_this_packet": False,
        "method_ranking_in_this_packet": False,
        "performance_table_in_this_packet": False,
        "best_method_claim_in_this_packet": False,
        "traffic_improvement_claim_in_this_packet": False,
        "paper_result_claim_in_this_packet": False,
        "adapter_entrypoint": "build_guarded_reference_eval_evaluator",
        "guarded_runner_entrypoint": "execute_guarded_reference_eval_request",
        "adapter_contract": {
            "real_cityflow_evaluator_executed_in_this_packet": False,
            "mock_only_tests_required": True,
            "checkpoint_sha_recheck_before_policy_load": True,
            "deterministic_action_required": True,
            "temperature_required": 0.0,
            "stochastic_sampling_allowed": False,
            "exploration_noise_allowed": False,
            "episodes": 5,
            "min_action_time": 30,
            "max_decision_steps_per_episode": 120,
            "cityflow_seed_binding": "seed_id",
            "required_common_metrics": [
                "average_travel_time",
                "throughput",
                "mean_queue_length",
            ],
        },
        "provenance": {
            "runner_packet": str(runner_path),
            "runner_packet_sha256": runner_sha,
            "runner_commit": runner_commit,
            "guard_packet_sha256": provenance.get("guard_packet_sha256"),
            "proposal_packet_sha256": provenance.get("proposal_packet_sha256"),
            "analysis_packet_sha256": provenance.get("analysis_packet_sha256"),
            "upstream_guard_packet_sha256": provenance.get("upstream_guard_packet_sha256"),
            "verification_packet_sha256": provenance.get("verification_packet_sha256"),
            "request_packet_sha256": provenance.get("request_packet_sha256"),
            "execution_audit_packet_sha256": provenance.get("execution_audit_packet_sha256"),
        },
    }
    if packet["overall_pass"]:
        validate_reference_eval_evaluator_adapter_packet(packet)
    output = Path(out_dir)
    _write_json(output / "formal_jinan_3seed_reference_eval_evaluator_adapter.json", packet)
    write_markdown(packet, output / "formal_jinan_3seed_reference_eval_evaluator_adapter.md")
    return packet


def validate_reference_eval_evaluator_adapter_packet(packet: dict[str, Any]) -> None:
    if packet.get("packet_type") != ADAPTER_PACKET_TYPE:
        raise ValueError(f"packet_type must be {ADAPTER_PACKET_TYPE}")
    for key in FORBIDDEN_ADAPTER_TRUE_FLAGS:
        if packet.get(key) is not False:
            raise ValueError(f"{key} must be false")
    contract = packet.get("adapter_contract") or {}
    if contract.get("real_cityflow_evaluator_executed_in_this_packet") is not False:
        raise ValueError("real_cityflow_evaluator_executed_in_this_packet must be false")
    if contract.get("checkpoint_sha_recheck_before_policy_load") is not True:
        raise ValueError("checkpoint_sha_recheck_before_policy_load must be true")
    if contract.get("deterministic_action_required") is not True:
        raise ValueError("deterministic_action_required must be true")
    if float(contract.get("temperature_required", -1.0)) != 0.0:
        raise ValueError("temperature_required must be 0.0")
    if int(contract.get("episodes", -1)) != 5:
        raise ValueError("episodes must be 5")
    if int(contract.get("max_decision_steps_per_episode", -1)) != 120:
        raise ValueError("max_decision_steps_per_episode must be 120")
    if packet.get("overall_pass") is not (not packet.get("failures")):
        raise ValueError("overall_pass must match failures")


def write_markdown(packet: dict[str, Any], path: str | Path) -> None:
    contract = packet["adapter_contract"]
    lines = [
        "# Formal Jinan 3-Seed Reference-Eval Evaluator Adapter",
        "",
        f"- adapter_status: `{packet['adapter_status']}`",
        f"- overall_pass: `{packet['overall_pass']}`",
        f"- reference eval run in this packet: `{packet['reference_eval_run_in_this_packet']}`",
        f"- CityFlow run in this packet: `{packet['cityflow_run_in_this_packet']}`",
        f"- model rollout in this packet: `{packet['model_rollout_in_this_packet']}`",
        f"- traffic result value reading in this packet: `{packet['traffic_result_value_reading_in_this_packet']}`",
        f"- numeric aggregation in this packet: `{packet['numeric_traffic_aggregation_in_this_packet']}`",
        f"- adapter entrypoint: `{packet['adapter_entrypoint']}`",
        "",
        "## Adapter Contract",
        "",
        f"- real CityFlow evaluator executed in this packet: `{contract['real_cityflow_evaluator_executed_in_this_packet']}`",
        f"- checkpoint SHA recheck before policy load: `{contract['checkpoint_sha_recheck_before_policy_load']}`",
        f"- deterministic action required: `{contract['deterministic_action_required']}`",
        f"- temperature required: `{contract['temperature_required']}`",
        f"- episodes: `{contract['episodes']}`",
        f"- min action time: `{contract['min_action_time']}`",
        f"- max decision steps per episode: `{contract['max_decision_steps_per_episode']}`",
        "",
        "## Provenance",
        "",
        f"- runner packet sha256: `{packet['provenance']['runner_packet_sha256']}`",
        f"- guard packet sha256: `{packet['provenance']['guard_packet_sha256']}`",
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
            "This packet builds the evaluator adapter contract with mock-only tests. It does not run CityFlow, load checkpoints for real inference, read traffic metric values, aggregate traffic values, rank methods, generate performance tables, or make traffic-improvement or paper-ready claims.",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--runner_packet", default=DEFAULT_RUNNER_PACKET)
    parser.add_argument("--runner_commit", default="c980523")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet = build_reference_eval_evaluator_adapter_packet(
        out_dir=args.out_dir,
        runner_packet_path=args.runner_packet,
        runner_commit=args.runner_commit,
    )
    print(json.dumps({"overall_pass": packet["overall_pass"], "failure_count": len(packet["failures"])}, sort_keys=True))
    if not packet["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
