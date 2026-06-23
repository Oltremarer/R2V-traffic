#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.common.scenario import build_llmlight_env_config
from pareto.data.snapshot import capture_snapshot
from pareto.rl.formal_ppo_config import load_formal_ppo_dryrun_config
from pareto.rl.paper_final_experiment_manifest import (
    PAPER_FINAL_SEEDS,
    REQUIRED_CITY_TRAFFIC,
    REQUIRED_PREFERENCE_TEMPLATES,
)
from pareto.rl.paper_final_reference_runner import REFERENCE_METRIC_KEYS
from pareto.rl.paper_final_wandb_uploader import upload_paper_final_learned_eval_to_wandb
from pareto.rl.state_encoder import ParetoStateEncoder


LEARNED_PPO_METHOD_IDS = {
    "Cond-Scalar-RL": "film_scalar_potential",
    "Weighted-RL": "weighted_proxy",
    "VectorQ-PPO": "vector_quality_potential",
}
LEARNED_METHOD_DISPLAY_NAMES = {value: key for key, value in LEARNED_PPO_METHOD_IDS.items()}
PAPER_FINAL_LEARNED_EVAL_ROOT = "records/paper_final/eval_20260602_v1"
PAPER_FINAL_LEARNED_TRAIN_DONE = "PAPER_FINAL_SCOPE_LIMITED_RUN_DONE"
PAPER_FINAL_LEARNED_EVAL_DONE = "PAPER_FINAL_LEARNED_EVAL_DONE"
PAPER_FINAL_EVAL_RUN_COUNTS = 3600
PAPER_FINAL_EVAL_MIN_ACTION_TIME = 30
PAPER_FINAL_LEARNED_EVAL_FILES = (
    "paper_final_learned_eval_metadata.json",
    "paper_final_learned_eval_status.json",
    "paper_final_learned_eval_metrics.json",
    "command.txt",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prepare_llmlight_work_dir(
    dic_agent_conf: dict[str, Any],
    dic_traffic_env_conf: dict[str, Any],
    dic_path: dict[str, str],
) -> None:
    work_dir = Path(dic_path["PATH_TO_WORK_DIRECTORY"])
    model_dir = Path(dic_path["PATH_TO_MODEL"])
    work_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "agent.conf").write_text(json.dumps(dic_agent_conf, indent=2, sort_keys=True), encoding="utf-8")
    (work_dir / "traffic_env.conf").write_text(
        json.dumps(dic_traffic_env_conf, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    data_dir = Path(dic_path["PATH_TO_DATA"])
    shutil.copy(data_dir / dic_traffic_env_conf["TRAFFIC_FILE"], work_dir / dic_traffic_env_conf["TRAFFIC_FILE"])
    shutil.copy(data_dir / dic_traffic_env_conf["ROADNET_FILE"], work_dir / dic_traffic_env_conf["ROADNET_FILE"])


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _normalize_method(method: str) -> tuple[str, str]:
    if method in LEARNED_PPO_METHOD_IDS:
        return method, LEARNED_PPO_METHOD_IDS[method]
    if method in LEARNED_METHOD_DISPLAY_NAMES:
        return LEARNED_METHOD_DISPLAY_NAMES[method], method
    raise ValueError(f"unknown paper-final learned method: {method}")


def _require_paper_final_relative_path(path: str | Path, *, label: str) -> str:
    normalized = str(path).strip()
    if not normalized.startswith("records/paper_final/"):
        raise ValueError(f"{label} must be under records/paper_final")
    if ".." in Path(normalized).parts:
        raise ValueError(f"{label} must not contain '..'")
    return normalized


def default_learned_eval_out_dir(
    *,
    eval_root: str | Path = PAPER_FINAL_LEARNED_EVAL_ROOT,
    city: str,
    traffic_file: str,
    method: str,
    seed_id: int,
    fixed_preference_template: str,
) -> str:
    display_method, _method_id = _normalize_method(method)
    traffic_stem = Path(traffic_file).stem
    return (
        Path(str(eval_root))
        / city
        / traffic_stem
        / display_method
        / f"seed{int(seed_id)}"
        / fixed_preference_template
    ).as_posix()


def validate_learned_eval_request(
    *,
    spec: str,
    method: str,
    seed_id: int,
    train_dir: str,
    eval_out_dir: str | None,
    fixed_preference_template: str,
    checkpoint: str | None = None,
    run_counts: int = PAPER_FINAL_EVAL_RUN_COUNTS,
    min_action_time: int = PAPER_FINAL_EVAL_MIN_ACTION_TIME,
    execute: bool = False,
    require_train_done: bool = True,
) -> dict[str, Any]:
    config = load_formal_ppo_dryrun_config(spec)
    display_method, method_id = _normalize_method(method)
    city = str(config.pilot.get("scenario"))
    traffic_file = str(config.pilot.get("traffic_file"))
    if city not in REQUIRED_CITY_TRAFFIC:
        raise ValueError(f"paper-final learned eval scenario is not approved: {city}")
    if traffic_file != REQUIRED_CITY_TRAFFIC[city]:
        raise ValueError(f"paper-final learned eval traffic_file mismatch for {city}")
    if int(seed_id) not in PAPER_FINAL_SEEDS:
        raise ValueError(f"seed_id {seed_id} is not approved for paper-final learned eval")
    if fixed_preference_template not in REQUIRED_PREFERENCE_TEMPLATES:
        raise ValueError(f"unknown paper-final learned eval preference: {fixed_preference_template}")
    if int(run_counts) != PAPER_FINAL_EVAL_RUN_COUNTS:
        raise ValueError("paper-final learned eval must use RUN_COUNTS=3600 to match reference baselines")
    if int(min_action_time) != PAPER_FINAL_EVAL_MIN_ACTION_TIME:
        raise ValueError("paper-final learned eval must use MIN_ACTION_TIME=30 to match reference baselines")

    train_dir_rel = _require_paper_final_relative_path(train_dir, label="train_dir")
    eval_dir = eval_out_dir or default_learned_eval_out_dir(
        city=city,
        traffic_file=traffic_file,
        method=display_method,
        seed_id=seed_id,
        fixed_preference_template=fixed_preference_template,
    )
    eval_dir_rel = _require_paper_final_relative_path(eval_dir, label="eval_out_dir")
    checkpoint_rel = checkpoint or (Path(train_dir_rel) / "checkpoint_last.pt").as_posix()
    _require_paper_final_relative_path(checkpoint_rel, label="checkpoint")

    if execute:
        checkpoint_path = ROOT / checkpoint_rel
        if not checkpoint_path.is_file():
            raise ValueError(f"paper-final learned eval checkpoint missing: {checkpoint_rel}")
        status_path = ROOT / train_dir_rel / "status.json"
        if require_train_done:
            if not status_path.is_file():
                raise ValueError(f"paper-final learned eval requires completed train status: {status_path}")
            status = _read_json(status_path)
            if status.get("status") != PAPER_FINAL_LEARNED_TRAIN_DONE:
                raise ValueError(f"paper-final learned train status is not complete: {status.get('status')}")

    return {
        "spec": spec,
        "method": display_method,
        "method_id": method_id,
        "city": city,
        "traffic_file": traffic_file,
        "seed_id": int(seed_id),
        "train_dir": train_dir_rel,
        "eval_out_dir": eval_dir_rel,
        "checkpoint": checkpoint_rel,
        "fixed_preference_template": fixed_preference_template,
        "fixed_preference_weights": list(REQUIRED_PREFERENCE_TEMPLATES[fixed_preference_template]),
        "run_counts": int(run_counts),
        "min_action_time": int(min_action_time),
        "execute": bool(execute),
        "require_train_done": bool(require_train_done),
    }


def build_learned_eval_command(request: dict[str, Any], *, python_bin: str = sys.executable) -> list[str]:
    command = [
        python_bin,
        "pareto/rl/paper_final_learned_eval_runner.py",
        "--spec",
        str(request["spec"]),
        "--method",
        str(request["method"]),
        "--seed_id",
        str(request["seed_id"]),
        "--train_dir",
        str(request["train_dir"]),
        "--eval_out_dir",
        str(request["eval_out_dir"]),
        "--checkpoint",
        str(request["checkpoint"]),
        "--fixed_preference_template",
        str(request["fixed_preference_template"]),
        "--run_counts",
        str(request["run_counts"]),
        "--min_action_time",
        str(request["min_action_time"]),
        "--execute",
    ]
    if request.get("row_index") is not None:
        command.extend(["--row_index", str(request["row_index"])])
    return command


def prepare_eval_output_dir(eval_out_dir: str | Path, *, overwrite: bool = False) -> None:
    out = ROOT / str(eval_out_dir)
    status_path = out / "paper_final_learned_eval_status.json"
    if status_path.is_file():
        status = _read_json(status_path)
        if status.get("status") == PAPER_FINAL_LEARNED_EVAL_DONE and not overwrite:
            raise FileExistsError(f"paper-final learned eval is already complete: {eval_out_dir}")
    if out.exists():
        if not overwrite:
            unexpected = sorted(path.name for path in out.iterdir())
            raise FileExistsError(f"paper-final learned eval output dir already exists: {eval_out_dir}: {unexpected}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def _as_float(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {label}: {value}")
    return result


def _as_float_allow_nan(value: Any, *, label: str) -> float:
    result = float(value)
    if math.isinf(result):
        raise ValueError(f"infinite {label}: {value}")
    return result


def _travel_time_mean(env: Any, *, run_counts: int) -> float:
    vehicle_travel_times: dict[str, list[float]] = {}
    for inter in getattr(env, "list_intersection", []):
        arrive_leave = getattr(inter, "dic_vehicle_arrive_leave_time", {})
        for vehicle_id, row in arrive_leave.items():
            if "shadow" in str(vehicle_id):
                continue
            enter_time = _as_float_allow_nan(row.get("enter_time"), label="enter_time")
            if math.isnan(enter_time):
                continue
            leave_raw = _as_float_allow_nan(row.get("leave_time"), label="leave_time")
            leave_time = float(run_counts) if math.isnan(leave_raw) else leave_raw
            duration = leave_time - enter_time
            if duration < 0.0 or not math.isfinite(duration):
                raise ValueError(f"invalid travel duration for {vehicle_id}: {duration}")
            vehicle_travel_times.setdefault(str(vehicle_id), []).append(duration)
    if not vehicle_travel_times:
        raise ValueError("paper-final learned eval produced no vehicle travel-time rows")
    return float(sum(sum(values) for values in vehicle_travel_times.values()) / len(vehicle_travel_times))


def _encode_observations(
    env: Any,
    *,
    encoder: ParetoStateEncoder,
    expected_obs_dim: int,
    expected_feature_hash: str | None,
) -> tuple[Any, str | None]:
    import torch

    rows: list[list[float]] = []
    observed_hash: str | None = None
    for idx in range(len(getattr(env, "list_intersection", []))):
        features, _names, debug = encoder.encode_snapshot(capture_snapshot(env, idx))
        feature_hash = str(debug["feature_names_hash"])
        if expected_feature_hash and feature_hash != expected_feature_hash:
            raise ValueError(
                f"paper-final learned eval feature hash mismatch: expected {expected_feature_hash}, got {feature_hash}"
            )
        observed_hash = feature_hash
        values = [float(value) for value in features.tolist()]
        if len(values) != int(expected_obs_dim):
            raise ValueError(f"paper-final learned eval obs_dim mismatch: expected {expected_obs_dim}, got {len(values)}")
        if any(not math.isfinite(value) for value in values):
            raise ValueError("paper-final learned eval observed non-finite obs features")
        rows.append(values)
    if not rows:
        raise ValueError("paper-final learned eval has no intersections to evaluate")
    return torch.tensor(rows, dtype=torch.float32), observed_hash


def compute_paper_final_learned_eval_metrics(
    *,
    env: Any,
    policy: Any,
    preference_weights: Sequence[float],
    run_counts: int,
    min_action_time: int,
    expected_feature_hash: str | None = None,
    device: str = "cpu",
) -> tuple[dict[str, float], dict[str, Any]]:
    import torch

    encoder = ParetoStateEncoder("hybrid_v1")
    device_obj = torch.device(device)
    policy = policy.to(device_obj)
    policy.eval()
    total_reward = 0.0
    queue_length_episode: list[float] = []
    waiting_time_episode: list[float] = []
    action_counts: Counter[int] = Counter()
    observed_feature_hash: str | None = None

    done = False
    env.reset()
    step_count = int(run_counts / min_action_time)
    w_tensor = torch.tensor([float(value) for value in preference_weights], dtype=torch.float32, device=device_obj)
    for _step in range(step_count):
        if done:
            break
        obs, feature_hash = _encode_observations(
            env,
            encoder=encoder,
            expected_obs_dim=int(policy.obs_dim),
            expected_feature_hash=expected_feature_hash,
        )
        observed_feature_hash = feature_hash
        w_batch = w_tensor.reshape(1, -1).expand(obs.shape[0], -1)
        with torch.no_grad():
            logits, _values = policy.forward(obs.to(device_obj), w_batch)
            actions = torch.argmax(logits, dim=-1).detach().cpu().tolist()
        action_list = [int(action) for action in actions]
        action_counts.update(action_list)
        _next_state, rewards, done, _info = env.step(action_list)
        if isinstance(rewards, (list, tuple)):
            total_reward += float(sum(float(value) for value in rewards))
        else:
            total_reward += float(rewards)
        queue_length_inter = [
            sum(float(value) for value in inter.dic_feature.get("lane_num_waiting_vehicle_in", []))
            for inter in getattr(env, "list_intersection", [])
        ]
        queue_length_episode.append(float(sum(queue_length_inter)))
        waiting_times = [
            float(row["time"])
            for row in getattr(env, "waiting_vehicle_list", {}).values()
            if row.get("time") is not None
        ]
        waiting_time_episode.append(float(sum(waiting_times) / len(waiting_times)) if waiting_times else 0.0)

    if not queue_length_episode:
        raise ValueError("paper-final learned eval produced no decision steps")
    metrics = {
        "test_reward_over": float(total_reward),
        "test_avg_queue_len_over": float(sum(queue_length_episode) / len(queue_length_episode)),
        "test_queuing_vehicle_num_over": float(sum(queue_length_episode)),
        "test_avg_waiting_time_over": float(sum(waiting_time_episode) / len(waiting_time_episode)),
        "test_avg_travel_time_over": _travel_time_mean(env, run_counts=run_counts),
    }
    for key in REFERENCE_METRIC_KEYS:
        _as_float(metrics[key], label=key)
    debug = {
        "decision_steps": len(queue_length_episode),
        "run_counts": int(run_counts),
        "min_action_time": int(min_action_time),
        "action_counts": {str(key): int(value) for key, value in sorted(action_counts.items())},
        "observed_feature_hash": observed_feature_hash,
        "travel_time_semantics": "legacy_test_over_incomplete_leave_time_filled_with_RUN_COUNTS",
        "metric_keys": list(REFERENCE_METRIC_KEYS),
    }
    return metrics, debug


def run_paper_final_learned_eval(
    request: dict[str, Any],
    *,
    overwrite: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    prepare_eval_output_dir(request["eval_out_dir"], overwrite=overwrite)
    out = ROOT / str(request["eval_out_dir"])
    (out / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    config = load_formal_ppo_dryrun_config(request["spec"])
    dic_agent_conf, dic_traffic_env_conf, dic_path = build_llmlight_env_config(
        scenario=request["city"],
        traffic_file=request["traffic_file"],
        seed=int(request["seed_id"]),
        run_counts=int(request["run_counts"]),
        min_action_time=int(request["min_action_time"]),
        model_name="Random",
        work_dir=str(out / "llmlight_work"),
    )
    _prepare_llmlight_work_dir(dic_agent_conf, dic_traffic_env_conf, dic_path)

    from utils.cityflow_env import CityFlowEnv
    from pareto.rl.ppo_actor_critic import load_actor_critic_checkpoint

    env = CityFlowEnv(
        path_to_log=dic_path["PATH_TO_WORK_DIRECTORY"],
        path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path,
    )
    policy, checkpoint_payload = load_actor_critic_checkpoint(ROOT / str(request["checkpoint"]))
    checkpoint_metadata = checkpoint_payload.get("metadata") if isinstance(checkpoint_payload, dict) else {}
    expected_hash = None
    if isinstance(checkpoint_metadata, dict):
        expected_hash = checkpoint_metadata.get("observed_feature_hash") or checkpoint_metadata.get("state_encoder_hash")
    metadata = {
        "paper_final_learned_eval": True,
        "same_eval_settings_as_reference_baselines": True,
        "continues_training": False,
        "checkpoint_inference_only": True,
        "method": request["method"],
        "method_id": request["method_id"],
        "city": request["city"],
        "traffic_file": request["traffic_file"],
        "seed_id": int(request["seed_id"]),
        "train_dir": request["train_dir"],
        "checkpoint": request["checkpoint"],
        "spec": request["spec"],
        "run_counts": int(request["run_counts"]),
        "min_action_time": int(request["min_action_time"]),
        "fixed_preference_template": request["fixed_preference_template"],
        "fixed_preference_weights": list(request["fixed_preference_weights"]),
        "reference_metric_keys": list(REFERENCE_METRIC_KEYS),
        "config_ppo_hash": config.ppo_config_hash(),
        "expected_feature_hash": expected_hash,
        "action_selection": "argmax_deterministic",
        "stochastic_sampling_allowed": False,
        "exploration_noise_allowed": False,
        "temperature": 0.0,
    }
    _write_json(out / "paper_final_learned_eval_metadata.json", metadata)
    try:
        metrics, debug = compute_paper_final_learned_eval_metrics(
            env=env,
            policy=policy,
            preference_weights=request["fixed_preference_weights"],
            run_counts=int(request["run_counts"]),
            min_action_time=int(request["min_action_time"]),
            expected_feature_hash=expected_hash,
            device=device,
        )
        if hasattr(env, "batch_log_2"):
            env.batch_log_2()
        if hasattr(env, "end_cityflow"):
            env.end_cityflow()
    except Exception:
        if hasattr(env, "end_cityflow"):
            env.end_cityflow()
        raise
    status = {
        "status": PAPER_FINAL_LEARNED_EVAL_DONE,
        "metrics_file": "paper_final_learned_eval_metrics.json",
        "metadata_file": "paper_final_learned_eval_metadata.json",
        "same_eval_settings_as_reference_baselines": True,
        "continues_training": False,
        "method": request["method"],
        "city": request["city"],
        "seed_id": int(request["seed_id"]),
        "fixed_preference_template": request["fixed_preference_template"],
        **debug,
    }
    _write_json(out / "paper_final_learned_eval_metrics.json", metrics)
    _write_json(out / "paper_final_learned_eval_status.json", status)
    status["wandb_upload"] = upload_paper_final_learned_eval_to_wandb(
        {
            "row_index": request.get("row_index", 0),
            "method": request["method"],
            "city": request["city"],
            "seed": int(request["seed_id"]),
            "preference_template": request["fixed_preference_template"],
            "eval_out_dir": request["eval_out_dir"],
        }
    )
    _write_json(out / "paper_final_learned_eval_status.json", status)
    return {"status": PAPER_FINAL_LEARNED_EVAL_DONE, "eval_out_dir": request["eval_out_dir"], "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed_id", type=int, required=True)
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--eval_out_dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--fixed_preference_template", required=True)
    parser.add_argument("--run_counts", type=int, default=PAPER_FINAL_EVAL_RUN_COUNTS)
    parser.add_argument("--min_action_time", type=int, default=PAPER_FINAL_EVAL_MIN_ACTION_TIME)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--row_index", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow_incomplete_train_status", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = validate_learned_eval_request(
        spec=args.spec,
        method=args.method,
        seed_id=args.seed_id,
        train_dir=args.train_dir,
        eval_out_dir=args.eval_out_dir,
        checkpoint=args.checkpoint,
        fixed_preference_template=args.fixed_preference_template,
        run_counts=args.run_counts,
        min_action_time=args.min_action_time,
        execute=args.execute,
        require_train_done=not args.allow_incomplete_train_status,
    )
    if args.row_index is not None:
        request["row_index"] = int(args.row_index)
    if not args.execute:
        print(
            json.dumps(
                {
                    "paper_final_learned_eval_preview": True,
                    "executes_now": False,
                    "request": request,
                    "command_argv": build_learned_eval_command(request),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    try:
        payload = run_paper_final_learned_eval(request, overwrite=args.overwrite, device=args.device)
    except Exception as exc:
        out = ROOT / str(request["eval_out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        _write_json(
            out / "paper_final_learned_eval_status.json",
            {
                "status": "PAPER_FINAL_LEARNED_EVAL_FAILED",
                "error": str(exc),
                "method": request["method"],
                "city": request["city"],
                "seed_id": int(request["seed_id"]),
                "fixed_preference_template": request["fixed_preference_template"],
            },
        )
        raise
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
