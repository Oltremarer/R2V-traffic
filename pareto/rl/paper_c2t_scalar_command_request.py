from __future__ import annotations

from typing import Any

from pareto.rl.paper_c2t_scalar_guard import REQUIRED_C2T_SCALAR_EXCLUSION_APPROVAL
from pareto.rl.paper_final_experiment_manifest import PAPER_FINAL_SEEDS, REQUIRED_CITY_TRAFFIC


C2T_SCALAR_COMMAND_FAMILY = "c2t_scalar_reward_training"
FORBIDDEN_C2T_MAPPINGS = {"env_reward"}
PAPER_SCALE_PPO_BUDGET = {
    "rollout_steps": 3600,
    "minibatch_size": 2048,
    "update_epochs": 10,
    "total_env_steps_per_seed": 1_000_000,
}
ALLOWED_C2T_REQUEST_STATUSES = {"request_only", "excluded_by_reviewer", "missing_blocker"}


def build_c2t_scalar_command_request(
    *,
    command_family: str | None = None,
    exclusion_approval_phrase: str | None = None,
    paper_claim_limitation: str | None = None,
    output_root: str = "records/paper_final/train_20260602_v1",
) -> dict[str, Any]:
    if command_family in FORBIDDEN_C2T_MAPPINGS:
        raise ValueError("env_reward is not C2T-scalar")
    if exclusion_approval_phrase == REQUIRED_C2T_SCALAR_EXCLUSION_APPROVAL:
        return validate_c2t_scalar_command_request(
            {
                "baseline": "C2T-scalar",
                "status": "excluded_by_reviewer",
                "approval_phrase": exclusion_approval_phrase,
                "paper_claim_limitation": paper_claim_limitation,
                "executes_training_now": False,
            }
        )
    if not command_family:
        return validate_c2t_scalar_command_request(
            {
                "baseline": "C2T-scalar",
                "status": "missing_blocker",
                "blocker": "concrete scalar reward-learning command missing",
                "executes_training_now": False,
            }
        )
    rows = []
    for city, traffic_file in REQUIRED_CITY_TRAFFIC.items():
        for seed in PAPER_FINAL_SEEDS:
            row_output_root = f"{output_root}/{city}/seed{int(seed)}/C2T-scalar"
            rows.append(
                {
                    "city": city,
                    "traffic_file": traffic_file,
                    "seed": int(seed),
                    "command_family": command_family,
                    "output_root": row_output_root,
                    "ppo_budget": dict(PAPER_SCALE_PPO_BUDGET),
                    "executes_training_now": False,
                    "command_preview": (
                        f"{command_family} --city {city} --traffic_file {traffic_file} "
                        f"--seed_id {int(seed)} --out_dir {row_output_root}"
                    ),
                }
            )
    return validate_c2t_scalar_command_request(
        {
            "baseline": "C2T-scalar",
            "status": "request_only",
            "command_family": command_family,
            "rows": rows,
            "executes_training_now": False,
            "artifact_hash_claimed": False,
        }
    )


def validate_c2t_scalar_command_request(request: dict[str, Any]) -> dict[str, Any]:
    status = request.get("status")
    if request.get("baseline") not in {None, "C2T-scalar"}:
        raise ValueError("C2T command request must target C2T-scalar")
    if status not in ALLOWED_C2T_REQUEST_STATUSES:
        raise ValueError(f"unknown C2T command request status: {status}")
    if request.get("executes_training_now") is not False:
        raise ValueError("C2T command request must be non-executing")
    if status == "missing_blocker":
        if not request.get("blocker"):
            raise ValueError("missing C2T command request must include blocker")
        return dict(request)
    if status == "excluded_by_reviewer":
        if request.get("approval_phrase") != REQUIRED_C2T_SCALAR_EXCLUSION_APPROVAL:
            raise ValueError("C2T exclusion requires exact reviewer phrase")
        if not request.get("paper_claim_limitation"):
            raise ValueError("C2T exclusion requires paper claim limitation")
        return dict(request)

    command_family = str(request.get("command_family") or "")
    if command_family in FORBIDDEN_C2T_MAPPINGS:
        raise ValueError("env_reward is not C2T-scalar")
    if not command_family:
        raise ValueError("C2T command request missing command_family")
    rows = request.get("rows") or []
    expected = {(city, seed) for city in REQUIRED_CITY_TRAFFIC for seed in PAPER_FINAL_SEEDS}
    observed = {(row.get("city"), int(row.get("seed"))) for row in rows}
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"C2T command request missing city/seed rows: {missing}")
    for row in rows:
        if row.get("executes_training_now") is not False:
            raise ValueError("C2T command row must be non-executing")
        if row.get("command_family") in FORBIDDEN_C2T_MAPPINGS:
            raise ValueError("env_reward is not C2T-scalar")
        if not str(row.get("output_root") or "").startswith("records/paper_final/"):
            raise ValueError("C2T command output_root must be under records/paper_final")
        budget = row.get("ppo_budget") or {}
        for key, value in PAPER_SCALE_PPO_BUDGET.items():
            if int(budget.get(key, -1)) != value:
                raise ValueError(f"C2T command row requires {key}={value}")
    return dict(request)


def c2t_scalar_command_blockers(request: dict[str, Any]) -> list[str]:
    validated = validate_c2t_scalar_command_request(request)
    if validated["status"] == "missing_blocker":
        return [f"C2T-scalar: {validated['blocker']}"]
    return []
