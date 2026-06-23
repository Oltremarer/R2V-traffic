from __future__ import annotations

from pathlib import Path
from typing import Any

from pareto.rl.paper_learned_artifact_inventory import learned_artifact_blockers
from pareto.rl.paper_final_reference_runner import REFERENCE_BASELINE_SCRIPTS

LEARNED_PPO_METHOD_IDS = {
    "Cond-Scalar-RL": "film_scalar_potential",
    "Weighted-RL": "weighted_proxy",
    "VectorQ-PPO": "vector_quality_potential",
}
LEARNED_ARTIFACT_BASELINES = {
    "Cond-Scalar-RL": "Cond-Scalar-RL",
    "VectorQ-PPO": "VectorQ-PPO",
}
PAPER_FINAL_CITY_SPECS = {
    "jinan": "configs/formal/paper_final_jinan_5seed_ppo.json",
    "hangzhou": "configs/formal/paper_final_hangzhou_5seed_ppo.json",
    "newyork_28x7": "configs/formal/paper_final_newyork_28x7_5seed_ppo.json",
}
PAPER_FINAL_APPROVAL_ENV = "PPTS_PARETO_PPO_FINAL_EXECUTION_APPROVAL_PHRASE"
PAPER_FINAL_LEARNED_EPISODES_PER_SEED = 278
PAPER_FINAL_LEARNED_DECISION_STEPS_PER_EPISODE = 120
ALLOWED_EXECUTABLE_RUNNER_STATUSES = {
    "executable_preview",
    "excluded_by_scope_limitation",
    "missing_blocker",
}


def _dataset_for_city(city: str) -> str:
    if city in {"jinan", "hangzhou", "newyork_28x7"}:
        return city
    raise ValueError(f"unknown paper-final city for executable runner: {city}")


def _artifact_rows_by_key(learned_artifact_inventory: dict[str, Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
    if learned_artifact_inventory is None:
        return {}
    if learned_artifact_inventory.get("coverage_status") != "complete" or learned_artifact_blockers(learned_artifact_inventory):
        return {}
    return {
        (str(row.get("baseline")), str(row.get("city"))): dict(row)
        for row in learned_artifact_inventory.get("rows") or []
        if isinstance(row, dict)
    }


def _normalizer_artifact_for_city(
    artifacts: dict[tuple[str, str], dict[str, Any]],
    city: str,
) -> dict[str, Any] | None:
    for baseline in ("Cond-Scalar-RL", "VectorQ-PPO"):
        row = artifacts.get((baseline, city))
        if row and row.get("objective_normalizer_path") and row.get("objective_normalizer_hash"):
            return row
    return None


def _reference_row(row: dict[str, Any], *, python_bin: str) -> dict[str, Any]:
    method = str(row.get("method") or "")
    script = REFERENCE_BASELINE_SCRIPTS[method]
    city = str(row.get("city") or "")
    traffic_file = str(row.get("traffic_file") or "")
    seed = int(row.get("seed", 0))
    out_dir = str(row.get("out_dir") or "")
    argv = [
        python_bin,
        "pareto/rl/paper_final_reference_runner.py",
        "--method",
        method,
        "--dataset",
        _dataset_for_city(city),
        "--traffic_file",
        traffic_file,
        "--seed_id",
        str(seed),
        "--out_dir",
        out_dir,
        "--legacy_script",
        script,
    ]
    return {
        "method": method,
        "city": city,
        "traffic_file": traffic_file,
        "seed": seed,
        "preference_template": row.get("preference_template"),
        "status": "executable_preview",
        "runner_family": "paper_final_reference_adapter",
        "legacy_script": script,
        "command_argv": argv,
        "out_dir": out_dir,
        "executes_now": False,
        "reads_result_values": False,
        "generates_ranking": False,
        "paper_table_generated": False,
        "paper_result_text_generated": False,
    }


def _c2t_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("scope_limitation") != "excluded_by_reviewer" or row.get("command") is not None:
        return {
            "method": "C2T-scalar",
            "city": row.get("city"),
            "seed": row.get("seed"),
            "status": "missing_blocker",
            "blocker": "C2T-scalar must be an approved exclusion with no executable command",
            "command_argv": [],
            "executes_now": False,
        }
    return {
        "method": "C2T-scalar",
        "city": row.get("city"),
        "traffic_file": row.get("traffic_file"),
        "seed": row.get("seed"),
        "preference_template": row.get("preference_template"),
        "status": "excluded_by_scope_limitation",
        "scope_limitation": "excluded_by_reviewer",
        "paper_claim_limitation": row.get("paper_claim_limitation"),
        "command_argv": [],
        "out_dir": row.get("out_dir"),
        "executes_now": False,
        "reads_result_values": False,
        "generates_ranking": False,
        "paper_table_generated": False,
        "paper_result_text_generated": False,
    }


def _missing_learned_row(row: dict[str, Any], blocker: str) -> dict[str, Any]:
    method = str(row.get("method") or "")
    return {
        "method": method,
        "method_id": LEARNED_PPO_METHOD_IDS[method],
        "city": row.get("city"),
        "traffic_file": row.get("traffic_file"),
        "seed": row.get("seed"),
        "preference_template": row.get("preference_template"),
        "status": "missing_blocker",
        "blocker": blocker,
        "command_argv": [],
        "out_dir": row.get("out_dir"),
        "executes_now": False,
        "reads_result_values": False,
        "generates_ranking": False,
        "paper_table_generated": False,
        "paper_result_text_generated": False,
    }


def _paper_final_spec_for_city(city: str) -> str:
    try:
        return PAPER_FINAL_CITY_SPECS[city]
    except KeyError as exc:
        raise ValueError(f"unknown paper-final city for learned runner: {city}") from exc


def _learned_row(
    row: dict[str, Any],
    *,
    python_bin: str,
    artifacts: dict[tuple[str, str], dict[str, Any]],
    device: str,
) -> dict[str, Any]:
    method = str(row.get("method") or "")
    method_id = LEARNED_PPO_METHOD_IDS[method]
    city = str(row.get("city") or "")
    seed = int(row.get("seed", 0))
    preference_template = str(row.get("preference_template") or "balanced")
    out_dir = str(row.get("out_dir") or "")
    normalizer_row = _normalizer_artifact_for_city(artifacts, city)
    if normalizer_row is None:
        return _missing_learned_row(row, "paper-final learned PPO objective normalizer artifact missing")

    argv = [
        python_bin,
        "pareto/rl/formal_pilot_runner.py",
        "--spec",
        _paper_final_spec_for_city(city),
        "--method",
        method_id,
        "--paper_final_execution",
        "--approval_phrase",
        f"${{{PAPER_FINAL_APPROVAL_ENV}}}",
        "--seed_id",
        str(seed),
        "--rollout_steps",
        "3600",
        "--total_env_steps_per_seed",
        "1000000",
        "--fixed_preference_template",
        preference_template,
        "--episodes",
        str(PAPER_FINAL_LEARNED_EPISODES_PER_SEED),
        "--max_decision_steps_per_episode",
        str(PAPER_FINAL_LEARNED_DECISION_STEPS_PER_EPISODE),
        "--objective_normalizer",
        str(normalizer_row["objective_normalizer_path"]),
        "--objective_normalizer_hash",
        str(normalizer_row["objective_normalizer_hash"]),
        "--out_dir",
        out_dir,
        "--device",
        device,
    ]
    artifact_row: dict[str, Any] | None = None
    if method in LEARNED_ARTIFACT_BASELINES:
        artifact_row = artifacts.get((LEARNED_ARTIFACT_BASELINES[method], city))
        if artifact_row is None:
            return _missing_learned_row(row, f"paper-final {method} model artifact missing")
        model_path = Path(str(artifact_row["model_path"]))
        argv.extend(
            [
                "--film_model_dir" if method == "Cond-Scalar-RL" else "--vector_model_dir",
                model_path.parent.as_posix(),
                "--film_model_hash" if method == "Cond-Scalar-RL" else "--vector_model_hash",
                str(artifact_row["model_hash"]),
            ]
        )

    output = {
        "method": method,
        "method_id": method_id,
        "city": city,
        "traffic_file": row.get("traffic_file"),
        "seed": seed,
        "preference_template": preference_template,
        "status": "executable_preview",
        "runner_family": "formal_pilot_paper_final",
        "spec_path": _paper_final_spec_for_city(city),
        "command_argv": argv,
        "approval_phrase_env": PAPER_FINAL_APPROVAL_ENV,
        "out_dir": out_dir,
        "objective_normalizer_path": normalizer_row["objective_normalizer_path"],
        "objective_normalizer_hash": normalizer_row["objective_normalizer_hash"],
        "objective_normalizer_file_sha256": normalizer_row.get("objective_normalizer_file_sha256"),
        "executes_now": False,
        "reads_result_values": False,
        "generates_ranking": False,
        "paper_table_generated": False,
        "paper_result_text_generated": False,
    }
    if artifact_row is not None:
        output["model_path"] = artifact_row["model_path"]
        output["model_hash"] = artifact_row["model_hash"]
    if method == "Weighted-RL":
        output["weighted_mapping_approved"] = bool(row.get("weighted_mapping_approved"))
        output["paper_claim_limitation"] = row.get("paper_claim_limitation")
    return output


def _runner_row(
    row: dict[str, Any],
    *,
    python_bin: str,
    artifacts: dict[tuple[str, str], dict[str, Any]],
    device: str,
) -> dict[str, Any]:
    if row.get("executes_training_now") is not False:
        raise ValueError("paper-final executable runner input rows must be non-executing previews")
    method = str(row.get("method") or "")
    if method in REFERENCE_BASELINE_SCRIPTS:
        return _reference_row(row, python_bin=python_bin)
    if method == "C2T-scalar":
        return _c2t_row(row)
    if method in LEARNED_PPO_METHOD_IDS:
        return _learned_row(row, python_bin=python_bin, artifacts=artifacts, device=device)
    raise ValueError(f"unknown paper-final runner method: {method}")


def validate_executable_runner_specs(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("packet_type") != "paper_final_executable_runner_specs":
        raise ValueError("executable runner specs packet_type mismatch")
    if spec.get("executes_now") is not False:
        raise ValueError("executable runner specs must be non-executing")
    rows = spec.get("rows")
    if not isinstance(rows, list):
        raise ValueError("executable runner specs rows must be a list")
    for row in rows:
        status = row.get("status")
        if status not in ALLOWED_EXECUTABLE_RUNNER_STATUSES:
            raise ValueError(f"unknown executable runner row status: {status}")
        if row.get("executes_now") is not False:
            raise ValueError("executable runner row must be non-executing")
        if status == "executable_preview" and not row.get("command_argv"):
            raise ValueError("executable preview row missing command_argv")
        if status == "executable_preview":
            out_dir = str(row.get("out_dir") or "")
            if not out_dir.startswith("records/paper_final/"):
                raise ValueError("executable preview out_dir must be under records/paper_final")
        if status == "executable_preview" and row.get("runner_family") == "paper_final_reference_adapter":
            argv = [str(item) for item in row.get("command_argv") or []]
            for flag in (
                "--method",
                "--dataset",
                "--traffic_file",
                "--seed_id",
                "--out_dir",
                "--legacy_script",
            ):
                if flag not in argv:
                    raise ValueError(f"reference executable preview missing {flag}")
        if status == "executable_preview" and row.get("runner_family") == "formal_pilot_paper_final":
            argv = [str(item) for item in row.get("command_argv") or []]
            for flag in (
                "--spec",
                "--method",
                "--paper_final_execution",
                "--approval_phrase",
                "--seed_id",
                "--rollout_steps",
                "--total_env_steps_per_seed",
                "--fixed_preference_template",
                "--objective_normalizer",
                "--objective_normalizer_hash",
                "--out_dir",
                "--episodes",
                "--max_decision_steps_per_episode",
            ):
                if flag not in argv:
                    raise ValueError(f"learned PPO executable preview missing {flag}")
        if status == "excluded_by_scope_limitation" and row.get("command_argv") != []:
            raise ValueError("excluded rows must not have executable command args")
        if status == "missing_blocker" and not row.get("blocker"):
            raise ValueError("missing executable runner row must include blocker")
    expected_status = "missing_blocker" if any(row["status"] == "missing_blocker" for row in rows) else "ready_request"
    if spec.get("status") != expected_status:
        raise ValueError(f"executable runner spec status must be {expected_status}")
    return dict(spec)


def build_executable_runner_specs(
    preview_rows: list[dict[str, Any]],
    *,
    python_bin: str = "python",
    learned_artifact_inventory: dict[str, Any] | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    artifacts = _artifact_rows_by_key(learned_artifact_inventory)
    rows = [_runner_row(row, python_bin=python_bin, artifacts=artifacts, device=device) for row in preview_rows]
    return validate_executable_runner_specs(
        {
            "packet_type": "paper_final_executable_runner_specs",
            "status": "missing_blocker" if any(row["status"] == "missing_blocker" for row in rows) else "ready_request",
            "rows": rows,
            "executes_now": False,
            "reads_result_values": False,
            "generates_ranking": False,
            "paper_table_generated": False,
            "paper_result_text_generated": False,
        }
    )
