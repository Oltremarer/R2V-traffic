from __future__ import annotations

from typing import Any

from pareto.rl.paper_final_experiment_manifest import PAPER_FINAL_SEEDS, REQUIRED_CITY_TRAFFIC


PAPER_FINAL_DATA_POLICIES = (
    "random",
    "fixedtime",
    "maxpressure",
    "advanced_maxpressure",
)
PAPER_FINAL_SPLITS = ("train", "val", "test")
PAPER_FINAL_DATA_DEFAULTS = {
    "episodes_per_policy_seed": 1,
    "max_steps": 3600,
    "encoder_id": "hybrid_v1",
    "split_seed": 20260602,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "record_group_key": "time_block",
    "time_block_size": 300,
    "normalizer_clip": 5.0,
    "num_objective_pairs": 1000,
    "num_preference_pairs": 1000,
    "num_dominance_pairs": 300,
    "num_reversal_pairs": 200,
    "tie_margin": 0.05,
    "min_efficiency_stability_conflict": 40,
    "reversal_template_quota": {
        "efficiency__fairness": 20,
        "efficiency__stability": 20,
        "safety__fairness": 20,
        "safety__stability": 20,
    },
    "strict_min_objective_per_head": 100,
    "strict_min_preference_pairs": 500,
    "strict_min_dominance_pairs": 100,
    "strict_min_reversal_pairs": 100,
    "positive_ratio_by_objective_low": 0.2,
    "positive_ratio_by_objective_high": 0.8,
}
ALLOWED_DATA_GENERATION_REQUEST_STATUSES = {"ready_request", "missing_blocker"}


def _buffer_root(city: str, policy: str, seed: int, run_id: str) -> str:
    return f"records/paper_final_data_buffers/{run_id}/{city}/{policy}/seed{seed}"


def _buffer_jsonl(city: str, policy: str, seed: int, run_id: str) -> str:
    return f"{_buffer_root(city, policy, seed, run_id)}/records_raw.jsonl"


def _transition_jsonl(city: str, policy: str, seed: int, run_id: str) -> str:
    return f"{_buffer_root(city, policy, seed, run_id)}/transitions_raw.jsonl"


def _collect_command(city: str, traffic_file: str, policy: str, seed: int, run_id: str, params: dict[str, Any]) -> str:
    return (
        "python pareto/data/collect_pareto_buffer.py "
        f"--scenario {city} --traffic_file {traffic_file} --policy {policy} --seed {seed} "
        f"--episodes {params['episodes_per_policy_seed']} --max_steps {params['max_steps']} "
        f"--encoder_id {params['encoder_id']} --out {_buffer_jsonl(city, policy, seed, run_id)} "
        f"--transitions_out {_transition_jsonl(city, policy, seed, run_id)} "
        f"--work_dir {_buffer_root(city, policy, seed, run_id)}/work"
    )


def _split_command(city: str, run_id: str, params: dict[str, Any]) -> str:
    inputs = " ".join(
        _buffer_jsonl(city, policy, seed, run_id)
        for policy in PAPER_FINAL_DATA_POLICIES
        for seed in PAPER_FINAL_SEEDS
    )
    return (
        "python pareto/data/split_records.py "
        f"--inputs {inputs} --out_dir data/pareto_records_split/{city}/paper_final "
        f"--seed {params['split_seed']} --train_ratio {params['train_ratio']} --val_ratio {params['val_ratio']} "
        f"--group_key {params['record_group_key']} --time_block_size {params['time_block_size']}"
    )


def _normalizer_path(city: str) -> str:
    return f"data/normalizers/{city}/objective_norm_paper_final.json"


def _fit_normalizer_command(city: str, params: dict[str, Any]) -> str:
    return (
        "python pareto/data/fit_objective_normalizer.py "
        f"--buffers data/pareto_records_split/{city}/paper_final/train_raw.jsonl "
        f"--out {_normalizer_path(city)} --clip {params['normalizer_clip']}"
    )


def _apply_normalizer_command(city: str) -> str:
    return (
        "python pareto/data/apply_objective_normalizer.py "
        f"--normalizer {_normalizer_path(city)} "
        f"--inputs data/pareto_records_split/{city}/paper_final/train_raw.jsonl "
        f"data/pareto_records_split/{city}/paper_final/val_raw.jsonl "
        f"data/pareto_records_split/{city}/paper_final/test_raw.jsonl "
        f"--out_dir data/pareto_records_split_norm/{city}/paper_final"
    )


def _quota_args(params: dict[str, Any]) -> str:
    return " ".join(
        f"--reversal_template_quota {key}:{value}"
        for key, value in params["reversal_template_quota"].items()
    )


def _build_pairs_command(city: str, split: str, params: dict[str, Any]) -> str:
    return (
        "python pareto/data/build_pairs.py "
        f"--buffers data/pareto_records_split_norm/{city}/paper_final/{split}_raw.jsonl "
        f"--out_dir data/pareto_pairs/{city}/paper_final/{split} "
        f"--num_objective_pairs {params['num_objective_pairs']} "
        f"--num_preference_pairs {params['num_preference_pairs']} "
        f"--num_dominance_pairs {params['num_dominance_pairs']} "
        f"--num_reversal_pairs {params['num_reversal_pairs']} "
        f"--seed {params['split_seed']} --tie_margin {params['tie_margin']} --split {split} "
        f"--min_efficiency_stability_conflict {params['min_efficiency_stability_conflict']} "
        f"{_quota_args(params)}"
    ).strip()


def _validate_pairs_command(city: str, split: str, params: dict[str, Any]) -> str:
    return (
        "python pareto/data/validate_pairs.py "
        f"--pairs_dir data/pareto_pairs/{city}/paper_final/{split} "
        f"--report records/paper_final_data_audit/{city}/{split}_pair_report.json "
        "--strict --require_no_ties "
        f"--min_objective_per_head {params['strict_min_objective_per_head']} "
        f"--min_preference_pairs {params['strict_min_preference_pairs']} "
        f"--min_dominance_pairs {params['strict_min_dominance_pairs']} "
        f"--min_reversal_pairs {params['strict_min_reversal_pairs']} "
        f"--positive_ratio_by_objective_low {params['positive_ratio_by_objective_low']} "
        f"--positive_ratio_by_objective_high {params['positive_ratio_by_objective_high']}"
    )


def _merge_params(overrides: dict[str, Any] | None) -> dict[str, Any]:
    params = dict(PAPER_FINAL_DATA_DEFAULTS)
    if overrides:
        params.update(overrides)
    return params


def build_paper_final_data_generation_request(
    *,
    run_id: str,
    parameter_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = _merge_params(parameter_overrides)
    collection_rows = []
    city_rows = []
    for city, traffic_file in REQUIRED_CITY_TRAFFIC.items():
        for policy in PAPER_FINAL_DATA_POLICIES:
            for seed in PAPER_FINAL_SEEDS:
                collection_rows.append(
                    {
                        "city": city,
                        "traffic_file": traffic_file,
                        "policy": policy,
                        "seed": int(seed),
                        "output_jsonl": _buffer_jsonl(city, policy, int(seed), run_id),
                        "transitions_jsonl": _transition_jsonl(city, policy, int(seed), run_id),
                        "command_preview": _collect_command(city, traffic_file, policy, int(seed), run_id, params),
                        "executes_generation_now": False,
                    }
                )
        pair_rows = []
        for split in PAPER_FINAL_SPLITS:
            pair_rows.append(
                {
                    "split": split,
                    "pairs_dir": f"data/pareto_pairs/{city}/paper_final/{split}",
                    "build_pairs_command_preview": _build_pairs_command(city, split, params),
                    "validate_pairs_command_preview": _validate_pairs_command(city, split, params),
                    "executes_generation_now": False,
                }
            )
        city_rows.append(
            {
                "city": city,
                "traffic_file": traffic_file,
                "raw_split_root": f"data/pareto_records_split/{city}/paper_final",
                "normalized_records_root": f"data/pareto_records_split_norm/{city}/paper_final",
                "pairs_root": f"data/pareto_pairs/{city}/paper_final",
                "normalizer_path": _normalizer_path(city),
                "split_records_command_preview": _split_command(city, run_id, params),
                "fit_normalizer_command_preview": _fit_normalizer_command(city, params),
                "apply_normalizer_command_preview": _apply_normalizer_command(city),
                "pair_rows": pair_rows,
                "final_audit_command_preview": "python -m pareto.data.paper_final_data_roots",
                "executes_generation_now": False,
            }
        )
    return validate_paper_final_data_generation_request(
        {
            "packet_type": "paper_final_data_generation_request",
            "status": "ready_request",
            "run_id": run_id,
            "parameters": params,
            "collection_rows": collection_rows,
            "city_rows": city_rows,
            "execution_allowed_now": False,
            "executes_generation_now": False,
            "reads_final_traffic_result_values": False,
            "writes_records_paper_final": False,
        }
    )


def validate_paper_final_data_generation_request(request: dict[str, Any]) -> dict[str, Any]:
    if request.get("packet_type") != "paper_final_data_generation_request":
        raise ValueError("packet_type must be paper_final_data_generation_request")
    if request.get("status") not in ALLOWED_DATA_GENERATION_REQUEST_STATUSES:
        raise ValueError(f"unknown data generation request status: {request.get('status')}")
    if request.get("execution_allowed_now") is not False:
        raise ValueError("data generation request must keep execution_allowed_now false")
    if request.get("executes_generation_now") is not False:
        raise ValueError("data generation request must be non-executing")
    if request.get("reads_final_traffic_result_values") is not False:
        raise ValueError("data generation request must not read final traffic result values")
    if request.get("writes_records_paper_final") is not False:
        raise ValueError("data generation request must not write records/paper_final")
    if request["status"] == "missing_blocker":
        if not request.get("blocker"):
            raise ValueError("missing data generation request must include blocker")
        return dict(request)

    expected_collection = {
        (city, policy, seed)
        for city in REQUIRED_CITY_TRAFFIC
        for policy in PAPER_FINAL_DATA_POLICIES
        for seed in PAPER_FINAL_SEEDS
    }
    observed_collection = {
        (row.get("city"), row.get("policy"), row.get("seed"))
        for row in request.get("collection_rows") or []
    }
    missing_collection = sorted(expected_collection - observed_collection)
    if missing_collection:
        raise ValueError(f"data generation request missing collection rows: {missing_collection}")

    city_rows = request.get("city_rows") or []
    observed_cities = {row.get("city") for row in city_rows}
    missing_cities = sorted(set(REQUIRED_CITY_TRAFFIC) - observed_cities)
    if missing_cities:
        raise ValueError(f"data generation request missing city rows: {missing_cities}")
    for row in city_rows:
        city = row.get("city")
        if city not in REQUIRED_CITY_TRAFFIC:
            raise ValueError(f"unknown city in data generation request: {city}")
        if row.get("executes_generation_now") is not False:
            raise ValueError("city data generation row must be non-executing")
        if not str(row.get("normalized_records_root", "")).startswith(f"data/pareto_records_split_norm/{city}/"):
            raise ValueError("normalized_records_root must be under data/pareto_records_split_norm")
        if not str(row.get("pairs_root", "")).startswith(f"data/pareto_pairs/{city}/"):
            raise ValueError("pairs_root must be under data/pareto_pairs")
        pair_rows = row.get("pair_rows") or []
        observed_splits = {pair_row.get("split") for pair_row in pair_rows}
        if observed_splits != set(PAPER_FINAL_SPLITS):
            raise ValueError(f"data generation request pair splits mismatch for {city}: {observed_splits}")
        for pair_row in pair_rows:
            if pair_row.get("executes_generation_now") is not False:
                raise ValueError("pair generation row must be non-executing")
    return dict(request)


def data_generation_request_blockers(request: dict[str, Any]) -> list[str]:
    validated = validate_paper_final_data_generation_request(request)
    if validated["status"] == "missing_blocker":
        return [f"data generation request: {validated['blocker']}"]
    return []
