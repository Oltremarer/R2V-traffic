from __future__ import annotations

from typing import Any

from pareto.data.paper_final_data_roots import data_root_blockers, validate_paper_final_data_root_audit
from pareto.rl.paper_learned_artifact_inventory import LEARNED_ARTIFACT_FAMILIES, REQUIRED_LEARNED_ARTIFACT_BASELINES


ALLOWED_LEARNED_COMMAND_STATUSES = {"ready_request", "missing_blocker"}
PAPER_LEARNED_HIDDEN_DIM = 256


def _row_by_city(data_audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["city"]): dict(row) for row in data_audit.get("rows") or []}


def _normalizer_source_for(city: str) -> str:
    return f"data/normalizers/{city}/objective_norm_paper_final.json"


def _with_normalizer_copy(command: str, normalizer_source: str, output_dir: str) -> str:
    return f"{command} && cp {normalizer_source} {output_dir}/objective_normalizer.json"


def _command_for(baseline: str, records_root: str, pairs_root: str, output_dir: str, normalizer_source: str) -> str:
    if baseline == "Cond-Scalar-RL":
        return _with_normalizer_copy(
            "python pareto/train_conditioned_scalar.py "
            f"--records_root {records_root} --pairs_root {pairs_root} --output_dir {output_dir} "
            "--architecture film --training_schedule joint --reversal_sampler template_balanced "
            f"--hidden_dim {PAPER_LEARNED_HIDDEN_DIM} --epochs 40 --batch_size 128 --device cuda",
            normalizer_source,
            output_dir,
        )
    if baseline == "VectorQ-PPO":
        return _with_normalizer_copy(
            "python pareto/train_vector_quality.py "
            f"--records_root {records_root} --pairs_root {pairs_root} --output_dir {output_dir} "
            "--architecture residual_tower --training_schedule joint --reversal_sampler template_balanced "
            "--score_mode low_rank_interaction --dominance_coord_loss_weight 1.5 "
            "--dominance_utility_loss_weight 0.3 --isotonic_dominance_weight 3.0 "
            "--use_objective_margins_for_dominance "
            f"--hidden_dim {PAPER_LEARNED_HIDDEN_DIM} --epochs 40 --batch_size 128 --device cuda",
            normalizer_source,
            output_dir,
        )
    raise ValueError(f"unknown learned artifact baseline: {baseline}")


def build_learned_artifact_command_preview(*, data_audit: dict[str, Any], run_id: str) -> dict[str, Any]:
    validate_paper_final_data_root_audit(data_audit)
    blockers = data_root_blockers(data_audit)
    if blockers:
        return validate_learned_artifact_command_preview(
            {
                "packet_type": "paper_learned_artifact_command_preview",
                "status": "missing_blocker",
                "blocker": "data roots incomplete",
                "source_blockers": blockers,
                "rows": [],
                "executes_training_now": False,
            }
        )
    by_city = _row_by_city(data_audit)
    rows = []
    for baseline in REQUIRED_LEARNED_ARTIFACT_BASELINES:
        family = LEARNED_ARTIFACT_FAMILIES[baseline]
        for city, data_row in by_city.items():
            output_dir = f"model_weights/{family}/{city}/paper_final/{run_id}"
            normalizer_source = _normalizer_source_for(city)
            rows.append(
                {
                    "baseline": baseline,
                    "city": city,
                    "records_root": data_row["records_root"],
                    "pairs_root": data_row["pairs_root"],
                    "output_dir": output_dir,
                    "objective_normalizer_source_path": normalizer_source,
                    "model_hash": None,
                    "objective_normalizer_hash": None,
                    "command_preview": _command_for(
                        baseline,
                        data_row["records_root"],
                        data_row["pairs_root"],
                        output_dir,
                        normalizer_source,
                    ),
                    "executes_training_now": False,
                }
            )
    return validate_learned_artifact_command_preview(
        {
            "packet_type": "paper_learned_artifact_command_preview",
            "status": "ready_request",
            "run_id": run_id,
            "rows": rows,
            "executes_training_now": False,
        }
    )


def validate_learned_artifact_command_preview(preview: dict[str, Any]) -> dict[str, Any]:
    if preview.get("status") not in ALLOWED_LEARNED_COMMAND_STATUSES:
        raise ValueError(f"unknown learned artifact command preview status: {preview.get('status')}")
    if preview.get("executes_training_now") is not False:
        raise ValueError("learned artifact command preview must be non-executing")
    if preview.get("status") == "missing_blocker":
        if not preview.get("blocker"):
            raise ValueError("missing learned artifact command preview must include blocker")
        return dict(preview)
    rows = preview.get("rows") or []
    expected = {(baseline, city) for baseline in REQUIRED_LEARNED_ARTIFACT_BASELINES for city in ("jinan", "hangzhou", "newyork_28x7")}
    observed = {(row.get("baseline"), row.get("city")) for row in rows}
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"learned artifact command preview missing rows: {missing}")
    for row in rows:
        baseline = row.get("baseline")
        city = row.get("city")
        if baseline not in REQUIRED_LEARNED_ARTIFACT_BASELINES:
            raise ValueError(f"unknown learned artifact baseline: {baseline}")
        family = LEARNED_ARTIFACT_FAMILIES[str(baseline)]
        if not str(row.get("output_dir") or "").startswith(f"model_weights/{family}/{city}/paper_final/"):
            raise ValueError("learned artifact output_dir must be under paper_final")
        if row.get("model_hash") is not None:
            raise ValueError("learned artifact preview row must not claim model_hash")
        if row.get("objective_normalizer_hash") is not None:
            raise ValueError("learned artifact preview row must not claim objective_normalizer_hash")
        if row.get("executes_training_now") is not False:
            raise ValueError("learned artifact command row must be non-executing")
        command = str(row.get("command_preview") or "")
        expected_normalizer_source = _normalizer_source_for(str(city))
        if row.get("objective_normalizer_source_path") != expected_normalizer_source:
            raise ValueError("learned artifact normalizer source path must use paper_final train-only normalizer")
        if f"--hidden_dim {PAPER_LEARNED_HIDDEN_DIM}" not in command:
            raise ValueError("learned artifact command preview must use paper-scale hidden_dim")
        if f"cp {expected_normalizer_source} {row.get('output_dir')}/objective_normalizer.json" not in command:
            raise ValueError("learned artifact command preview must copy objective normalizer into artifact dir")
        if baseline == "Cond-Scalar-RL" and "train_conditioned_scalar.py" not in command:
            raise ValueError("Cond-Scalar-RL command preview must use train_conditioned_scalar.py")
        if baseline == "VectorQ-PPO" and "train_vector_quality.py" not in command:
            raise ValueError("VectorQ-PPO command preview must use train_vector_quality.py")
    return dict(preview)


def learned_artifact_command_blockers(preview: dict[str, Any]) -> list[str]:
    validated = validate_learned_artifact_command_preview(preview)
    if validated["status"] == "missing_blocker":
        return [f"learned artifacts: {validated['blocker']}"]
    return []
