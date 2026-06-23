from __future__ import annotations

from typing import Any

from pareto.eval.paper_representation_artifact_sources import (
    REQUIRED_REPRESENTATION_ARTIFACT_CITIES,
    REQUIRED_REPRESENTATION_METRIC_KEYS,
    REQUIRED_REPRESENTATION_MODEL_FAMILIES,
)


ALLOWED_REPRESENTATION_COMMAND_STATUSES = {"ready_request", "missing_blocker"}


def _slug(value: str) -> str:
    return value.lower().replace("-", "_")


DEFAULT_LEARNED_ARTIFACT_RUN_ID = "paper_final_20260603_v1"


def build_representation_command_preview(
    *,
    learned_artifacts_ready: bool,
    request_id: str,
    learned_artifact_run_id: str = DEFAULT_LEARNED_ARTIFACT_RUN_ID,
    vector_evidence_id: str | None = None,
    scalar_evidence_id: str | None = None,
    evidence_dir_suffix: str = "paper_final_evidence",
) -> dict[str, Any]:
    vector_evidence_id = vector_evidence_id or learned_artifact_run_id
    scalar_evidence_id = scalar_evidence_id or learned_artifact_run_id
    if not learned_artifacts_ready:
        return validate_representation_command_preview(
            {
                "packet_type": "paper_representation_command_preview",
                "status": "missing_blocker",
                "blocker": "learned artifacts or evidence dirs incomplete",
                "rows": [],
                "executes_generation_now": False,
                "reads_final_traffic_result_values": False,
            }
        )
    rows = []
    for city in REQUIRED_REPRESENTATION_ARTIFACT_CITIES:
        evidence_dir = f"docs/pro_reviews/pareto_offline_representation_{city}_{evidence_dir_suffix}"
        out_dir = f"docs/pro_reviews/{request_id}/{city}"
        for model_family in REQUIRED_REPRESENTATION_MODEL_FAMILIES:
            model_slug = _slug(model_family)
            vector_model_dir = f"model_weights/pareto_quality/{city}/paper_final/{learned_artifact_run_id}"
            scalar_model_dir = f"model_weights/cond_scalar/{city}/paper_final/{learned_artifact_run_id}"
            row_out_dir = f"{out_dir}/{model_slug}"
            rows.append(
                {
                    "city": city,
                    "model_family": model_family,
                    "learned_artifact_run_id": learned_artifact_run_id,
                    "vector_evidence_id": vector_evidence_id,
                    "scalar_evidence_id": scalar_evidence_id,
                    "evidence_dir": evidence_dir,
                    "out_dir": row_out_dir,
                    "packet_keys": {metric: list(keys) for metric, keys in REQUIRED_REPRESENTATION_METRIC_KEYS.items()},
                    "command_preview": (
                        "python -m pareto.eval.representation_formal_gate_packet "
                        f"--evidence_dir {evidence_dir} --out_dir {row_out_dir} "
                        f"--vector_run_id {vector_evidence_id} --scalar_run_id {scalar_evidence_id} "
                        f"--vector_model_dir {vector_model_dir} --scalar_model_dir {scalar_model_dir} "
                        f"--city {city} --normalizer_filename objective_norm_paper_final.json"
                    ),
                    "executes_generation_now": False,
                    "reads_final_traffic_result_values": False,
                    "paper_result_claim": False,
                }
            )
    return validate_representation_command_preview(
        {
            "packet_type": "paper_representation_command_preview",
            "status": "ready_request",
            "request_id": request_id,
            "learned_artifact_run_id": learned_artifact_run_id,
            "vector_evidence_id": vector_evidence_id,
            "scalar_evidence_id": scalar_evidence_id,
            "evidence_dir_suffix": evidence_dir_suffix,
            "rows": rows,
            "executes_generation_now": False,
            "reads_final_traffic_result_values": False,
        }
    )


def validate_representation_command_preview(preview: dict[str, Any]) -> dict[str, Any]:
    if preview.get("status") not in ALLOWED_REPRESENTATION_COMMAND_STATUSES:
        raise ValueError(f"unknown representation command preview status: {preview.get('status')}")
    if preview.get("executes_generation_now") is not False:
        raise ValueError("representation command preview must be non-executing")
    if preview.get("reads_final_traffic_result_values") is not False:
        raise ValueError("representation command preview must not read final traffic result values")
    if preview.get("status") == "missing_blocker":
        if not preview.get("blocker"):
            raise ValueError("missing representation command preview must include blocker")
        return dict(preview)
    rows = preview.get("rows") or []
    expected = {
        (city, family)
        for city in REQUIRED_REPRESENTATION_ARTIFACT_CITIES
        for family in REQUIRED_REPRESENTATION_MODEL_FAMILIES
    }
    observed = {(row.get("city"), row.get("model_family")) for row in rows}
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"representation command preview missing rows: {missing}")
    for row in rows:
        if not str(row.get("out_dir") or "").startswith("docs/pro_reviews/"):
            raise ValueError("representation command out_dir must be under docs/pro_reviews")
        if row.get("executes_generation_now") is not False:
            raise ValueError("representation command row must be non-executing")
        if row.get("reads_final_traffic_result_values") is not False:
            raise ValueError("representation command row must not read final traffic result values")
        if row.get("paper_result_claim") is not False:
            raise ValueError("representation command row must not make paper result claims")
        packet_keys = row.get("packet_keys") or {}
        for metric, keys in REQUIRED_REPRESENTATION_METRIC_KEYS.items():
            if list(packet_keys.get(metric) or []) != list(keys):
                raise ValueError(f"representation command packet_keys mismatch for {metric}")
        if "python -m pareto.eval.representation_formal_gate_packet" not in str(row.get("command_preview") or ""):
            raise ValueError("representation command preview must use representation_formal_gate_packet")
        vector_evidence_id = str(row.get("vector_evidence_id") or preview.get("vector_evidence_id") or "")
        scalar_evidence_id = str(row.get("scalar_evidence_id") or preview.get("scalar_evidence_id") or "")
        command = str(row.get("command_preview") or "")
        if vector_evidence_id and scalar_evidence_id and (
            f"--vector_run_id {vector_evidence_id}" not in command
            or f"--scalar_run_id {scalar_evidence_id}" not in command
        ):
            raise ValueError("representation command preview must use requested vector/scalar evidence ids")
        if f"--city {row.get('city')}" not in command:
            raise ValueError("representation command preview must include city")
        if "--normalizer_filename objective_norm_paper_final.json" not in command:
            raise ValueError("representation command preview must use paper-final normalizer filename")
    return dict(preview)


def representation_command_blockers(preview: dict[str, Any]) -> list[str]:
    validated = validate_representation_command_preview(preview)
    if validated["status"] == "missing_blocker":
        return [f"representation: {validated['blocker']}"]
    return []
