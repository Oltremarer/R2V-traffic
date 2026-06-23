from __future__ import annotations

from typing import Any

from pareto.eval.paper_representation_artifact_sources import (
    REQUIRED_REPRESENTATION_ARTIFACT_CITIES,
    REQUIRED_REPRESENTATION_METRIC_KEYS,
    REQUIRED_REPRESENTATION_MODEL_FAMILIES,
)


ALLOWED_REPRESENTATION_GENERATION_STATUSES = {"request_only"}


def _slug(value: str) -> str:
    return value.lower().replace("-", "_")


def build_representation_generation_request(*, request_id: str) -> dict[str, Any]:
    if "/" in request_id or not request_id:
        raise ValueError("representation generation request_id must be a simple path segment")
    rows = []
    for city in REQUIRED_REPRESENTATION_ARTIFACT_CITIES:
        for model_family in REQUIRED_REPRESENTATION_MODEL_FAMILIES:
            rows.append(
                {
                    "city": city,
                    "model_family": model_family,
                    "status": "request_only",
                    "packet_path": (
                        "docs/pro_reviews/"
                        f"{request_id}/{city}/{_slug(model_family)}/representation_formal_gate_packet.json"
                    ),
                    "packet_keys": {metric: list(keys) for metric, keys in REQUIRED_REPRESENTATION_METRIC_KEYS.items()},
                    "executes_generation_now": False,
                    "reads_final_traffic_result_values": False,
                    "paper_result_claim": False,
                }
            )
    return validate_representation_generation_request(
        {
            "packet_type": "paper_representation_generation_request",
            "status": "request_only",
            "request_id": request_id,
            "rows": rows,
            "executes_generation_now": False,
            "reads_final_traffic_result_values": False,
        }
    )


def validate_representation_generation_request(request: dict[str, Any]) -> dict[str, Any]:
    if request.get("status") not in ALLOWED_REPRESENTATION_GENERATION_STATUSES:
        raise ValueError(f"unknown representation generation status: {request.get('status')}")
    if request.get("executes_generation_now") is not False:
        raise ValueError("representation generation request must be non-executing")
    if request.get("reads_final_traffic_result_values") is not False:
        raise ValueError("representation generation request must not read final traffic result values")
    rows = request.get("rows") or []
    expected = {
        (city, model_family)
        for city in REQUIRED_REPRESENTATION_ARTIFACT_CITIES
        for model_family in REQUIRED_REPRESENTATION_MODEL_FAMILIES
    }
    observed = {(row.get("city"), row.get("model_family")) for row in rows}
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"representation generation request missing rows: {missing}")
    for row in rows:
        if row.get("executes_generation_now") is not False:
            raise ValueError("representation generation row must be non-executing")
        if row.get("reads_final_traffic_result_values") is not False:
            raise ValueError("representation generation row must not read final traffic result values")
        if row.get("paper_result_claim") is not False:
            raise ValueError("representation generation row must not make paper result claims")
        if not str(row.get("packet_path") or "").startswith("docs/pro_reviews/"):
            raise ValueError("representation generation packet_path must be under docs/pro_reviews")
        packet_keys = row.get("packet_keys") or {}
        for metric, keys in REQUIRED_REPRESENTATION_METRIC_KEYS.items():
            if list(packet_keys.get(metric) or []) != list(keys):
                raise ValueError(f"representation generation packet_keys mismatch for {metric}")
    return dict(request)
