from __future__ import annotations

from typing import Any

from pareto.rl.paper_final_experiment_manifest import REQUIRED_CITY_TRAFFIC
from pareto.rl.paper_learned_artifact_inventory import LEARNED_ARTIFACT_FAMILIES, REQUIRED_LEARNED_ARTIFACT_BASELINES


ALLOWED_LEARNED_GENERATION_STATUSES = {"request_only"}


def build_learned_artifact_generation_request(*, run_id: str) -> dict[str, Any]:
    if "/" in run_id or not run_id:
        raise ValueError("learned artifact generation run_id must be a simple path segment")
    rows = []
    for baseline in REQUIRED_LEARNED_ARTIFACT_BASELINES:
        family = LEARNED_ARTIFACT_FAMILIES[baseline]
        for city in REQUIRED_CITY_TRAFFIC:
            root = f"model_weights/{family}/{city}/paper_final/{run_id}"
            rows.append(
                {
                    "baseline": baseline,
                    "city": city,
                    "artifact_family": family,
                    "run_id": run_id,
                    "model_path": f"{root}/model.pt",
                    "objective_normalizer_path": f"{root}/objective_normalizer.json",
                    "model_hash": None,
                    "objective_normalizer_hash": None,
                    "status": "request_only",
                    "executes_training_now": False,
                }
            )
    return validate_learned_artifact_generation_request(
        {
            "packet_type": "paper_learned_artifact_generation_request",
            "status": "request_only",
            "run_id": run_id,
            "rows": rows,
            "executes_training_now": False,
        }
    )


def validate_learned_artifact_generation_request(request: dict[str, Any]) -> dict[str, Any]:
    if request.get("status") not in ALLOWED_LEARNED_GENERATION_STATUSES:
        raise ValueError(f"unknown learned artifact generation status: {request.get('status')}")
    if request.get("executes_training_now") is not False:
        raise ValueError("learned artifact generation request must be non-executing")
    rows = request.get("rows") or []
    expected = {(baseline, city) for baseline in REQUIRED_LEARNED_ARTIFACT_BASELINES for city in REQUIRED_CITY_TRAFFIC}
    observed = {(row.get("baseline"), row.get("city")) for row in rows}
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"learned artifact generation request missing rows: {missing}")
    for row in rows:
        baseline = row.get("baseline")
        city = row.get("city")
        if baseline not in REQUIRED_LEARNED_ARTIFACT_BASELINES:
            raise ValueError(f"unknown learned artifact baseline: {baseline}")
        if city not in REQUIRED_CITY_TRAFFIC:
            raise ValueError(f"unknown learned artifact city: {city}")
        family = LEARNED_ARTIFACT_FAMILIES[str(baseline)]
        expected_prefix = f"model_weights/{family}/{city}/paper_final/"
        if not str(row.get("model_path") or "").startswith(expected_prefix):
            raise ValueError("learned artifact model_path must be under paper_final root")
        if not str(row.get("objective_normalizer_path") or "").startswith(expected_prefix):
            raise ValueError("learned artifact objective_normalizer_path must be under paper_final root")
        if row.get("model_hash") is not None:
            raise ValueError("request row must not claim model_hash before artifact exists")
        if row.get("objective_normalizer_hash") is not None:
            raise ValueError("request row must not claim objective_normalizer_hash before artifact exists")
        if row.get("executes_training_now") is not False:
            raise ValueError("learned artifact generation row must be non-executing")
    return dict(request)
