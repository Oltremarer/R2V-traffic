from __future__ import annotations

from typing import Any


WEIGHTED_MAPPING_CANDIDATE = "Weighted-RL -> weighted_proxy"
REQUIRED_WEIGHTED_MAPPING_APPROVAL = (
    "PPTS PARETO PPO WEIGHTED_RL WEIGHTED_PROXY MAPPING APPROVED FOR FINAL SUITE"
)
ALLOWED_WEIGHTED_MAPPING_STATUSES = {"implemented_guarded_preview", "missing_blocker"}


def build_weighted_mapping_policy(*, approval_phrase: str | None = None) -> dict[str, Any]:
    if approval_phrase == REQUIRED_WEIGHTED_MAPPING_APPROVAL:
        return validate_weighted_mapping_policy(
            {
                "baseline": "Weighted-RL",
                "method_id": "weighted_proxy",
                "status": "implemented_guarded_preview",
                "mapping_candidate": WEIGHTED_MAPPING_CANDIDATE,
                "approval_phrase": approval_phrase,
                "executes_training_now": False,
                "paper_claim_limitation": "Weighted-RL is reported as the reviewer-approved weighted_proxy mapping.",
            }
        )
    return validate_weighted_mapping_policy(
        {
            "baseline": "Weighted-RL",
            "method_id": "weighted_proxy",
            "status": "missing_blocker",
            "mapping_candidate": WEIGHTED_MAPPING_CANDIDATE,
            "approval_phrase": approval_phrase or "",
            "executes_training_now": False,
            "blocker": "Weighted-RL mapping requires exact reviewer approval phrase",
        }
    )


def validate_weighted_mapping_policy(row: dict[str, Any]) -> dict[str, Any]:
    status = row.get("status")
    if status not in ALLOWED_WEIGHTED_MAPPING_STATUSES:
        raise ValueError(f"unknown weighted mapping status: {status}")
    if row.get("baseline") not in {None, "Weighted-RL"}:
        raise ValueError("weighted mapping row must target Weighted-RL")
    if row.get("method_id") not in {None, "weighted_proxy"}:
        raise ValueError("weighted mapping row must target weighted_proxy")
    if row.get("mapping_candidate") != WEIGHTED_MAPPING_CANDIDATE:
        raise ValueError("weighted mapping candidate must be Weighted-RL -> weighted_proxy")
    if row.get("executes_training_now") is not False:
        raise ValueError("weighted mapping policy must be non-executing")
    if status == "implemented_guarded_preview":
        if row.get("approval_phrase") != REQUIRED_WEIGHTED_MAPPING_APPROVAL:
            raise ValueError("Weighted-RL mapping requires exact reviewer approval phrase")
        if not row.get("paper_claim_limitation"):
            raise ValueError("Weighted-RL mapping preview must record paper claim limitation")
    if status == "missing_blocker" and not row.get("blocker"):
        raise ValueError("missing Weighted-RL mapping policy must include blocker")
    return dict(row)
