from __future__ import annotations

from typing import Any

from pareto.rl.paper_final_experiment_manifest import REQUIRED_CITY_TRAFFIC


REQUIRED_C2T_SCALAR_EXCLUSION_APPROVAL = (
    "PPTS PARETO PPO C2T_SCALAR EXCLUSION APPROVED WITH PAPER CLAIM LIMITATION"
)
ALLOWED_C2T_SCALAR_STATUSES = {"implemented_guarded_preview", "excluded_by_reviewer", "missing_blocker"}
ALL_PAPER_CITIES = tuple(REQUIRED_CITY_TRAFFIC)


def _validate_command_preview(command_preview: dict[str, Any]) -> dict[str, Any]:
    if not command_preview.get("command_family"):
        raise ValueError("C2T-scalar command preview missing command_family")
    if command_preview.get("executes_training_now") is not False:
        raise ValueError("C2T-scalar command preview must be non-executing")
    missing_cities = sorted(set(ALL_PAPER_CITIES) - set(command_preview.get("city_support") or []))
    if missing_cities:
        raise ValueError(f"C2T-scalar command preview missing city support: {missing_cities}")
    output_root = str(command_preview.get("output_root") or "")
    if not output_root.startswith("records/paper_final/"):
        raise ValueError("C2T-scalar command preview output_root must be under records/paper_final")
    return dict(command_preview)


def build_c2t_scalar_readiness(
    *,
    command_preview: dict[str, Any] | None = None,
    exclusion_approval_phrase: str | None = None,
    paper_claim_limitation: str | None = None,
) -> dict[str, Any]:
    if command_preview is not None and exclusion_approval_phrase:
        raise ValueError("C2T-scalar readiness must use command preview or exclusion, not both")
    if command_preview is not None:
        preview = _validate_command_preview(command_preview)
        return validate_c2t_scalar_readiness(
            {
                "baseline": "C2T-scalar",
                "status": "implemented_guarded_preview",
                "command_family": preview["command_family"],
                "city_support": list(preview["city_support"]),
                "output_root": preview["output_root"],
                "executes_training_now": False,
            }
        )
    if exclusion_approval_phrase == REQUIRED_C2T_SCALAR_EXCLUSION_APPROVAL:
        return validate_c2t_scalar_readiness(
            {
                "baseline": "C2T-scalar",
                "status": "excluded_by_reviewer",
                "approval_phrase": exclusion_approval_phrase,
                "paper_claim_limitation": paper_claim_limitation,
                "executes_training_now": False,
            }
        )
    return validate_c2t_scalar_readiness(
        {
            "baseline": "C2T-scalar",
            "status": "missing_blocker",
            "approval_phrase": exclusion_approval_phrase or "",
            "executes_training_now": False,
            "blocker": "C2T-scalar command or exclusion approval missing",
        }
    )


def validate_c2t_scalar_readiness(row: dict[str, Any]) -> dict[str, Any]:
    status = row.get("status")
    if status not in ALLOWED_C2T_SCALAR_STATUSES:
        raise ValueError(f"unknown C2T-scalar readiness status: {status}")
    if row.get("baseline") not in {None, "C2T-scalar"}:
        raise ValueError("C2T-scalar readiness row must target C2T-scalar")
    if row.get("executes_training_now") is not False:
        raise ValueError("C2T-scalar readiness must be non-executing")
    if status == "implemented_guarded_preview":
        _validate_command_preview(
            {
                "command_family": row.get("command_family"),
                "city_support": row.get("city_support"),
                "executes_training_now": row.get("executes_training_now"),
                "output_root": row.get("output_root"),
            }
        )
    if status == "excluded_by_reviewer":
        if row.get("approval_phrase") != REQUIRED_C2T_SCALAR_EXCLUSION_APPROVAL:
            raise ValueError("C2T-scalar exclusion requires exact reviewer exclusion approval phrase")
        if not row.get("paper_claim_limitation"):
            raise ValueError("C2T-scalar exclusion must record paper claim limitation")
    if status == "missing_blocker" and not row.get("blocker"):
        raise ValueError("missing C2T-scalar readiness must include blocker")
    return dict(row)
