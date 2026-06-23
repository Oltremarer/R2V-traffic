from __future__ import annotations

from typing import Any


REQUIRED_REPRESENTATION_METRICS = ("obj_acc", "pref_acc", "rev_acc", "dpr")
REPRESENTATION_ABLATIONS = (
    "scalar_head",
    "cond_scalar_rl",
    "without_l_obj",
    "without_l_pref",
    "without_l_dom",
    "full_vectorq",
)
ALLOWED_REPRESENTATION_STATUSES = {"implemented", "missing_blocker"}


def validate_representation_diagnostic_plan(plan: dict[str, Any]) -> dict[str, Any]:
    metrics = plan.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("representation metrics must be an object")
    for metric in REQUIRED_REPRESENTATION_METRICS:
        if metric not in metrics:
            raise ValueError(f"missing representation metric: {metric}")
        if metrics[metric].get("status") not in ALLOWED_REPRESENTATION_STATUSES:
            raise ValueError(f"unknown representation metric status: {metric}")

    ablations = plan.get("ablations")
    if not isinstance(ablations, dict):
        raise ValueError("representation ablations must be an object")
    for ablation in REPRESENTATION_ABLATIONS:
        if ablation not in ablations:
            raise ValueError(f"missing representation ablation: {ablation}")
        row = ablations[ablation]
        status = row.get("status")
        if status not in ALLOWED_REPRESENTATION_STATUSES:
            raise ValueError(f"unknown representation ablation status: {ablation}")
        if status == "implemented":
            if not row.get("checkpoint_source"):
                raise ValueError(f"representation ablation {ablation} missing checkpoint_source")
            if not row.get("training_command"):
                raise ValueError(f"representation ablation {ablation} missing training_command")

    outputs = plan.get("outputs") or {}
    if outputs.get("raw_only") is not True:
        raise ValueError("representation diagnostics must be raw-only before result gates")
    if outputs.get("paper_table_generated") is not False:
        raise ValueError("representation diagnostic packet must not generate a paper table")
    if outputs.get("ranking_generated") is not False:
        raise ValueError("representation diagnostic packet must not generate ranking")
    return dict(plan)


def representation_diagnostic_blockers(plan: dict[str, Any]) -> list[str]:
    validated = validate_representation_diagnostic_plan(plan)
    blockers: list[str] = []
    for metric, row in validated.get("metrics", {}).items():
        if row.get("status") == "missing_blocker":
            blockers.append(f"metric {metric}: {row.get('blocker') or 'source missing'}")
    for ablation, row in validated.get("ablations", {}).items():
        if row.get("status") == "missing_blocker":
            blockers.append(f"ablation {ablation}: {row.get('blocker') or 'artifact missing'}")
    return blockers
