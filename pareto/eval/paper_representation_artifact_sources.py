from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pareto.rl.paper_final_experiment_manifest import REQUIRED_CITY_TRAFFIC


REQUIRED_REPRESENTATION_ARTIFACT_CITIES = tuple(REQUIRED_CITY_TRAFFIC)
REQUIRED_REPRESENTATION_MODEL_FAMILIES = ("VectorQ-PPO", "Cond-Scalar-RL")
REQUIRED_REPRESENTATION_METRIC_KEYS = {
    "obj_acc": ("obj_acc_mean",),
    "pref_acc": ("pref_acc",),
    "rev_acc": ("rev_acc",),
    "dpr": ("dpr_head", "dpr_utility"),
}
REPRESENTATION_PACKET_VERSION = "representation-formal-gate-packet-v1"
ALLOWED_REPRESENTATION_ARTIFACT_STATUSES = {"implemented_guarded_preview", "missing_blocker"}
PAPER_FINAL_LEARNED_ARTIFACT_RUN_ID = "paper_final_20260603_v1"
PAPER_FINAL_VECTOR_EVIDENCE_ID = "pareto_quality_paper_final_20260603_v1"
PAPER_FINAL_SCALAR_EVIDENCE_ID = "cond_scalar_paper_final_20260603_v1"
REMEDIATION_LEARNED_ARTIFACT_RUN_ID = "paper_final_rep_remediate_20260603_v1"
REMEDIATION_VECTOR_EVIDENCE_ID = "pareto_quality_paper_final_rep_remediate_20260603_v1"
REMEDIATION_SCALAR_EVIDENCE_ID = "cond_scalar_paper_final_rep_remediate_20260603_v1"
ALLOWED_PAPER_FINAL_REPRESENTATION_IDS = (
    (
        PAPER_FINAL_LEARNED_ARTIFACT_RUN_ID,
        PAPER_FINAL_VECTOR_EVIDENCE_ID,
        PAPER_FINAL_SCALAR_EVIDENCE_ID,
    ),
    (
        REMEDIATION_LEARNED_ARTIFACT_RUN_ID,
        REMEDIATION_VECTOR_EVIDENCE_ID,
        REMEDIATION_SCALAR_EVIDENCE_ID,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _normalize_city(value: Any) -> str:
    text = str(value or "").lower()
    aliases = {
        "jinan": "jinan",
        "hangzhou": "hangzhou",
        "newyork_28x7": "newyork_28x7",
        "new york": "newyork_28x7",
        "newyork": "newyork_28x7",
    }
    for alias, city in aliases.items():
        if alias in text:
            return city
    raise ValueError(f"unknown representation packet city: {value}")


def _packet_model_families(packet: dict[str, Any]) -> list[str]:
    families: list[str] = []
    if "selected_vector_run" in packet or "vector_metrics" in packet:
        families.append("VectorQ-PPO")
    if "condscalar_metrics" in packet or "condscalar_baseline_fairness_status" in packet:
        families.append("Cond-Scalar-RL")
    if not families:
        raise ValueError("representation packet missing model-family evidence")
    return families


def _model_family_slug(model_family: str) -> str:
    return model_family.lower().replace("-", "_")


def _path_matches_model_family(packet_path: Path, model_family: str) -> bool:
    return packet_path.parent.name == _model_family_slug(model_family)


def _require_paper_final_packet(packet: dict[str, Any], city: str) -> None:
    scope = packet.get("scope") or {}
    expected_scope_city = f"{city} paper-final offline records only"
    if str(scope.get("city") or "").lower() != expected_scope_city:
        raise ValueError("representation packet is not paper-final scope")

    selected_vector = packet.get("selected_vector_run") or {}
    dangerous_scalar = packet.get("dangerous_scalar_baseline") or {}
    for learned_run_id, vector_evidence_id, scalar_evidence_id in ALLOWED_PAPER_FINAL_REPRESENTATION_IDS:
        expected_vector_model_dir = f"model_weights/pareto_quality/{city}/paper_final/{learned_run_id}"
        expected_scalar_model_dir = f"model_weights/cond_scalar/{city}/paper_final/{learned_run_id}"
        if (
            selected_vector.get("run_id") == vector_evidence_id
            and dangerous_scalar.get("run_id") == scalar_evidence_id
            and str(selected_vector.get("model_dir") or "") == expected_vector_model_dir
            and str(dangerous_scalar.get("model_dir") or "") == expected_scalar_model_dir
        ):
            return
    raise ValueError("representation packet has non-paper-final or unapproved remediation ids")


def validate_representation_packet(packet: dict[str, Any]) -> dict[str, Any]:
    if packet.get("packet_version") != REPRESENTATION_PACKET_VERSION:
        raise ValueError("representation packet has unknown packet_version")
    scope = packet.get("scope") or {}
    if scope.get("traffic_result_value_reading_executed") is not False:
        raise ValueError("representation packet must not read final traffic result values")
    if scope.get("paper_result_claim") is not False:
        raise ValueError("representation packet must not make paper result claims")
    if packet.get("formal_experiment_requires_new_pro_approval") is not True:
        raise ValueError("representation packet must require new approval before formal experiment")
    if not packet.get("pro_approval_phrase"):
        raise ValueError("representation packet missing reviewer approval phrase")
    threshold_checks = packet.get("threshold_checks") or {}
    packet_keys: dict[str, list[str]] = {}
    for metric, source_keys in REQUIRED_REPRESENTATION_METRIC_KEYS.items():
        missing = [key for key in source_keys if key not in threshold_checks]
        if missing:
            raise ValueError(f"missing representation packet key for {metric}: {missing}")
        packet_keys[metric] = list(source_keys)
    city = _normalize_city(scope.get("city"))
    _require_paper_final_packet(packet, city)
    return {
        "city": city,
        "metrics": list(REQUIRED_REPRESENTATION_METRIC_KEYS),
        "packet_keys": packet_keys,
        "model_families": _packet_model_families(packet),
        "traffic_result_value_reading_executed": False,
        "paper_result_claim": False,
    }


def _missing_row(city: str, model_family: str, blocker: str) -> dict[str, Any]:
    return validate_representation_artifact_row(
        {
            "city": city,
            "model_family": model_family,
            "status": "missing_blocker",
            "blocker": blocker,
            "executes_training_now": False,
            "reads_final_traffic_result_values": False,
        }
    )


def inventory_representation_artifact_sources(root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    packet_paths = sorted((root_path / "docs" / "pro_reviews").glob("**/representation_formal_gate_packet.json"))
    discovered: dict[tuple[str, str], dict[str, Any]] = {}
    ignored_packets: list[dict[str, str]] = []
    for packet_path in packet_paths:
        import json

        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        try:
            source = validate_representation_packet(packet)
        except ValueError as exc:
            ignored_packets.append(
                {
                    "packet_path": _rel(packet_path, root_path),
                    "reason": str(exc),
                }
            )
            continue
        for model_family in source["model_families"]:
            key = (source["city"], model_family)
            row = {
                "city": source["city"],
                "model_family": model_family,
                "status": "implemented_guarded_preview",
                "packet_path": _rel(packet_path, root_path),
                "packet_hash": _sha256(packet_path),
                "metrics": source["metrics"],
                "packet_keys": source["packet_keys"],
                "executes_training_now": False,
                "reads_final_traffic_result_values": False,
                "paper_result_claim": False,
            }
            existing = discovered.get(key)
            if existing is None or (
                not str(existing.get("packet_path") or "").split("/")[-2] == _model_family_slug(model_family)
                and _path_matches_model_family(packet_path, model_family)
            ):
                discovered[key] = row

    rows: list[dict[str, Any]] = []
    for city in REQUIRED_REPRESENTATION_ARTIFACT_CITIES:
        for model_family in REQUIRED_REPRESENTATION_MODEL_FAMILIES:
            row = discovered.get((city, model_family))
            if row is None:
                row = _missing_row(city, model_family, "representation artifact source missing")
            rows.append(validate_representation_artifact_row(row))
    audit = {
        "packet_type": "paper_representation_artifact_sources",
        "root": root_path.as_posix(),
        "required_cities": list(REQUIRED_REPRESENTATION_ARTIFACT_CITIES),
        "required_model_families": list(REQUIRED_REPRESENTATION_MODEL_FAMILIES),
        "required_metrics": list(REQUIRED_REPRESENTATION_METRIC_KEYS),
        "rows": rows,
        "ignored_packets": ignored_packets,
        "executes_training_now": False,
    }
    audit["coverage_status"] = "missing_blocker" if representation_artifact_blockers(audit) else "complete"
    return audit


def validate_representation_artifact_row(row: dict[str, Any]) -> dict[str, Any]:
    city = row.get("city")
    model_family = row.get("model_family")
    status = row.get("status")
    if city not in REQUIRED_REPRESENTATION_ARTIFACT_CITIES:
        raise ValueError(f"unknown representation artifact city: {city}")
    if model_family not in REQUIRED_REPRESENTATION_MODEL_FAMILIES:
        raise ValueError(f"unknown representation model family: {model_family}")
    if status not in ALLOWED_REPRESENTATION_ARTIFACT_STATUSES:
        raise ValueError(f"unknown representation artifact status: {status}")
    if row.get("executes_training_now") is not False:
        raise ValueError("representation artifact source inventory must be non-executing")
    if row.get("reads_final_traffic_result_values") is not False:
        raise ValueError("representation artifact source must not read final traffic result values")
    if status == "implemented_guarded_preview":
        if not str(row.get("packet_path") or "").endswith("representation_formal_gate_packet.json"):
            raise ValueError("representation artifact source must point to representation_formal_gate_packet.json")
        if len(str(row.get("packet_hash") or "")) != 64:
            raise ValueError("representation artifact source missing packet hash")
        metrics = list(row.get("metrics") or [])
        if metrics != list(REQUIRED_REPRESENTATION_METRIC_KEYS):
            raise ValueError("representation artifact source has incomplete metric coverage")
        packet_keys = row.get("packet_keys") or {}
        for metric, source_keys in REQUIRED_REPRESENTATION_METRIC_KEYS.items():
            if list(packet_keys.get(metric) or []) != list(source_keys):
                raise ValueError(f"representation artifact source has wrong packet keys for {metric}")
        if row.get("paper_result_claim") is not False:
            raise ValueError("representation artifact source must not make paper result claims")
    if status == "missing_blocker" and not row.get("blocker"):
        raise ValueError("missing representation artifact row must include blocker")
    return dict(row)


def representation_artifact_blockers(audit: dict[str, Any]) -> list[str]:
    rows = audit.get("rows") or []
    blockers: list[str] = []
    for row in rows:
        validated = validate_representation_artifact_row(row)
        if validated["status"] == "missing_blocker":
            blockers.append(
                f"representation {validated['city']} {validated['model_family']}: {validated['blocker']}"
            )
    return blockers
