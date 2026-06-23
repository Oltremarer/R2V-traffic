from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pareto.eval.paper_representation_artifact_sources import validate_representation_packet
from pareto.rl.paper_final_experiment_manifest import REQUIRED_CITY_TRAFFIC


PAPER_REPRESENTATION_PACKET_GLOB = "docs/pro_reviews/**/representation_formal_gate_packet.json"
PAPER_FINAL_REPRESENTATION_PACKET_OUTPUT_ROOT = (
    "docs/pro_reviews/pareto_ppo_final_representation_packet_generation_execution_2026-06-03"
)
FORBIDDEN_TRUE_SCOPE_FLAGS = (
    "traffic_result_value_reading_executed",
    "method_ranking_executed",
    "paper_result_claim",
    "new_cityflow_ppo_run_executed",
    "multi_seed_executed",
    "city_expansion_executed",
)
REQUIRED_REPRESENTATION_DIAGNOSTIC_LIMITATION_APPROVAL = (
    "PPTS PARETO PPO REPRESENTATION FORMAL GATE DIAGNOSTIC LIMITATION APPROVED FOR SCOPE-LIMITED FINAL SUITE"
)
ALLOWED_REPRESENTATION_SCOPE_LIMITATION_STATUSES = {
    "diagnostic_limitation_by_reviewer",
    "missing_blocker",
}


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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _paper_final_city_from_scope(packet: dict[str, Any]) -> str | None:
    scope_city = str((packet.get("scope") or {}).get("city") or "").lower()
    if "paper-final" not in scope_city:
        return None
    for city in REQUIRED_CITY_TRAFFIC:
        if city in scope_city:
            return city
    return None


def _is_under_packet_output_root(
    packet_path: str,
    *,
    packet_output_root: str = PAPER_FINAL_REPRESENTATION_PACKET_OUTPUT_ROOT,
) -> bool:
    return packet_path.startswith(f"{packet_output_root}/")


def _packet_blockers(packet: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    gate = packet.get("formal_gate_decision") or {}
    if packet.get("formal_representation_pass") is not True:
        blockers.append("formal_representation_pass=false")
    if gate.get("representation_gate_pass") is not True:
        blockers.append("representation_gate_pass=false")
    claim_mode = gate.get("claim_mode")
    if claim_mode == "diagnostics_only":
        blockers.append("claim_mode=diagnostics_only")
    failed_reasons = list(gate.get("failed_reasons") or [])
    blockers.extend(str(reason) for reason in failed_reasons)
    if packet.get("formal_experiment_requires_new_pro_approval") is not True:
        blockers.append("formal_experiment_requires_new_pro_approval must be true")

    scope = packet.get("scope") or {}
    for flag in FORBIDDEN_TRUE_SCOPE_FLAGS:
        if scope.get(flag) is not False:
            blockers.append(f"{flag} must be false")

    for section in ("threshold_checks", "bootstrap_lower_bound_checks"):
        for metric, status in (packet.get(section) or {}).items():
            if isinstance(status, dict) and status.get("pass") is False:
                blockers.append(f"{section}.{metric} failed")
    return blockers


def build_representation_formal_pass_audit(
    root: str | Path = ".",
    *,
    packet_output_root: str = PAPER_FINAL_REPRESENTATION_PACKET_OUTPUT_ROOT,
    ignore_packets_outside_output_root: bool = False,
) -> dict[str, Any]:
    root_path = Path(root)
    discovered: dict[str, list[dict[str, Any]]] = {city: [] for city in REQUIRED_CITY_TRAFFIC}
    ignored_packets: list[dict[str, str]] = []
    for packet_path in sorted(root_path.glob(PAPER_REPRESENTATION_PACKET_GLOB)):
        rel_packet_path = _rel(packet_path, root_path)
        if ignore_packets_outside_output_root and not _is_under_packet_output_root(
            rel_packet_path,
            packet_output_root=packet_output_root,
        ):
            ignored_packets.append({"packet_path": rel_packet_path, "reason": "outside selected representation packet output root"})
            continue
        packet = _load_json(packet_path)
        try:
            source = validate_representation_packet(packet)
        except ValueError as exc:
            city = _paper_final_city_from_scope(packet)
            if city in discovered:
                gate = packet.get("formal_gate_decision") or {}
                discovered[city].append(
                    {
                        "packet_path": rel_packet_path,
                        "packet_hash": _sha256(packet_path),
                        "formal_representation_pass": packet.get("formal_representation_pass"),
                        "claim_mode": gate.get("claim_mode"),
                        "failed_reasons": list(gate.get("failed_reasons") or []),
                        "blockers": [str(exc), *_packet_blockers(packet)],
                    }
                )
                continue
            ignored_packets.append({"packet_path": rel_packet_path, "reason": str(exc)})
            continue
        city = source["city"]
        if city not in discovered:
            ignored_packets.append({"packet_path": rel_packet_path, "reason": f"unexpected city: {city}"})
            continue
        gate = packet.get("formal_gate_decision") or {}
        discovered[city].append(
            {
                "packet_path": rel_packet_path,
                "packet_hash": _sha256(packet_path),
                "formal_representation_pass": packet.get("formal_representation_pass"),
                "claim_mode": gate.get("claim_mode"),
                "failed_reasons": list(gate.get("failed_reasons") or []),
                "blockers": _packet_blockers(packet),
            }
        )

    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for city in REQUIRED_CITY_TRAFFIC:
        packets = discovered[city]
        if not packets:
            message = "missing paper-final representation packet"
            rows.append({"city": city, "status": "blocked", "packet_hashes": [], "packet_paths": [], "blockers": [message]})
            blockers.append(f"representation formal gate {city}: {message}")
            continue

        packet_hashes = sorted({row["packet_hash"] for row in packets})
        city_blockers: list[str] = []
        if len(packet_hashes) != 1:
            city_blockers.append(f"multiple paper-final packet hashes: {packet_hashes}")
        for packet_row in packets:
            if not _is_under_packet_output_root(str(packet_row["packet_path"]), packet_output_root=packet_output_root):
                city_blockers.append(
                    f"packet outside paper-final representation packet output root: {packet_row['packet_path']}"
                )
            city_blockers.extend(packet_row["blockers"])

        unique_blockers = sorted(set(city_blockers))
        status = "pass" if not unique_blockers else "blocked"
        rows.append(
            {
                "city": city,
                "status": status,
                "packet_hashes": packet_hashes,
                "packet_paths": [row["packet_path"] for row in packets],
                "formal_representation_pass": all(row["formal_representation_pass"] is True for row in packets),
                "claim_modes": sorted({str(row["claim_mode"]) for row in packets}),
                "failed_reasons": sorted({str(reason) for row in packets for reason in row["failed_reasons"]}),
                "blockers": unique_blockers,
            }
        )
        blockers.extend(f"representation formal gate {city}: {blocker}" for blocker in unique_blockers)

    return {
        "packet_type": "paper_representation_formal_pass_audit",
        "status": "pass" if not blockers else "blocked",
        "packet_output_root": packet_output_root,
        "rows": rows,
        "blockers": blockers,
        "ignored_packets": ignored_packets,
        "executes_now": False,
        "reads_final_traffic_result_values": False,
        "paper_result_claim": False,
    }


def representation_formal_pass_blockers(audit: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    packet_output_root = str(audit.get("packet_output_root") or PAPER_FINAL_REPRESENTATION_PACKET_OUTPUT_ROOT)
    if audit.get("packet_type") != "paper_representation_formal_pass_audit":
        blockers.append("representation formal-pass audit has wrong packet_type")
    if audit.get("executes_now") is not False:
        blockers.append("representation formal-pass audit must be non-executing")
    if audit.get("reads_final_traffic_result_values") is not False:
        blockers.append("representation formal-pass audit must not read final traffic result values")
    if audit.get("paper_result_claim") is not False:
        blockers.append("representation formal-pass audit must not make paper claims")

    rows = audit.get("rows")
    if not isinstance(rows, list):
        blockers.append("representation formal-pass audit rows must be a list")
        rows = []
    rows_by_city = {str(row.get("city")): row for row in rows if isinstance(row, dict)}
    for city in REQUIRED_CITY_TRAFFIC:
        row = rows_by_city.get(city)
        if row is None:
            blockers.append(f"missing required city in representation formal-pass audit: {city}")
            continue
        if row.get("status") != "pass":
            blockers.append(f"representation formal-pass row not pass for {city}")
        packet_hashes = list(row.get("packet_hashes") or [])
        if len(packet_hashes) != 1 or len(str(packet_hashes[0] if packet_hashes else "")) != 64:
            blockers.append(f"representation formal-pass row must contain one 64-char packet hash for {city}")
        packet_paths = list(row.get("packet_paths") or [])
        if not packet_paths:
            blockers.append(f"representation formal-pass row missing packet path for {city}")
        for packet_path in packet_paths:
            if not _is_under_packet_output_root(str(packet_path), packet_output_root=packet_output_root):
                blockers.append(f"packet outside paper-final representation packet output root for {city}: {packet_path}")
        if row.get("formal_representation_pass") is not True:
            blockers.append(f"formal_representation_pass must be true for {city}")
        if "diagnostics_only" in set(row.get("claim_modes") or []):
            blockers.append(f"claim_mode diagnostics_only is not allowed for {city}")
        row_blockers = list(row.get("blockers") or [])
        blockers.extend(f"representation formal-pass row blocker for {city}: {blocker}" for blocker in row_blockers)

    blockers.extend(str(blocker) for blocker in (audit.get("blockers") or []))
    return sorted(set(blockers))


def build_representation_scope_limitation(
    *,
    approval_phrase: str | None = None,
    paper_claim_limitation: str | None = None,
) -> dict[str, Any]:
    if approval_phrase == REQUIRED_REPRESENTATION_DIAGNOSTIC_LIMITATION_APPROVAL:
        return validate_representation_scope_limitation(
            {
                "metric_family": "representation",
                "status": "diagnostic_limitation_by_reviewer",
                "approval_phrase": approval_phrase,
                "paper_claim_limitation": paper_claim_limitation,
                "executes_now": False,
                "reads_final_traffic_result_values": False,
                "paper_result_claim": False,
            }
        )
    return validate_representation_scope_limitation(
        {
            "metric_family": "representation",
            "status": "missing_blocker",
            "approval_phrase": approval_phrase or "",
            "blocker": "representation diagnostic limitation requires exact reviewer approval phrase",
            "executes_now": False,
            "reads_final_traffic_result_values": False,
            "paper_result_claim": False,
        }
    )


def validate_representation_scope_limitation(row: dict[str, Any]) -> dict[str, Any]:
    status = row.get("status")
    if status not in ALLOWED_REPRESENTATION_SCOPE_LIMITATION_STATUSES:
        raise ValueError(f"unknown representation scope limitation status: {status}")
    if row.get("metric_family") not in {None, "representation"}:
        raise ValueError("representation scope limitation must target representation metric family")
    if row.get("executes_now") is not False:
        raise ValueError("representation scope limitation must be non-executing")
    if row.get("reads_final_traffic_result_values") is not False:
        raise ValueError("representation scope limitation must not read final traffic result values")
    if row.get("paper_result_claim") is not False:
        raise ValueError("representation scope limitation must not make paper result claims")
    if status == "diagnostic_limitation_by_reviewer":
        if row.get("approval_phrase") != REQUIRED_REPRESENTATION_DIAGNOSTIC_LIMITATION_APPROVAL:
            raise ValueError("representation diagnostic limitation requires exact reviewer approval phrase")
        if not row.get("paper_claim_limitation"):
            raise ValueError("representation diagnostic limitation must record paper claim limitation")
    if status == "missing_blocker" and not row.get("blocker"):
        raise ValueError("missing representation scope limitation must include blocker")
    return dict(row)
