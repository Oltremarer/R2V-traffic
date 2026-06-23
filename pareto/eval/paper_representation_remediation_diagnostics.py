from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pareto.eval.paper_representation_artifact_sources import validate_representation_packet
from pareto.eval.paper_representation_formal_pass_guard import PAPER_FINAL_REPRESENTATION_PACKET_OUTPUT_ROOT
from pareto.rl.paper_final_experiment_manifest import REQUIRED_CITY_TRAFFIC


DIAGNOSTIC_VERSION = "paper-representation-remediation-diagnostics-v1"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _threshold_margins(packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lower_checks = packet.get("bootstrap_lower_bound_checks") or {}
    margins: dict[str, dict[str, Any]] = {}
    for metric, check in (packet.get("threshold_checks") or {}).items():
        if not isinstance(check, dict):
            continue
        value = check.get("value")
        threshold = check.get("threshold")
        bootstrap = check.get("bootstrap") or {}
        low = bootstrap.get("low", (lower_checks.get(metric) or {}).get("low"))
        margins[metric] = {
            "value": value,
            "threshold": threshold,
            "pass": check.get("pass"),
            "point_margin": round(float(value) - float(threshold), 10) if value is not None and threshold is not None else None,
            "bootstrap_low": low,
            "bootstrap_low_margin": round(float(low) - float(threshold), 10) if low is not None and threshold is not None else None,
        }
    return margins


def _test_pair_counts(packet: dict[str, Any]) -> dict[str, Any]:
    test_split = ((packet.get("pair_coverage_check") or {}).get("splits") or {}).get("test", {})
    counts = dict(test_split.get("counts") or {})
    reversal_by_template_pair = dict(test_split.get("reversal_by_template_pair") or {})
    objective_counts = dict(test_split.get("objective_counts") or {})
    if "reversal_pairs" not in counts and reversal_by_template_pair:
        counts["reversal_pairs"] = sum(int(value) for value in reversal_by_template_pair.values())
    if "objective_pairs" not in counts and objective_counts:
        counts["objective_pairs"] = sum(int(value) for value in objective_counts.values())
    return counts


def _diagnostic_row(city: str, packet_path: Path, root: Path) -> dict[str, Any]:
    packet = _load_json(packet_path)
    validate_representation_packet(packet)
    gate = packet.get("formal_gate_decision") or {}
    return {
        "city": city,
        "packet_path": packet_path.relative_to(root).as_posix(),
        "formal_representation_pass": packet.get("formal_representation_pass"),
        "claim_mode": gate.get("claim_mode"),
        "failed_reasons": list(gate.get("failed_reasons") or []),
        "threshold_margins": _threshold_margins(packet),
        "pair_counts": _test_pair_counts(packet),
        "pair_coverage_pass": (packet.get("pair_coverage_check") or {}).get("pass"),
        "split_leakage_pass": (packet.get("split_leakage_check") or {}).get("pass"),
        "train_only_normalizer_pass": (packet.get("train_only_normalizer_check") or {}).get("pass"),
        "bootstrap_pair_count_consistency_pass": (packet.get("bootstrap_pair_count_consistency_check") or {}).get("pass"),
        "objective_sanity_pass": (packet.get("objective_sanity_status") or {}).get("pass"),
        "dominance_audit_summary": packet.get("dominance_audit_summary") or {},
        "failure_interpretation": "formal_threshold_failure" if gate.get("failed_reasons") else "no_formal_failure_reason",
    }


def build_representation_remediation_diagnostics(root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    output_root = root_path / PAPER_FINAL_REPRESENTATION_PACKET_OUTPUT_ROOT
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for city in REQUIRED_CITY_TRAFFIC:
        city_packets = sorted((output_root / city).glob("*/representation_formal_gate_packet.json"))
        if not city_packets:
            blockers.append(f"{city}: paper-final representation packet missing")
            continue
        hashes = {path.read_bytes() for path in city_packets}
        if len(hashes) > 1:
            blockers.append(f"{city}: multiple distinct paper-final representation packets")
        rows.append(_diagnostic_row(city, city_packets[0], root_path))

    return {
        "packet_type": DIAGNOSTIC_VERSION,
        "status": "missing_blocker" if blockers else "diagnostic_only",
        "rows": rows,
        "blockers": blockers,
        "executes_training_now": False,
        "regenerates_evidence": False,
        "overwrites_representation_packets": False,
        "writes_records_paper_final": False,
        "reads_final_traffic_result_values": False,
        "generates_ranking": False,
        "generates_plots": False,
        "generates_paper_tables": False,
        "paper_result_claim": False,
    }


def remediation_diagnostic_blockers(diagnostics: dict[str, Any]) -> list[str]:
    blockers = list(diagnostics.get("blockers") or [])
    for key in (
        "executes_training_now",
        "regenerates_evidence",
        "overwrites_representation_packets",
        "writes_records_paper_final",
        "reads_final_traffic_result_values",
        "generates_ranking",
        "generates_plots",
        "generates_paper_tables",
        "paper_result_claim",
    ):
        if diagnostics.get(key) is not False:
            blockers.append(f"{key} must be false")
    return blockers


def write_representation_remediation_diagnostics(diagnostics: dict[str, Any], out_dir: str | Path) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "representation_remediation_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Representation Remediation Diagnostics",
        "",
        f"Status: `{diagnostics.get('status')}`",
        "",
        "This diagnostic is non-executing. It does not regenerate evidence, train models, read final traffic results, rank methods, plot, create paper tables, or make paper claims.",
        "",
        "## City Failures",
        "",
        "| City | Formal Pass | Claim Mode | Failed Reasons | Key Margins |",
        "|---|---|---|---|---|",
    ]
    for row in diagnostics.get("rows") or []:
        margins = row.get("threshold_margins") or {}
        failed = [key for key, value in margins.items() if value.get("pass") is False]
        margin_text = ", ".join(f"{key}: {margins[key].get('point_margin')}" for key in failed)
        lines.append(
            f"| `{row.get('city')}` | `{row.get('formal_representation_pass')}` | `{row.get('claim_mode')}` | `{row.get('failed_reasons')}` | `{margin_text}` |"
        )
    (out_path / "representation_remediation_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
