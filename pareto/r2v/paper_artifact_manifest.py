from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from pareto.r2v.experiment_readiness import (
    PAPER_DIFFUSION_PROXY_ADAPTERS,
    PAPER_DIFFUSION_REQUIRED_PROVENANCE,
)
from pareto.r2v.generative_scorer import load_generative_score_artifact
from pareto.r2v.result_aggregation import PERFORMANCE_METRIC_ALIASES, PERFORMANCE_METRICS
from pareto.r2v.traffic_artifact_schema import validate_r2v_traffic_artifact


ALLOWED_ARTIFACT_TYPES = frozenset(
    {
        "aggregation",
        "diffusion_score",
        "experiment_plan",
        "integrity",
        "performance",
        "readiness",
        "weighted_transitions",
    }
)


def build_paper_artifact_manifest(
    artifacts: Iterable[dict[str, Any]],
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    root_path = Path(root) if root is not None else None
    entries = [_manifest_entry(artifact, root=root_path) for artifact in artifacts]
    _attach_aggregation_bundle_link_status(entries)
    _attach_weighted_score_artifact_link_status(entries)
    failed = [entry for entry in entries if entry["status"] != "present"]
    type_counts: dict[str, int] = {}
    for entry in entries:
        artifact_type = str(entry["artifact_type"])
        type_counts[artifact_type] = type_counts.get(artifact_type, 0) + 1
    return {
        "schema_version": "r2v-traffic-paper-artifact-manifest-v1",
        "status": "BLOCKED" if failed else "READY",
        "artifact_type_counts": dict(sorted(type_counts.items())),
        "entry_count": len(entries),
        "failed_count": len(failed),
        "entries": entries,
        "failed_entries": failed,
        "claim_boundary": "performance artifacts and integrity/status artifacts are tracked as separate artifact types",
    }


def parse_artifact_spec(value: str) -> tuple[str, str, str]:
    parts = value.split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError("artifact spec must use artifact_type:name:path")
    return parts[0], parts[1], parts[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an R2V-Traffic paper artifact manifest.")
    parser.add_argument("--root", default=None)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Artifact spec as artifact_type:name:path. Repeatable.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_paper_artifact_manifest(
        [
            {"artifact_type": artifact_type, "name": name, "path": path}
            for artifact_type, name, path in (parse_artifact_spec(value) for value in args.artifact)
        ],
        root=args.root,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    output_path.write_text(output_text, encoding="utf-8")
    print(output_text, end="")
    return 2 if manifest["status"] != "READY" else 0


def _manifest_entry(artifact: dict[str, Any], *, root: Path | None) -> dict[str, Any]:
    artifact_type = str(artifact.get("artifact_type", ""))
    if artifact_type not in ALLOWED_ARTIFACT_TYPES:
        raise ValueError(f"unknown artifact_type: {artifact_type!r}")
    name = str(artifact.get("name", ""))
    if not name:
        raise ValueError("artifact name is required")
    raw_path = artifact.get("path")
    if raw_path is None:
        raise ValueError(f"artifact path is required for {name!r}")
    path = Path(raw_path)
    resolved_path = path if path.is_absolute() or root is None else root / path
    entry: dict[str, Any] = {
        "artifact_type": artifact_type,
        "name": name,
        "path": str(path),
        "resolved_path": str(resolved_path),
    }
    if not resolved_path.is_file():
        entry.update(
            {
                "status": "missing",
                "sha256": None,
                "size_bytes": None,
                "line_count": None,
                "json_format": None,
                "schema_version": None,
            }
        )
        return entry
    entry.update(
        {
            "status": "present",
            "sha256": _sha256_file(resolved_path),
            "size_bytes": resolved_path.stat().st_size,
            "line_count": _line_count(resolved_path),
        }
    )
    entry.update(_json_metadata(resolved_path))
    if artifact_type == "performance":
        entry.update(_performance_content_metadata(resolved_path))
    if artifact_type == "diffusion_score":
        entry.update(_diffusion_score_content_metadata(resolved_path))
    if artifact_type == "weighted_transitions":
        entry.update(_weighted_transition_content_metadata(resolved_path))
    if artifact_type == "readiness":
        entry.update(_readiness_content_metadata(resolved_path))
    if artifact_type == "aggregation":
        entry.update(_aggregation_content_metadata(resolved_path))
    if artifact_type == "integrity":
        entry.update(_integrity_content_metadata(resolved_path))
    return entry


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    if not text:
        return 0
    return len(text.splitlines())


def _json_metadata(path: Path) -> dict[str, Any]:
    if path.suffix == ".jsonl":
        return {"json_format": "jsonl", "schema_version": None}
    if path.suffix != ".json":
        return {"json_format": None, "schema_version": None}
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version") if isinstance(payload, dict) else None
    return {"json_format": "json", "schema_version": schema_version}


def _performance_content_metadata(path: Path) -> dict[str, Any]:
    metric_count = 0
    row_count = 0
    observed_metrics: set[str] = set()
    for row in _load_json_rows(path):
        row_count += 1
        for canonical, aliases in PERFORMANCE_METRIC_ALIASES.items():
            if any(alias in row for alias in aliases):
                observed_metrics.add(canonical)
                metric_count += 1
    if metric_count <= 0:
        return {
            "status": "invalid_content",
            "performance_row_count": row_count,
            "performance_metric_count": 0,
            "performance_metrics": [],
            "performance_missing_metrics": list(PERFORMANCE_METRICS),
            "message": "no performance metrics found in artifact labeled as performance",
        }
    missing_metrics = [name for name in PERFORMANCE_METRICS if name not in observed_metrics]
    if missing_metrics:
        return {
            "status": "invalid_content",
            "performance_row_count": row_count,
            "performance_metric_count": metric_count,
            "performance_metrics": sorted(observed_metrics),
            "performance_missing_metrics": missing_metrics,
            "message": "performance artifact missing required traffic metrics",
        }
    return {
        "performance_row_count": row_count,
        "performance_metric_count": metric_count,
        "performance_metrics": sorted(observed_metrics),
        "performance_missing_metrics": [],
        "message": "performance artifact contains recognized traffic metrics",
    }


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"unsupported JSON payload in {path}")


def _diffusion_score_content_metadata(path: Path) -> dict[str, Any]:
    try:
        load_generative_score_artifact(path, backend="diffusion")
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            "status": "invalid_content",
            "diffusion_score_row_count": 0,
            "message": str(exc),
        }
    rows = _load_json_rows(path)
    ineligible_ids: list[str] = []
    missing_provenance_ids: list[str] = []
    proxy_adapter_ids: list[str] = []
    for idx, row in enumerate(rows):
        transition_id = str(row.get("transition_id", f"row_{idx}"))
        if row.get("paper_claim_eligible") is not True:
            ineligible_ids.append(transition_id)
        if any(_blank(row.get(field)) for field in PAPER_DIFFUSION_REQUIRED_PROVENANCE):
            missing_provenance_ids.append(transition_id)
        adapter = str(row.get("adapter", ""))
        if adapter in PAPER_DIFFUSION_PROXY_ADAPTERS:
            proxy_adapter_ids.append(transition_id)
    if ineligible_ids or missing_provenance_ids or proxy_adapter_ids:
        if ineligible_ids:
            message = "diffusion score artifact is not paper-claim eligible"
        elif missing_provenance_ids:
            message = "diffusion score artifact missing paper diffusion provenance"
        else:
            message = "diffusion score artifact uses proxy adapter"
        return {
            "status": "invalid_content",
            "diffusion_score_row_count": len(rows),
            "paper_claim_eligible": not ineligible_ids,
            "paper_claim_ineligible_count": len(ineligible_ids),
            "paper_claim_ineligible_transition_ids": ineligible_ids[:10],
            "paper_claim_required_provenance": list(PAPER_DIFFUSION_REQUIRED_PROVENANCE),
            "paper_claim_provenance_missing_count": len(missing_provenance_ids),
            "paper_claim_provenance_missing_transition_ids": missing_provenance_ids[:10],
            "paper_claim_proxy_adapter_count": len(proxy_adapter_ids),
            "paper_claim_proxy_adapter_transition_ids": proxy_adapter_ids[:10],
            "message": message,
        }
    return {
        "diffusion_score_row_count": len(rows),
        "paper_claim_eligible": True,
        "paper_claim_required_provenance": list(PAPER_DIFFUSION_REQUIRED_PROVENANCE),
        "paper_claim_ineligible_count": 0,
        "paper_claim_provenance_missing_count": 0,
        "paper_claim_proxy_adapter_count": 0,
        "message": "diffusion score artifact is paper-claim eligible with required provenance",
    }


def _weighted_transition_content_metadata(path: Path) -> dict[str, Any]:
    try:
        summary = validate_r2v_traffic_artifact(_load_json_rows(path))
    except ValueError as exc:
        return {
            "status": "invalid_content",
            "message": str(exc),
        }
    return {
        "weighted_transition_schema_version": summary["schema_version"],
        "weighted_transition_row_count": summary["row_count"],
        "weighted_transition_admitted_count": summary["admitted_count"],
        "weighted_transition_gate_counts": summary["gate_counts"],
        "weighted_transition_weight_min": summary["weight_min"],
        "weighted_transition_weight_max": summary["weight_max"],
        "weighted_transition_weight_mean": summary["weight_mean"],
        "weighted_transition_gate_variants": summary["gate_variants"],
        "weighted_transition_generative_backends": summary["generative_backends"],
        "weighted_transition_admission_modes": summary["admission_modes"],
        "weighted_transition_row_roles": summary["row_roles"],
        "weighted_transition_score_artifact_paths": summary["score_artifact_paths"],
        "message": "weighted transition artifact satisfies R2V-Traffic schema",
    }


def _readiness_content_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "status": "invalid_content",
            "message": f"readiness artifact is not valid JSON: {exc}",
        }
    if not isinstance(payload, dict):
        return {
            "status": "invalid_content",
            "message": "readiness artifact must be a JSON object",
        }
    readiness_status = str(payload.get("status", ""))
    failed_count = int(payload.get("failed_count", 0) or 0)
    check_count = len(payload.get("checks") or [])
    metadata = {
        "readiness_status": readiness_status,
        "readiness_failed_count": failed_count,
        "readiness_check_count": check_count,
    }
    if readiness_status != "READY" or failed_count != 0:
        metadata.update(
            {
                "status": "invalid_content",
                "message": "readiness artifact is not READY",
            }
        )
    else:
        metadata["message"] = "readiness artifact is READY"
    return metadata


def _aggregation_content_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "status": "invalid_content",
            "message": f"result aggregation artifact is not valid JSON: {exc}",
        }
    if not isinstance(payload, dict):
        return {
            "status": "invalid_content",
            "message": "result aggregation artifact must be a JSON object",
        }
    schema_version = str(payload.get("schema_version", ""))
    if schema_version != "r2v-traffic-result-aggregation-v1":
        return {
            "status": "invalid_content",
            "aggregation_schema_version": schema_version,
            "message": "result aggregation schema is missing or unsupported",
        }
    performance = payload.get("performance")
    if not isinstance(performance, dict):
        return {
            "status": "invalid_content",
            "aggregation_schema_version": schema_version,
            "message": "result aggregation missing performance section",
        }
    metrics = sorted(str(name) for name in dict(performance.get("metrics") or {}))
    missing_metrics = [name for name in PERFORMANCE_METRICS if name not in metrics]
    row_count = int(performance.get("row_count", 0) or 0)
    metric_value_count = int(performance.get("metric_value_count", 0) or 0)
    by_method = performance.get("by_method")
    by_method_count = len(by_method) if isinstance(by_method, dict) else 0
    input_metadata = _aggregation_input_artifact_metadata(payload)
    metadata = {
        "aggregation_schema_version": schema_version,
        "aggregation_metrics": metrics,
        "aggregation_missing_metrics": missing_metrics,
        "aggregation_performance_row_count": row_count,
        "aggregation_metric_value_count": metric_value_count,
        "aggregation_method_count": by_method_count,
        **input_metadata,
    }
    if missing_metrics:
        metadata.update(
            {
                "status": "invalid_content",
                "message": "result aggregation missing required traffic metrics",
            }
        )
    elif row_count <= 0 or metric_value_count <= 0 or by_method_count <= 0:
        metadata.update(
            {
                "status": "invalid_content",
                "message": "result aggregation contains no performance rows",
            }
        )
    elif input_metadata["aggregation_missing_input_artifact_roles"] or input_metadata["aggregation_invalid_input_artifact_count"]:
        metadata.update(
            {
                "status": "invalid_content",
                "message": "result aggregation missing required input artifact hashes",
            }
        )
    else:
        metadata["message"] = "result aggregation artifact contains required traffic metrics"
    return metadata


def _aggregation_input_artifact_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    input_artifacts = payload.get("input_artifacts")
    counts = {"integrity": 0, "performance": 0}
    hashes = {"integrity": [], "performance": []}
    hash_count = 0
    missing_roles: list[str] = []
    invalid_count = 0
    if not isinstance(input_artifacts, dict):
        missing_roles.extend(["integrity", "performance"])
        return {
            "aggregation_input_artifact_counts": counts,
            "aggregation_input_artifact_hashes": hashes,
            "aggregation_input_artifact_hash_count": 0,
            "aggregation_missing_input_artifact_roles": missing_roles,
            "aggregation_invalid_input_artifact_count": 0,
        }
    for role in ("integrity", "performance"):
        entries = input_artifacts.get(role)
        if not isinstance(entries, list) or not entries:
            missing_roles.append(role)
            continue
        counts[role] = len(entries)
        for entry in entries:
            if not _valid_input_artifact_entry(entry):
                invalid_count += 1
            else:
                hashes[role].append(str(entry["sha256"]))
                hash_count += 1
    return {
        "aggregation_input_artifact_counts": counts,
        "aggregation_input_artifact_hashes": hashes,
        "aggregation_input_artifact_hash_count": hash_count,
        "aggregation_missing_input_artifact_roles": missing_roles,
        "aggregation_invalid_input_artifact_count": invalid_count,
    }


def _attach_aggregation_bundle_link_status(entries: list[dict[str, Any]]) -> None:
    bundled_hashes: dict[str, set[str]] = {"integrity": set(), "performance": set()}
    for entry in entries:
        artifact_type = str(entry.get("artifact_type", ""))
        if artifact_type in bundled_hashes and entry.get("status") == "present":
            sha256 = entry.get("sha256")
            if isinstance(sha256, str):
                bundled_hashes[artifact_type].add(sha256)
    for entry in entries:
        if entry.get("artifact_type") != "aggregation" or entry.get("status") != "present":
            continue
        input_hashes = entry.get("aggregation_input_artifact_hashes")
        if not isinstance(input_hashes, dict):
            continue
        unmatched: dict[str, list[str]] = {}
        for role in ("integrity", "performance"):
            hashes = [str(value) for value in input_hashes.get(role) or []]
            missing = sorted(value for value in hashes if value not in bundled_hashes[role])
            if missing:
                unmatched[role] = missing
        if unmatched:
            entry.update(
                {
                    "status": "invalid_content",
                    "aggregation_unmatched_input_artifact_hashes": unmatched,
                    "message": "result aggregation input artifact hashes do not match bundled artifacts",
                }
            )


def _attach_weighted_score_artifact_link_status(entries: list[dict[str, Any]]) -> None:
    diffusion_score_paths: set[str] = set()
    for entry in entries:
        if entry.get("artifact_type") != "diffusion_score" or entry.get("status") != "present":
            continue
        for key in ("path", "resolved_path"):
            value = entry.get(key)
            if isinstance(value, str) and value:
                diffusion_score_paths.add(value)
        resolved_path = entry.get("resolved_path")
        if isinstance(resolved_path, str) and resolved_path:
            try:
                diffusion_score_paths.add(str(Path(resolved_path).resolve()))
            except OSError:
                pass

    for entry in entries:
        if entry.get("artifact_type") != "weighted_transitions" or entry.get("status") != "present":
            continue
        backends = set(str(value) for value in entry.get("weighted_transition_generative_backends") or [])
        if "diffusion" not in backends:
            continue
        paths = [str(value) for value in entry.get("weighted_transition_score_artifact_paths") or [] if str(value)]
        if not paths:
            entry.update(
                {
                    "status": "invalid_content",
                    "message": "diffusion weighted transitions must record source score artifact path",
                }
            )
            continue
        unmatched = sorted(path for path in paths if path not in diffusion_score_paths and _resolved_path_string(path) not in diffusion_score_paths)
        if unmatched:
            entry.update(
                {
                    "status": "invalid_content",
                    "weighted_transition_unmatched_score_artifact_paths": unmatched,
                    "message": "diffusion weighted transitions do not match bundled diffusion score artifacts",
                }
            )


def _resolved_path_string(path: str) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return path


def _valid_input_artifact_entry(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    if _blank(entry.get("path")) or not _is_sha256_hex(entry.get("sha256")):
        return False
    try:
        size_bytes = int(entry.get("size_bytes"))
        line_count = int(entry.get("line_count"))
    except (TypeError, ValueError):
        return False
    return size_bytes >= 0 and line_count >= 0


def _integrity_content_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "status": "invalid_content",
            "message": f"R2V integrity artifact is not valid JSON: {exc}",
        }
    if not isinstance(payload, dict):
        return {
            "status": "invalid_content",
            "message": "R2V integrity artifact must be a JSON object",
        }
    schema_version = str(payload.get("schema_version", ""))
    if schema_version != "r2v-tsc-candidate-summary-v1":
        return {
            "status": "invalid_content",
            "integrity_schema_version": schema_version,
            "message": "R2V integrity artifact candidate summary schema is missing or unsupported",
        }
    try:
        candidate_count = _non_negative_int(payload.get("candidate_count"), "candidate_count")
        admitted_count = _non_negative_int(payload.get("admitted_count"), "admitted_count")
    except ValueError as exc:
        return {
            "status": "invalid_content",
            "integrity_schema_version": schema_version,
            "message": str(exc),
        }
    gate_counts = payload.get("gate_counts")
    if not isinstance(gate_counts, dict):
        return {
            "status": "invalid_content",
            "integrity_schema_version": schema_version,
            "integrity_candidate_count": candidate_count,
            "integrity_admitted_count": admitted_count,
            "message": "R2V integrity artifact missing gate_counts",
        }
    required_gates = ("rare", "value", "support", "safety")
    missing_gates = [name for name in required_gates if name not in gate_counts]
    if missing_gates:
        return {
            "status": "invalid_content",
            "integrity_schema_version": schema_version,
            "integrity_candidate_count": candidate_count,
            "integrity_admitted_count": admitted_count,
            "integrity_missing_gate_counts": missing_gates,
            "message": "R2V integrity artifact missing required gate counts",
        }
    try:
        parsed_gate_counts = {
            name: _non_negative_int(gate_counts.get(name), f"gate_counts.{name}")
            for name in required_gates
        }
    except ValueError as exc:
        return {
            "status": "invalid_content",
            "integrity_schema_version": schema_version,
            "integrity_candidate_count": candidate_count,
            "integrity_admitted_count": admitted_count,
            "message": str(exc),
        }
    if admitted_count > candidate_count:
        return {
            "status": "invalid_content",
            "integrity_schema_version": schema_version,
            "integrity_candidate_count": candidate_count,
            "integrity_admitted_count": admitted_count,
            "integrity_gate_counts": parsed_gate_counts,
            "message": "R2V integrity artifact admitted_count exceeds candidate_count",
        }
    return {
        "integrity_schema_version": schema_version,
        "integrity_candidate_count": candidate_count,
        "integrity_admitted_count": admitted_count,
        "integrity_gate_counts": parsed_gate_counts,
        "message": "R2V integrity artifact contains candidate admission summary",
    }


def _non_negative_int(value: Any, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"R2V integrity artifact invalid {label}: {value!r}") from None
    if parsed < 0:
        raise ValueError(f"R2V integrity artifact invalid {label}: {value!r}")
    return parsed


def _blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _is_sha256_hex(value: Any) -> bool:
    text = str(value)
    if len(text) != 64:
        return False
    return all(char in "0123456789abcdef" for char in text)


if __name__ == "__main__":
    raise SystemExit(main())
