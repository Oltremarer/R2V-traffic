#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.common.io import write_json
from pareto.common.scenario import resolve_scenario
from pareto.r2v.generative_scorer import load_generative_score_artifact
from pareto.r2v.result_aggregation import PERFORMANCE_METRIC_ALIASES, PERFORMANCE_METRICS


PAPER_DIFFUSION_REQUIRED_PROVENANCE = ("model_checkpoint", "config_hash", "normalization_id")
PAPER_DIFFUSION_PROXY_ADAPTERS = {"traffic_feature_density_proxy", "traffic_feature_center_distance_proxy"}
COMPLETED_PERFORMANCE_STATUSES = {"DONE", "COMPLETED", "COMPLETE", "SUCCESS", "FINISHED"}
REPAIR_METADATA_POLICIES = {"require_metadata", "metadata_or_proxy"}
STRICT_REPAIR_METADATA_POLICY = "require_metadata"


def check_r2v_traffic_readiness(
    *,
    root: str | Path = ROOT,
    scenario: str = "jinan",
    traffic_file: str | None = None,
    transition_glob: str | None = None,
    seeds: Iterable[int] = (0,),
    diffusion_artifacts: dict[int, str | Path] | None = None,
    require_cityflow_data: bool = True,
    require_diffusion_artifacts: bool = False,
    require_paper_claim_eligible_diffusion: bool = False,
    performance_paths: Iterable[str | Path] = (),
    require_performance_metrics: bool = False,
    expected_performance_methods: Iterable[str] = (),
    expected_performance_seeds: Iterable[int] = (),
    require_completed_performance_status: bool = False,
    repair_metadata_policy: str | None = None,
    require_strict_repair_metadata_policy: bool = False,
) -> dict[str, Any]:
    root_path = Path(root)
    seed_tuple = tuple(int(seed) for seed in seeds)
    if not seed_tuple:
        raise ValueError("at least one seed is required")
    checks: list[dict[str, Any]] = []
    meta = resolve_scenario(scenario)
    selected_traffic = traffic_file or meta["default_traffic_file"]
    if selected_traffic not in meta["traffic_files"]:
        raise ValueError(f"traffic_file {selected_traffic!r} is not registered for {scenario!r}")
    if repair_metadata_policy is not None and repair_metadata_policy not in REPAIR_METADATA_POLICIES:
        raise ValueError(f"unknown repair_metadata_policy: {repair_metadata_policy!r}")

    if repair_metadata_policy is not None or require_strict_repair_metadata_policy:
        checks.append(
            _repair_metadata_policy_check(
                repair_metadata_policy or "unspecified",
                require_strict=require_strict_repair_metadata_policy,
            )
        )

    if require_cityflow_data:
        data_dir = root_path / "data" / meta["template"] / meta["roadnet"]
        checks.append(_path_check("traffic_file", data_dir / selected_traffic))
        checks.append(_path_check("roadnet_file", data_dir / f"roadnet_{meta['roadnet']}.json"))

    transition_ids_by_seed: dict[int, set[str]] = {}
    if transition_glob is not None:
        for seed in seed_tuple:
            pattern = str(root_path / transition_glob.format(seed=seed))
            matches = sorted(glob.glob(pattern))
            check, transition_ids = _transition_input_check(pattern, matches, seed=seed)
            checks.append(check)
            if transition_ids is not None:
                transition_ids_by_seed[seed] = transition_ids

    artifact_map = diffusion_artifacts or {}
    if require_diffusion_artifacts:
        for seed in seed_tuple:
            raw_path = artifact_map.get(seed)
            if raw_path is None:
                checks.append(
                    {
                        "name": "diffusion_score_artifact",
                        "seed": seed,
                        "status": "fail",
                        "path": None,
                        "message": "missing diffusion artifact mapping for seed",
                    }
                )
            else:
                checks.append(
                    _diffusion_artifact_check(
                        _rooted(root_path, raw_path),
                        seed=seed,
                        required_transition_ids=transition_ids_by_seed.get(seed),
                        require_paper_claim_eligible=require_paper_claim_eligible_diffusion,
                    )
                )
    else:
        for seed, raw_path in sorted(artifact_map.items()):
            checks.append(
                _diffusion_artifact_check(
                    _rooted(root_path, raw_path),
                    seed=seed,
                    required_transition_ids=transition_ids_by_seed.get(seed),
                    require_paper_claim_eligible=require_paper_claim_eligible_diffusion,
                )
            )

    performance_path_tuple = tuple(performance_paths)
    expected_method_tuple = tuple(str(method) for method in expected_performance_methods)
    expected_seed_tuple = tuple(int(seed) for seed in expected_performance_seeds)
    if (
        performance_path_tuple
        or require_performance_metrics
        or expected_method_tuple
        or expected_seed_tuple
        or require_completed_performance_status
    ):
        checks.extend(
            _performance_checks(
                root_path,
                performance_path_tuple,
                required=require_performance_metrics,
                expected_methods=expected_method_tuple,
                expected_seeds=expected_seed_tuple,
                require_completed_status=require_completed_performance_status,
            )
        )

    failed = [check for check in checks if check["status"] == "fail"]
    return {
        "schema_version": "r2v-traffic-readiness-v1",
        "status": "BLOCKED" if failed else "READY",
        "scenario": scenario,
        "traffic_file": selected_traffic,
        "seeds": list(seed_tuple),
        "checks": checks,
        "failed_count": len(failed),
        "failed_checks": failed,
    }


def parse_seed_artifact(values: Iterable[str]) -> dict[int, str]:
    artifacts: dict[int, str] = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"diffusion artifact must use seed:path format: {value!r}")
        seed_text, path = value.split(":", 1)
        if not path:
            raise ValueError(f"diffusion artifact path is empty: {value!r}")
        artifacts[int(seed_text)] = path
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check R2V-Traffic smoke/main experiment readiness.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--scenario", default="jinan")
    parser.add_argument("--traffic_file", default=None)
    parser.add_argument("--transition_glob", default=None)
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--diffusion_artifact", action="append", default=[], help="Seed-specific artifact as seed:path.")
    parser.add_argument("--require_cityflow_data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require_diffusion_artifacts", action="store_true")
    parser.add_argument("--require_paper_claim_eligible_diffusion", action="store_true")
    parser.add_argument("--performance_path", action="append", default=[])
    parser.add_argument("--require_performance_metrics", action="store_true")
    parser.add_argument("--expected_performance_method", action="append", default=[])
    parser.add_argument("--expected_performance_seed", type=int, action="append", default=[])
    parser.add_argument("--require_completed_performance_status", action="store_true")
    parser.add_argument("--repair_metadata_policy", choices=tuple(sorted(REPAIR_METADATA_POLICIES)), default=None)
    parser.add_argument("--require_strict_repair_metadata_policy", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = check_r2v_traffic_readiness(
        root=args.root,
        scenario=args.scenario,
        traffic_file=args.traffic_file,
        transition_glob=args.transition_glob,
        seeds=args.seed or (0,),
        diffusion_artifacts=parse_seed_artifact(args.diffusion_artifact),
        require_cityflow_data=args.require_cityflow_data,
        require_diffusion_artifacts=args.require_diffusion_artifacts,
        require_paper_claim_eligible_diffusion=args.require_paper_claim_eligible_diffusion,
        performance_paths=args.performance_path,
        require_performance_metrics=args.require_performance_metrics,
        expected_performance_methods=args.expected_performance_method,
        expected_performance_seeds=args.expected_performance_seed,
        require_completed_performance_status=args.require_completed_performance_status,
        repair_metadata_policy=args.repair_metadata_policy,
        require_strict_repair_metadata_policy=args.require_strict_repair_metadata_policy,
    )
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "READY":
        raise SystemExit(2)


def _repair_metadata_policy_check(policy: str, *, require_strict: bool) -> dict[str, Any]:
    ok = not require_strict or policy == STRICT_REPAIR_METADATA_POLICY
    if ok:
        message = "repair metadata policy is valid for requested readiness scope"
    else:
        message = (
            "strict paper diffusion readiness requires repair_metadata_policy=require_metadata; "
            "metadata_or_proxy is smoke/integration only"
        )
    return {
        "name": "repair_metadata_policy",
        "status": "pass" if ok else "fail",
        "repair_metadata_policy": policy,
        "require_strict_repair_metadata_policy": bool(require_strict),
        "strict_repair_metadata_policy": STRICT_REPAIR_METADATA_POLICY,
        "message": message,
    }


def _path_check(name: str, path: Path, *, seed: int | None = None) -> dict[str, Any]:
    exists = path.is_file()
    check = {
        "name": name,
        "status": "pass" if exists else "fail",
        "path": str(path),
        "message": "file exists" if exists else "file missing",
    }
    if seed is not None:
        check["seed"] = seed
    return check


def _rooted(root: Path, path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else root / path_obj


def _transition_input_check(pattern: str, matches: list[str], *, seed: int) -> tuple[dict[str, Any], set[str] | None]:
    if not matches:
        return (
            {
                "name": "transition_inputs",
                "seed": seed,
                "status": "fail",
                "pattern": pattern,
                "match_count": 0,
                "paths": [],
                "message": "no transition inputs matched",
            },
            None,
        )
    try:
        ids = _transition_ids(matches)
    except ValueError as exc:
        return (
            {
                "name": "transition_inputs",
                "seed": seed,
                "status": "fail",
                "pattern": pattern,
                "match_count": len(matches),
                "paths": matches,
                "message": str(exc),
            },
            None,
        )
    return (
        {
            "name": "transition_inputs",
            "seed": seed,
            "status": "pass",
            "pattern": pattern,
            "match_count": len(matches),
            "transition_count": len(ids),
            "paths": matches,
            "message": "transition inputs found",
        },
        ids,
    )


def _transition_ids(paths: Iterable[str | Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        path_obj = Path(path)
        for idx, line in enumerate(path_obj.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"transition row must be a JSON object: {path_obj}:{idx}")
            transition_id = row.get("transition_id")
            if transition_id is None or str(transition_id) == "":
                raise ValueError(f"missing transition_id in transition row: {path_obj}:{idx}")
            ids.add(str(transition_id))
    if not ids:
        raise ValueError("transition inputs contain no transition rows")
    return ids


def _diffusion_artifact_check(
    path: Path,
    *,
    seed: int,
    required_transition_ids: set[str] | None,
    require_paper_claim_eligible: bool = False,
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "name": "diffusion_score_artifact",
            "seed": seed,
            "status": "fail",
            "path": str(path),
            "message": "file missing",
        }
    try:
        score_rows, summary = load_generative_score_artifact(path, backend="diffusion")
    except ValueError as exc:
        return {
            "name": "diffusion_score_artifact",
            "seed": seed,
            "status": "fail",
            "path": str(path),
            "message": str(exc),
        }
    missing: list[str] = []
    if required_transition_ids is not None:
        missing = sorted(required_transition_ids - set(score_rows))
    ineligible_ids = [
        transition_id
        for transition_id, row in sorted(score_rows.items())
        if row.source.get("paper_claim_eligible") is not True
    ]
    missing_provenance_ids: list[str] = []
    proxy_adapter_ids: list[str] = []
    if require_paper_claim_eligible:
        for transition_id, row in sorted(score_rows.items()):
            if any(_blank(row.source.get(field)) for field in PAPER_DIFFUSION_REQUIRED_PROVENANCE):
                missing_provenance_ids.append(transition_id)
            adapter = str(row.source.get("adapter") or "")
            if adapter in PAPER_DIFFUSION_PROXY_ADAPTERS:
                proxy_adapter_ids.append(transition_id)
    artifact_ok = (
        not missing
        and (
            not require_paper_claim_eligible
            or (not ineligible_ids and not missing_provenance_ids and not proxy_adapter_ids)
        )
    )
    if missing:
        message = "diffusion score artifact missing transition ids"
    elif require_paper_claim_eligible and ineligible_ids:
        message = "diffusion score artifact is not paper-claim eligible"
    elif require_paper_claim_eligible and missing_provenance_ids:
        message = "diffusion score artifact missing paper diffusion provenance"
    elif require_paper_claim_eligible and proxy_adapter_ids:
        message = "diffusion score artifact uses proxy adapter"
    else:
        message = "diffusion score artifact is valid"
    return {
        "name": "diffusion_score_artifact",
        "seed": seed,
        "status": "pass" if artifact_ok else "fail",
        "path": str(path),
        "score_count": len(score_rows),
        "backend": summary["backend"],
        "missing_transition_count": len(missing),
        "missing_transition_ids": missing[:10],
        "paper_claim_eligible": not ineligible_ids,
        "require_paper_claim_eligible": bool(require_paper_claim_eligible),
        "paper_claim_ineligible_count": len(ineligible_ids),
        "paper_claim_ineligible_transition_ids": ineligible_ids[:10],
        "paper_claim_required_provenance": list(PAPER_DIFFUSION_REQUIRED_PROVENANCE),
        "paper_claim_provenance_missing_count": len(missing_provenance_ids),
        "paper_claim_provenance_missing_transition_ids": missing_provenance_ids[:10],
        "paper_claim_proxy_adapter_count": len(proxy_adapter_ids),
        "paper_claim_proxy_adapter_transition_ids": proxy_adapter_ids[:10],
        "message": message,
    }


def _blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _performance_checks(
    root: Path,
    paths: tuple[str | Path, ...],
    *,
    required: bool,
    expected_methods: tuple[str, ...] = (),
    expected_seeds: tuple[int, ...] = (),
    require_completed_status: bool = False,
) -> list[dict[str, Any]]:
    if not paths:
        return [
            {
                "name": "performance_metrics",
                "status": "fail",
                "paths": [],
                "message": "performance metrics are required but no paths were provided",
            }
        ]
    checks: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = _rooted(root, raw_path)
        if not path.is_file():
            checks.append(
                {
                    "name": "performance_metrics",
                    "status": "fail",
                    "path": str(path),
                    "message": "performance metrics file missing",
                }
            )
            continue
        rows = _load_json_rows(path)
        all_rows.extend(rows)
        missing_rows: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            values = _canonical_metric_values(row)
            missing = [metric for metric in PERFORMANCE_METRICS if metric not in values]
            if missing:
                missing_rows.append({"row": idx, "missing": missing})
        checks.append(
            {
                "name": "performance_metrics",
                "status": "fail" if missing_rows else "pass",
                "path": str(path),
                "row_count": len(rows),
                "required_metrics": list(PERFORMANCE_METRICS),
                "missing_rows": missing_rows,
                "message": "all required performance metrics found"
                if not missing_rows
                else "performance rows missing required metrics",
            }
        )
    if expected_methods or expected_seeds or require_completed_status:
        checks.append(
            _performance_coverage_check(
                all_rows,
                expected_methods=expected_methods,
                expected_seeds=expected_seeds,
                require_completed_status=require_completed_status,
            )
        )
    return checks


def _performance_coverage_check(
    rows: list[dict[str, Any]],
    *,
    expected_methods: tuple[str, ...],
    expected_seeds: tuple[int, ...],
    require_completed_status: bool,
) -> dict[str, Any]:
    observed_pairs: set[tuple[str, int]] = set()
    duplicate_pairs: list[dict[str, Any]] = []
    unfinished_rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, int]] = set()
    for row in rows:
        method = str(row.get("method", ""))
        seed_value = row.get("seed")
        try:
            seed = int(seed_value)
        except (TypeError, ValueError):
            continue
        pair = (method, seed)
        observed_pairs.add(pair)
        if pair in seen_pairs:
            duplicate_pairs.append({"method": method, "seed": seed})
        seen_pairs.add(pair)
        if require_completed_status and not _completed_status(row.get("status")):
            unfinished_rows.append({"method": method, "seed": seed, "status": row.get("status")})

    missing_pairs: list[dict[str, Any]] = []
    if expected_methods and expected_seeds:
        for method in expected_methods:
            for seed in expected_seeds:
                if (method, seed) not in observed_pairs:
                    missing_pairs.append({"method": method, "seed": seed})
    elif expected_methods:
        observed_methods = {method for method, _seed in observed_pairs}
        missing_pairs = [{"method": method, "seed": None} for method in expected_methods if method not in observed_methods]
    elif expected_seeds:
        observed_seeds = {seed for _method, seed in observed_pairs}
        missing_pairs = [{"method": None, "seed": seed} for seed in expected_seeds if seed not in observed_seeds]

    failed = bool(missing_pairs or duplicate_pairs or unfinished_rows)
    expected_row_count = len(expected_methods) * len(expected_seeds) if expected_methods and expected_seeds else None
    return {
        "name": "performance_coverage",
        "status": "fail" if failed else "pass",
        "row_count": len(rows),
        "expected_methods": list(expected_methods),
        "expected_seeds": list(expected_seeds),
        "expected_row_count": expected_row_count,
        "observed_method_seed_pairs": [
            {"method": method, "seed": seed}
            for method, seed in sorted(observed_pairs)
        ],
        "missing_method_seed_pairs": missing_pairs,
        "duplicate_method_seed_pairs": duplicate_pairs,
        "require_completed_status": bool(require_completed_status),
        "completed_statuses": sorted(COMPLETED_PERFORMANCE_STATUSES),
        "unfinished_rows": unfinished_rows,
        "message": "performance method/seed coverage is complete"
        if not failed
        else "performance rows missing expected coverage or completed status",
    }


def _completed_status(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().upper() in COMPLETED_PERFORMANCE_STATUSES


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else [payload]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"performance metrics must be JSON objects: {path}")
    return rows


def _canonical_metric_values(row: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for canonical, aliases in PERFORMANCE_METRIC_ALIASES.items():
        for alias in aliases:
            if alias not in row:
                continue
            parsed = _finite(row[alias], label=alias)
            values[canonical] = parsed
            break
    return values


def _finite(value: Any, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid performance metric {label!r}: {value!r}") from None
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite performance metric {label!r}: {value!r}")
    return parsed


if __name__ == "__main__":
    main()
