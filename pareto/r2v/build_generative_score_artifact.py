#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.common.io import append_jsonl, write_json
from pareto.r2v.artifact_validation import validate_unique_transition_ids


SUPPORTED_BACKENDS = {"diffusion"}
SUPPORTED_ADAPTERS = {"traffic_feature_density_proxy", "traffic_feature_center_distance_proxy"}


def build_score_artifact_from_files(
    *,
    transitions: Iterable[str | Path],
    output: str | Path,
    summary_output: str | Path,
    backend: str = "diffusion",
    adapter: str = "traffic_feature_density_proxy",
    density_neighbors: int = 1,
    model_checkpoint: str | None = None,
    config_hash: str | None = None,
    normalization_id: str | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"backend must be one of {sorted(SUPPORTED_BACKENDS)}")
    if adapter not in SUPPORTED_ADAPTERS:
        raise ValueError(f"adapter must be one of {sorted(SUPPORTED_ADAPTERS)}")
    if density_neighbors < 1:
        raise ValueError("density_neighbors must be at least 1")

    transition_rows = _load_jsonl(transitions)
    validate_unique_transition_ids(transition_rows)
    score_rows, summary = build_score_rows(
        transition_rows,
        backend=backend,
        adapter=adapter,
        density_neighbors=density_neighbors,
        model_checkpoint=model_checkpoint,
        config_hash=config_hash,
        normalization_id=normalization_id,
    )

    output_path = Path(output)
    summary_path = Path(summary_output)
    _prepare_output(output_path, overwrite=overwrite)
    _prepare_output(summary_path, overwrite=overwrite)
    append_jsonl(output_path, score_rows)
    write_json(summary_path, summary)
    return {
        "transition_count": len(transition_rows),
        "score_count": len(score_rows),
        "output": str(output_path),
        "summary_output": str(summary_path),
        "backend": backend,
        "adapter": adapter,
        "paper_claim_eligible": False,
    }


def build_score_rows(
    transitions: Iterable[dict[str, Any]],
    *,
    backend: str = "diffusion",
    adapter: str = "traffic_feature_density_proxy",
    density_neighbors: int = 1,
    model_checkpoint: str | None = None,
    config_hash: str | None = None,
    normalization_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(row) for row in transitions]
    feature_rows = [_transition_feature_vector(row) for row in rows]
    center, scale = _feature_center_scale(feature_rows)
    standardized_rows = [_standardize(features, center, scale) for features in feature_rows]
    if adapter == "traffic_feature_center_distance_proxy":
        rarity_scores = [_euclidean_distance(features, [0.0] * len(features)) for features in standardized_rows]
    else:
        rarity_scores = _local_density_rarity(standardized_rows, neighbor_count=density_neighbors)

    score_rows = []
    for idx, row in enumerate(rows):
        rarity_score = float(rarity_scores[idx])
        support_score = float(1.0 / (1.0 + max(0.0, rarity_score)))
        score_rows.append(
            {
                "schema_version": "r2v-generative-score-artifact-v1",
                "transition_id": str(row.get("transition_id")),
                "rarity_score": rarity_score,
                "support_score": support_score,
                "backend": backend,
                "adapter": adapter,
                "paper_claim_eligible": False,
                "rare_is_not_value_boundary": True,
                "model_checkpoint": model_checkpoint,
                "config_hash": config_hash,
                "normalization_id": normalization_id,
                "artifact_row_index": idx,
                "source_transition_index": idx,
            }
        )
    summary = {
        "schema_version": "r2v-generative-score-summary-v1",
        "transition_count": len(rows),
        "score_count": len(score_rows),
        "backend": backend,
        "adapter": adapter,
        "density_neighbors": int(density_neighbors),
        "model_checkpoint": model_checkpoint,
        "config_hash": config_hash,
        "normalization_id": normalization_id,
        "paper_claim_eligible": False,
        "rare_is_not_value_boundary": True,
        "rarity_min": min(rarity_scores) if rarity_scores else None,
        "rarity_max": max(rarity_scores) if rarity_scores else None,
        "support_min": min((row["support_score"] for row in score_rows), default=None),
        "support_max": max((row["support_score"] for row in score_rows), default=None),
    }
    return score_rows, summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a loader-compatible R2V generative score artifact from traffic transitions. "
            "The current adapters are deterministic traffic proxies for smoke/integration use."
        ),
    )
    parser.add_argument("--transitions", nargs="+", required=True, help="Input TransitionRecord JSONL path(s).")
    parser.add_argument("--output", required=True, help="Output score artifact JSONL path.")
    parser.add_argument("--summary_output", required=True, help="Output score summary JSON path.")
    parser.add_argument("--backend", choices=tuple(sorted(SUPPORTED_BACKENDS)), default="diffusion")
    parser.add_argument("--adapter", choices=tuple(sorted(SUPPORTED_ADAPTERS)), default="traffic_feature_density_proxy")
    parser.add_argument("--density_neighbors", type=int, default=1)
    parser.add_argument("--model_checkpoint", default=None)
    parser.add_argument("--config_hash", default=None)
    parser.add_argument("--normalization_id", default=None)
    parser.add_argument("--no_overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_score_artifact_from_files(
        transitions=args.transitions,
        output=args.output,
        summary_output=args.summary_output,
        backend=args.backend,
        adapter=args.adapter,
        density_neighbors=args.density_neighbors,
        model_checkpoint=args.model_checkpoint,
        config_hash=args.config_hash,
        normalization_id=args.normalization_id,
        overwrite=not args.no_overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _load_jsonl(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"transition rows must be JSON objects: {path}")
                    rows.append(row)
    return rows


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; remove --no_overwrite or choose another path")
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def _transition_feature_vector(row: dict[str, Any]) -> list[float]:
    obs = [_safe_float(value) for value in row.get("obs_features", [])]
    next_obs = [_safe_float(value) for value in row.get("next_obs_features", [])]
    if len(obs) != len(next_obs):
        raise ValueError(f"feature length mismatch for transition {row.get('transition_id')}")
    return obs + [right - left for left, right in zip(obs, next_obs)]


def _feature_center_scale(feature_rows: list[list[float]]) -> tuple[list[float], list[float]]:
    if not feature_rows:
        return [], []
    width = len(feature_rows[0])
    if any(len(row) != width for row in feature_rows):
        raise ValueError("all transition feature vectors must have the same length")
    center = []
    scale = []
    for idx in range(width):
        values = [row[idx] for row in feature_rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
        std = math.sqrt(max(variance, 0.0))
        center.append(mean)
        scale.append(std if std > 1e-9 else 1.0)
    return center, scale


def _standardize(features: list[float], center: list[float], scale: list[float]) -> list[float]:
    return [(value - mean) / std for value, mean, std in zip(features, center, scale)]


def _local_density_rarity(standardized_rows: list[list[float]], *, neighbor_count: int) -> list[float]:
    if len(standardized_rows) <= 1:
        return [0.0 for _ in standardized_rows]
    neighbor_count = min(max(1, int(neighbor_count)), len(standardized_rows) - 1)
    scores = []
    for idx, features in enumerate(standardized_rows):
        distances = [
            _euclidean_distance(features, other)
            for other_idx, other in enumerate(standardized_rows)
            if other_idx != idx
        ]
        distances.sort()
        scores.append(float(sum(distances[:neighbor_count]) / neighbor_count))
    return scores


def _euclidean_distance(features: list[float], other: list[float]) -> float:
    if not features:
        return 0.0
    squared = 0.0
    for value, right in zip(features, other):
        squared += (value - right) ** 2
    return float(math.sqrt(squared / len(features)))


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid numeric feature value: {value!r}") from None
    if not math.isfinite(result):
        raise ValueError(f"invalid numeric feature value: {value!r}")
    return result


if __name__ == "__main__":
    main()
