#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.common.io import append_jsonl, write_json
from pareto.r2v.artifact_validation import validate_unique_transition_ids
from pareto.r2v.traffic_candidate_selector import (
    R2VTrafficSelectorConfig,
    SUPPORTED_ADMISSION_MODES,
    SUPPORTED_REPAIR_METADATA_POLICIES,
    SUPPORTED_REPAIR_STORIES,
    apply_candidate_weights,
    select_r2v_candidates,
)


def load_jsonl(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        path_obj = Path(path)
        with path_obj.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def build_candidates_from_files(
    *,
    transitions: Iterable[str | Path],
    output: str | Path,
    summary_output: str | Path,
    config: R2VTrafficSelectorConfig,
    weighted_output: str | Path | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    transition_rows = load_jsonl(transitions)
    candidates, summary = select_r2v_candidates(transition_rows, config)

    output_path = Path(output)
    summary_path = Path(summary_output)
    weighted_path = Path(weighted_output) if weighted_output is not None else None
    if weighted_path is not None:
        validate_unique_transition_ids(transition_rows)
    _prepare_output(output_path, overwrite=overwrite)
    append_jsonl(output_path, candidates)
    write_json(summary_path, summary)

    if weighted_path is not None:
        _prepare_output(weighted_path, overwrite=overwrite)
        append_jsonl(
            weighted_path,
            apply_candidate_weights(
                transition_rows,
                candidates,
                config=config,
                summary=summary,
                summary_output=str(summary_path),
            ),
        )

    return {
        "transition_count": len(transition_rows),
        "candidate_count": len(candidates),
        "admitted_count": summary["admitted_count"],
        "output": str(output_path),
        "summary_output": str(summary_path),
        "weighted_output": str(weighted_path) if weighted_path is not None else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build R2V-TSC candidate replay rows from Pareto transition JSONL files.",
    )
    parser.add_argument("--transitions", nargs="+", required=True, help="Input TransitionRecord JSONL path(s).")
    parser.add_argument("--output", required=True, help="Output R2V candidate JSONL path.")
    parser.add_argument("--summary_output", required=True, help="Output summary JSON path.")
    parser.add_argument("--weighted_output", default=None, help="Optional transition JSONL copy with r2v_sample_weight metadata.")
    parser.add_argument("--candidate_model", default="feature_density_proxy")
    parser.add_argument(
        "--score_artifact",
        default=None,
        help=(
            "Optional JSONL/JSON artifact with precomputed generative rarity/support scores. "
            "Rows are joined by transition_id unless --score_artifact_id_key is changed."
        ),
    )
    parser.add_argument("--score_artifact_id_key", default="transition_id")
    parser.add_argument(
        "--score_artifact_backend",
        default=None,
        help="Optional backend label such as diffusion, flow, or score_matching.",
    )
    parser.add_argument(
        "--value_mode",
        choices=("objective_delta", "frozen_target_utility"),
        default="objective_delta",
        help="Value gate source: objective_delta preserves V1 behavior; frozen_target_utility uses explicit frozen targets.",
    )
    parser.add_argument("--rare_quantile", type=float, default=0.8)
    parser.add_argument("--value_quantile", type=float, default=0.6)
    parser.add_argument("--support_min_quantile", type=float, default=0.02)
    parser.add_argument("--safety_min", type=float, default=-1.0)
    parser.add_argument("--env_reward_weight", type=float, default=0.0)
    parser.add_argument("--density_neighbors", type=int, default=1)
    parser.add_argument(
        "--repair_story",
        choices=tuple(sorted(SUPPORTED_REPAIR_STORIES)),
        default="none",
        help=(
            "Repair-story admission semantics. Non-none values require source/final gate "
            "metadata and use final gates for admission."
        ),
    )
    parser.add_argument(
        "--repair_metadata_policy",
        choices=tuple(sorted(SUPPORTED_REPAIR_METADATA_POLICIES)),
        default="require_metadata",
        help=(
            "How to handle repair stories when source/final gate metadata is absent. "
            "require_metadata is fail-closed; metadata_or_proxy uses computed gates as a "
            "clearly marked proxy for smoke/integration runs."
        ),
    )
    parser.add_argument(
        "--gate_variant",
        choices=("full", "no_support", "no_ood", "no_dynamics"),
        default="full",
        help="Admission gate ablation. Full keeps rare/value/support/safety active.",
    )
    parser.add_argument(
        "--admission_mode",
        choices=tuple(sorted(SUPPORTED_ADMISSION_MODES)),
        default="weights_only",
        help=(
            "How admitted candidates enter replay. weights_only preserves the original "
            "transition set; weights_plus_repaired appends explicit repaired transition payloads."
        ),
    )
    parser.add_argument(
        "--source_gates_key",
        default="metadata.r2v_source_gates",
        help="Dotted path to the pre-repair gate map used by --repair_story.",
    )
    parser.add_argument(
        "--final_gates_key",
        default="metadata.r2v_final_gates",
        help="Dotted path to the post-repair gate map used by --repair_story.",
    )
    parser.add_argument(
        "--repaired_transition_key",
        default="metadata.r2v_repaired_transition",
        help="Dotted path to an explicit repaired transition payload for weights_plus_repaired.",
    )
    parser.add_argument("--base_weight", type=float, default=1.0)
    parser.add_argument(
        "--admitted_weight",
        type=float,
        default=None,
        help="Exact sample weight for admitted candidates. When provided, it overrides admitted_weight_bonus.",
    )
    parser.add_argument("--admitted_weight_bonus", type=float, default=2.0)
    parser.add_argument(
        "--repair_rejected_weight",
        type=float,
        default=2.0,
        help="Sample weight for explicit repaired proposals that remain rejected by admission gates.",
    )
    parser.add_argument("--max_weight", type=float, default=5.0)
    parser.add_argument(
        "--utility_weights",
        default=None,
        help="Optional JSON object with efficiency/safety/fairness/stability weights.",
    )
    parser.add_argument("--no_overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = json.loads(args.utility_weights) if args.utility_weights else None
    config = R2VTrafficSelectorConfig(
        candidate_model=args.candidate_model,
        value_mode=args.value_mode,
        rare_quantile=args.rare_quantile,
        value_quantile=args.value_quantile,
        support_min_quantile=args.support_min_quantile,
        safety_min=args.safety_min,
        env_reward_weight=args.env_reward_weight,
        density_neighbors=args.density_neighbors,
        repair_story=args.repair_story,
        repair_metadata_policy=args.repair_metadata_policy,
        gate_variant=args.gate_variant,
        admission_mode=args.admission_mode,
        source_gates_key=args.source_gates_key,
        final_gates_key=args.final_gates_key,
        repaired_transition_key=args.repaired_transition_key,
        base_weight=args.base_weight,
        admitted_weight=args.admitted_weight,
        admitted_weight_bonus=args.admitted_weight_bonus,
        repair_rejected_weight=args.repair_rejected_weight,
        max_weight=args.max_weight,
        utility_weights=weights,
        score_artifact_path=args.score_artifact,
        score_artifact_id_key=args.score_artifact_id_key,
        score_artifact_backend=args.score_artifact_backend,
    )
    report = build_candidates_from_files(
        transitions=args.transitions,
        output=args.output,
        summary_output=args.summary_output,
        weighted_output=args.weighted_output,
        config=config,
        overwrite=not args.no_overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite or remove --no_overwrite")
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
