#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.common.io import write_json
from pareto.data.schema import DominancePairLabel, ObjectivePairLabel, PreferencePairLabel


PAIR_FILES = {
    "objective_pairs": ("objective_pairs.jsonl", ObjectivePairLabel),
    "preference_pairs": ("preference_pairs.jsonl", PreferencePairLabel),
    "dominance_pairs": ("dominance_pairs.jsonl", DominancePairLabel),
}


def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _positive_ratio(rows: Iterable[Dict]) -> float | None:
    labels = [int(row["label"]) for row in rows if "label" in row]
    if not labels:
        return None
    return float(sum(labels) / len(labels))


def _nested_positive_ratio(groups: Dict[str, List[int]]) -> Dict[str, float]:
    ratios = {}
    for name, labels in groups.items():
        if labels:
            ratios[name] = float(sum(labels) / len(labels))
    return ratios


def _parse_gate_spec(values: Iterable[str] | None) -> Dict[str, int]:
    result = {}
    for value in values or []:
        if ":" not in value:
            raise ValueError(f"gate spec must be name:count, got {value}")
        name, count = value.split(":", 1)
        result[name] = int(count)
    return result


def summarize_pairs_dir(pairs_dir: str | Path) -> Dict:
    pairs_dir = Path(pairs_dir)
    errors = []
    summary = {
        "serialized_tie_count": 0,
        "invalid_objective_pair_count": 0,
        "counts": {},
        "positive_ratios": {},
        "positive_ratio_by_objective": {},
        "positive_ratio_by_strategy": {},
        "objective_counts": {},
        "sampling_strategy_counts": {},
        "reversal_by_template_pair": {},
        "split_counts": {},
    }
    labels_by_objective: Dict[str, List[int]] = {}
    labels_by_strategy: Dict[str, List[int]] = {}
    for name, (filename, cls) in PAIR_FILES.items():
        rows = _load_jsonl(pairs_dir / filename)
        summary["counts"][name] = len(rows)
        summary["positive_ratios"][name] = _positive_ratio(rows)
        if name == "objective_pairs":
            for row in rows:
                objective = row.get("objective")
                summary["objective_counts"][objective] = summary["objective_counts"].get(objective, 0) + 1
                if "label" in row and objective:
                    labels_by_objective.setdefault(objective, []).append(int(row["label"]))
        for idx, row in enumerate(rows):
            try:
                cls.from_json(row).validate()
                if row.get("is_tie", False):
                    summary["serialized_tie_count"] += 1
                split = row.get("split")
                if split:
                    summary["split_counts"][split] = summary["split_counts"].get(split, 0) + 1
                strategy = row.get("sampling_strategy")
                if strategy:
                    summary["sampling_strategy_counts"][strategy] = (
                        summary["sampling_strategy_counts"].get(strategy, 0) + 1
                    )
                    if "label" in row:
                        labels_by_strategy.setdefault(strategy, []).append(int(row["label"]))
                if name == "objective_pairs":
                    margin = float(row.get("margin_norm", 0.0))
                    if int(margin > 0) != int(row.get("label")):
                        errors.append({
                            "file": filename,
                            "line": idx + 1,
                            "error": "objective label does not match margin_norm sign",
                        })
            except Exception as exc:
                errors.append({"file": filename, "line": idx + 1, "error": str(exc)})

    reversal_rows = _load_jsonl(pairs_dir / "reversal_pairs.jsonl")
    summary["counts"]["reversal_pairs"] = len(reversal_rows)
    for idx, row in enumerate(reversal_rows):
        if row.get("label_1") == row.get("label_2"):
            errors.append({"file": "reversal_pairs.jsonl", "line": idx + 1, "error": "reversal labels do not differ"})
        split = row.get("split")
        if split:
            summary["split_counts"][split] = summary["split_counts"].get(split, 0) + 1
        key = f"{row.get('w_1_name')}__{row.get('w_2_name')}"
        summary["reversal_by_template_pair"][key] = summary["reversal_by_template_pair"].get(key, 0) + 1

    summary["error_count"] = len(errors)
    summary["errors"] = errors[:20]
    summary["positive_ratio_by_objective"] = _nested_positive_ratio(labels_by_objective)
    summary["positive_ratio_by_strategy"] = _nested_positive_ratio(labels_by_strategy)
    return summary


def apply_gates(
    summary: Dict,
    min_objective_per_head: int = 0,
    min_preference_pairs: int = 0,
    min_dominance_pairs: int = 0,
    min_reversal_pairs: int = 0,
    positive_ratio_low: float | None = None,
    positive_ratio_high: float | None = None,
    require_no_ties: bool = False,
    min_eff_controlled_fairness: int = 0,
    min_eff_controlled_stability: int = 0,
    min_efficiency_safety_conflict: int = 0,
    min_efficiency_stability_conflict: int = 0,
    min_split_pairs: Dict[str, int] | None = None,
    min_reversal_template_pair: Dict[str, int] | None = None,
    positive_ratio_by_objective_low: float | None = None,
    positive_ratio_by_objective_high: float | None = None,
) -> Dict:
    errors = list(summary.get("errors", []))
    counts = summary.get("counts", {})
    if counts.get("preference_pairs", 0) < min_preference_pairs:
        errors.append({"file": None, "line": None, "error": "preference_pairs count below minimum"})
    if counts.get("dominance_pairs", 0) < min_dominance_pairs:
        errors.append({"file": None, "line": None, "error": "dominance_pairs count below minimum"})
    if counts.get("reversal_pairs", 0) < min_reversal_pairs:
        errors.append({"file": None, "line": None, "error": "reversal_pairs count below minimum"})
    if require_no_ties and summary.get("serialized_tie_count", 0):
        errors.append({"file": None, "line": None, "error": "serialized tie pairs are not allowed"})
    for objective, count in summary.get("objective_counts", {}).items():
        if count < min_objective_per_head:
            errors.append({"file": None, "line": None, "error": f"{objective} objective pair count below minimum"})
    if min_objective_per_head:
        missing = {"efficiency", "safety", "fairness", "stability"} - set(summary.get("objective_counts", {}))
        for objective in sorted(missing):
            errors.append({"file": None, "line": None, "error": f"{objective} objective pair count below minimum"})
    if positive_ratio_low is not None and positive_ratio_high is not None:
        for name, ratio in summary.get("positive_ratios", {}).items():
            if ratio is None:
                continue
            if ratio < positive_ratio_low or ratio > positive_ratio_high:
                errors.append({"file": None, "line": None, "error": f"{name} positive ratio outside bounds"})
    strategy_counts = summary.get("sampling_strategy_counts", {})
    contrast_gates = {
        "eff_controlled_fairness": min_eff_controlled_fairness,
        "eff_controlled_stability": min_eff_controlled_stability,
        "efficiency_safety_conflict": min_efficiency_safety_conflict,
        "efficiency_stability_conflict": min_efficiency_stability_conflict,
    }
    for strategy, minimum in contrast_gates.items():
        if minimum and strategy_counts.get(strategy, 0) < minimum:
            errors.append({"file": None, "line": None, "error": f"{strategy} count below minimum"})
    for split, minimum in (min_split_pairs or {}).items():
        if summary.get("split_counts", {}).get(split, 0) < minimum:
            errors.append({"file": None, "line": None, "error": f"{split} split pair count below minimum"})
    for template_pair, minimum in (min_reversal_template_pair or {}).items():
        if summary.get("reversal_by_template_pair", {}).get(template_pair, 0) < minimum:
            errors.append({"file": None, "line": None, "error": f"{template_pair} reversal count below minimum"})
    if positive_ratio_by_objective_low is not None and positive_ratio_by_objective_high is not None:
        for objective, ratio in summary.get("positive_ratio_by_objective", {}).items():
            if ratio < positive_ratio_by_objective_low or ratio > positive_ratio_by_objective_high:
                errors.append({"file": None, "line": None, "error": f"{objective} positive ratio outside bounds"})
    summary = dict(summary)
    summary["error_count"] = len(errors)
    summary["errors"] = errors[:20]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--min_objective_per_head", type=int, default=0)
    parser.add_argument("--min_preference_pairs", type=int, default=0)
    parser.add_argument("--min_dominance_pairs", type=int, default=0)
    parser.add_argument("--min_reversal_pairs", type=int, default=0)
    parser.add_argument("--positive_ratio_low", type=float)
    parser.add_argument("--positive_ratio_high", type=float)
    parser.add_argument("--require_no_ties", action="store_true")
    parser.add_argument("--min_eff_controlled_fairness", type=int, default=0)
    parser.add_argument("--min_eff_controlled_stability", type=int, default=0)
    parser.add_argument("--min_efficiency_safety_conflict", type=int, default=0)
    parser.add_argument("--min_efficiency_stability_conflict", type=int, default=0)
    parser.add_argument("--min_split_pairs", action="append")
    parser.add_argument("--min_reversal_template_pair", action="append")
    parser.add_argument("--positive_ratio_by_objective_low", type=float)
    parser.add_argument("--positive_ratio_by_objective_high", type=float)
    args = parser.parse_args()
    report = summarize_pairs_dir(args.pairs_dir)
    if args.strict:
        report = apply_gates(
            report,
            min_objective_per_head=args.min_objective_per_head,
            min_preference_pairs=args.min_preference_pairs,
            min_dominance_pairs=args.min_dominance_pairs,
            min_reversal_pairs=args.min_reversal_pairs,
            positive_ratio_low=args.positive_ratio_low,
            positive_ratio_high=args.positive_ratio_high,
            require_no_ties=args.require_no_ties,
            min_eff_controlled_fairness=args.min_eff_controlled_fairness,
            min_eff_controlled_stability=args.min_eff_controlled_stability,
            min_efficiency_safety_conflict=args.min_efficiency_safety_conflict,
            min_efficiency_stability_conflict=args.min_efficiency_stability_conflict,
            min_split_pairs=_parse_gate_spec(args.min_split_pairs),
            min_reversal_template_pair=_parse_gate_spec(args.min_reversal_template_pair),
            positive_ratio_by_objective_low=args.positive_ratio_by_objective_low,
            positive_ratio_by_objective_high=args.positive_ratio_by_objective_high,
        )
    write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
