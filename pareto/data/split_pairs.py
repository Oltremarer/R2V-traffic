#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.common.io import append_jsonl, write_json


PAIR_FILENAMES = (
    "objective_pairs.jsonl",
    "objective_pairs_eff_controlled_fairness.jsonl",
    "objective_pairs_eff_controlled_stability.jsonl",
    "preference_pairs.jsonl",
    "dominance_pairs.jsonl",
    "reversal_pairs.jsonl",
)


def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    append_jsonl(path, rows)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def _component_assignment(
    rows: Iterable[Dict],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, str]:
    uf = _UnionFind()
    sample_ids = set()
    for row in rows:
        a_id = row.get("a_id")
        b_id = row.get("b_id")
        if not a_id or not b_id:
            continue
        sample_ids.update((a_id, b_id))
        uf.union(a_id, b_id)

    components: Dict[str, List[str]] = {}
    for sample_id in sample_ids:
        components.setdefault(uf.find(sample_id), []).append(sample_id)

    component_items: List[Tuple[str, List[str]]] = list(components.items())
    rng = random.Random(seed)
    rng.shuffle(component_items)
    component_items.sort(key=lambda item: len(item[1]), reverse=True)
    total_samples = sum(len(ids) for _, ids in component_items)
    targets = {
        "train": total_samples * train_ratio,
        "val": total_samples * val_ratio,
        "test": total_samples * max(0.0, 1.0 - train_ratio - val_ratio),
    }
    counts = {"train": 0, "val": 0, "test": 0}
    assignment = {}
    for _, ids in component_items:
        split = max(("train", "val", "test"), key=lambda name: targets[name] - counts[name])
        counts[split] += len(ids)
        for sample_id in ids:
            assignment[sample_id] = split
    return assignment


def _sample_overlap(rows: Iterable[Dict]) -> Dict[str, List[str]]:
    seen: Dict[str, str] = {}
    overlap: Dict[str, List[str]] = {}
    for row in rows:
        split = row.get("split", "train")
        for sample_id in (row.get("a_id"), row.get("b_id")):
            if not sample_id:
                continue
            previous = seen.setdefault(sample_id, split)
            if previous != split:
                overlap.setdefault(sample_id, sorted({previous, split}))
    return overlap


def split_pair_rows(
    rows: List[Dict],
    seed: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    assignment: Dict[str, str] | None = None,
) -> Tuple[List[Dict], Dict]:
    assignment = assignment or _component_assignment(rows, seed, train_ratio, val_ratio)
    split_rows = []
    dropped = 0
    for row in rows:
        a_split = assignment.get(row.get("a_id"))
        b_split = assignment.get(row.get("b_id"))
        if not a_split or not b_split or a_split != b_split:
            dropped += 1
            continue
        updated = dict(row)
        updated["split"] = a_split
        split_rows.append(updated)
    report = {
        "input_count": len(rows),
        "written_count": len(split_rows),
        "dropped_cross_split_count": dropped,
        "split_counts": {},
        "sample_overlap": _sample_overlap(split_rows),
    }
    for row in split_rows:
        split = row.get("split", "train")
        report["split_counts"][split] = report["split_counts"].get(split, 0) + 1
    return split_rows, report


def split_pairs_dir(
    pairs_dir: str | Path,
    out_dir: str | Path,
    seed: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict:
    pairs_dir = Path(pairs_dir)
    out_dir = Path(out_dir)
    rows_by_file = {
        filename: _load_jsonl(pairs_dir / filename)
        for filename in PAIR_FILENAMES
        if (pairs_dir / filename).exists()
    }
    all_rows = [row for rows in rows_by_file.values() for row in rows]
    assignment = _component_assignment(all_rows, seed, train_ratio, val_ratio)
    report = {"files": {}, "sample_overlap": {}, "assignment_count": len(assignment)}
    written_all = []
    for filename, rows in rows_by_file.items():
        split_rows, file_report = split_pair_rows(
            rows,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            assignment=assignment,
        )
        _write_jsonl(out_dir / filename, split_rows)
        report["files"][filename] = file_report
        written_all.extend(split_rows)
    report["sample_overlap"] = _sample_overlap(written_all)
    write_json(out_dir / "split_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()
    report = split_pairs_dir(
        args.pairs_dir,
        args.out_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["sample_overlap"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
