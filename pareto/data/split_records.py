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


def load_records(paths: Iterable[str | Path]) -> List[Dict]:
    records = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def _group_id(record: Dict, group_key: str, time_block_size: int) -> str:
    if group_key == "sample":
        return str(record["sample_id"])
    if group_key == "episode":
        return f"{record.get('run_id', '')}:{record.get('policy_id', '')}:ep{record.get('episode', 0)}"
    if group_key == "time_block":
        sim_time = float(record.get("sim_time_sec", record.get("step", 0)))
        block = int(sim_time // max(time_block_size, 1))
        return f"{record.get('run_id', '')}:{record.get('policy_id', '')}:ep{record.get('episode', 0)}:block{block}"
    raise ValueError(f"unknown group_key: {group_key}")


def _assign_groups(
    groups: Dict[str, List[Dict]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, str]:
    items: List[Tuple[str, List[Dict]]] = list(groups.items())
    rng = random.Random(seed)
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)
    total = sum(len(rows) for _, rows in items)
    targets = {
        "train": total * train_ratio,
        "val": total * val_ratio,
        "test": total * max(0.0, 1.0 - train_ratio - val_ratio),
    }
    counts = {"train": 0, "val": 0, "test": 0}
    assignment = {}
    for group_id, rows in items:
        split = max(("train", "val", "test"), key=lambda name: targets[name] - counts[name])
        assignment[group_id] = split
        counts[split] += len(rows)
    return assignment


def _sample_overlap(split_rows: Dict[str, List[Dict]]) -> Dict[str, List[str]]:
    seen = {}
    overlap = {}
    for split, rows in split_rows.items():
        for record in rows:
            sample_id = record.get("sample_id")
            if not sample_id:
                continue
            previous = seen.setdefault(sample_id, split)
            if previous != split:
                overlap[sample_id] = sorted({previous, split})
    return overlap


def split_records(
    inputs: Iterable[str | Path],
    out_dir: str | Path,
    seed: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    group_key: str = "sample",
    time_block_size: int = 300,
) -> Dict:
    records = load_records(inputs)
    groups: Dict[str, List[Dict]] = {}
    for record in records:
        groups.setdefault(_group_id(record, group_key, time_block_size), []).append(record)
    assignment = _assign_groups(groups, seed, train_ratio, val_ratio)
    split_rows = {"train": [], "val": [], "test": []}
    for group_id, rows in groups.items():
        split_rows[assignment[group_id]].extend(rows)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        path = out_dir / f"{split}_raw.jsonl"
        path.write_text("", encoding="utf-8")
        if rows:
            append_jsonl(path, rows)

    report = {
        "inputs": [str(path) for path in inputs],
        "out_dir": str(out_dir),
        "seed": seed,
        "group_key": group_key,
        "time_block_size": time_block_size,
        "record_count": len(records),
        "group_count": len(groups),
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "sample_overlap": _sample_overlap(split_rows),
    }
    write_json(out_dir / "split_records_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--group_key", choices=["sample", "episode", "time_block"], default="sample")
    parser.add_argument("--time_block_size", type=int, default=300)
    args = parser.parse_args()
    report = split_records(
        args.inputs,
        args.out_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        group_key=args.group_key,
        time_block_size=args.time_block_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["sample_overlap"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
