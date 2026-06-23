#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.data.normalization import RobustObjectiveNormalizer


def apply_file(input_path: str | Path, out_path: str | Path, normalizer: RobustObjectiveNormalizer) -> int:
    input_path = Path(input_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            record["objective_values_norm"] = normalizer.transform(record["objective_values_raw"])
            record.setdefault("metadata", {})["objective_normalizer_hash"] = normalizer.hash()
            record["metadata"]["objective_normalizer_version"] = normalizer.version
            dst.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalizer", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    normalizer = RobustObjectiveNormalizer.load(args.normalizer)
    out_dir = Path(args.out_dir)
    total = 0
    for input_path in args.inputs:
        total += apply_file(input_path, out_dir / Path(input_path).name, normalizer)
    print({"records": total, "normalizer_hash": normalizer.hash(), "out_dir": str(out_dir)})


if __name__ == "__main__":
    main()
