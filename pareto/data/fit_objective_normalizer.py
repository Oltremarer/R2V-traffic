#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.data.normalization import RobustObjectiveNormalizer, load_jsonl_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffers", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--clip", type=float, default=5.0)
    args = parser.parse_args()

    normalizer = RobustObjectiveNormalizer.fit(
        load_jsonl_records(args.buffers),
        clip=args.clip,
        fit_input_files=args.buffers,
    )
    normalizer.save(args.out)
    print(normalizer.to_dict())


if __name__ == "__main__":
    main()
