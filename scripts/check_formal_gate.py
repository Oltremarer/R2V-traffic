#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.eval.formal_gate import evaluate_formal_gate
from pareto.train_common import write_json


def _load_metrics(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "test" in payload and isinstance(payload["test"], dict):
        return payload["test"]
    if "metrics" in payload and isinstance(payload["metrics"], dict):
        return payload["metrics"]
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_metrics", required=True)
    parser.add_argument("--film_metrics")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vector_metrics = _load_metrics(args.vector_metrics)
    film_metrics = _load_metrics(args.film_metrics) if args.film_metrics else None
    decision = evaluate_formal_gate(vector_metrics, film_metrics)
    write_json(args.out, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
