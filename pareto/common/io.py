from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable


def ensure_parent(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj


def write_json(path: str | Path, data: Dict) -> None:
    path_obj = ensure_parent(path)
    path_obj.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: str | Path, rows: Iterable[Dict]) -> None:
    path_obj = ensure_parent(path)
    with path_obj.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def append_metrics_csv(path: str | Path, row: Dict[str, object]) -> None:
    path_obj = ensure_parent(path)
    write_header = not path_obj.exists()
    with path_obj.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
