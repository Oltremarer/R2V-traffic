from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from pareto.constants import OBJECTIVE_NAMES


@dataclass
class RobustObjectiveNormalizer:
    median: Dict[str, float]
    iqr: Dict[str, float]
    valid_count: Dict[str, int]
    raw_q25: Dict[str, float]
    raw_q50: Dict[str, float]
    raw_q75: Dict[str, float]
    zero_iqr_objectives: List[str]
    fit_input_files: List[str]
    clip: float = 5.0
    version: str = "robust-objective-normalizer-v1"

    @classmethod
    def fit(
        cls,
        records: Iterable[Dict],
        clip: float = 5.0,
        fit_input_files: Iterable[str | Path] | None = None,
    ) -> "RobustObjectiveNormalizer":
        values: Dict[str, List[float]] = {name: [] for name in OBJECTIVE_NAMES}
        for record in records:
            raw = record.get("objective_values_raw", record)
            valid = record.get("objective_valid_mask", {name: True for name in OBJECTIVE_NAMES})
            for name in OBJECTIVE_NAMES:
                if valid.get(name, False):
                    values[name].append(float(raw[name]))

        median = {}
        iqr = {}
        raw_q25 = {}
        raw_q50 = {}
        raw_q75 = {}
        valid_count = {}
        zero_iqr_objectives = []
        for name in OBJECTIVE_NAMES:
            arr = np.asarray(values[name], dtype=np.float64)
            valid_count[name] = int(arr.size)
            if arr.size == 0:
                median[name] = 0.0
                iqr[name] = 1.0
                raw_q25[name] = 0.0
                raw_q50[name] = 0.0
                raw_q75[name] = 0.0
                zero_iqr_objectives.append(name)
                continue
            q25, q50, q75 = np.percentile(arr, [25, 50, 75])
            scale = float(q75 - q25)
            median[name] = float(q50)
            raw_q25[name] = float(q25)
            raw_q50[name] = float(q50)
            raw_q75[name] = float(q75)
            iqr[name] = scale if scale > 1e-8 else 1.0
            if scale <= 1e-8:
                zero_iqr_objectives.append(name)
        input_files = [str(Path(path)) for path in (fit_input_files or [])]
        return cls(
            median=median,
            iqr=iqr,
            valid_count=valid_count,
            raw_q25=raw_q25,
            raw_q50=raw_q50,
            raw_q75=raw_q75,
            zero_iqr_objectives=zero_iqr_objectives,
            fit_input_files=input_files,
            clip=float(clip),
        )

    def transform(self, raw: Dict[str, float]) -> Dict[str, float]:
        result = {}
        for name in OBJECTIVE_NAMES:
            value = (float(raw[name]) - self.median[name]) / self.iqr[name]
            result[name] = float(np.clip(value, -self.clip, self.clip))
        return result

    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "objective_order": list(OBJECTIVE_NAMES),
            "median": self.median,
            "iqr": self.iqr,
            "valid_count": self.valid_count,
            "raw_q25": self.raw_q25,
            "raw_q50": self.raw_q50,
            "raw_q75": self.raw_q75,
            "zero_iqr_objectives": self.zero_iqr_objectives,
            "fit_input_files": self.fit_input_files,
            "clip": self.clip,
            "hash": self.hash(),
        }

    def hash(self) -> str:
        payload = json.dumps(
            {
                "version": self.version,
                "objective_order": list(OBJECTIVE_NAMES),
                "median": self.median,
                "iqr": self.iqr,
                "valid_count": self.valid_count,
                "raw_q25": self.raw_q25,
                "raw_q50": self.raw_q50,
                "raw_q75": self.raw_q75,
                "zero_iqr_objectives": self.zero_iqr_objectives,
                "fit_input_files": self.fit_input_files,
                "clip": self.clip,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RobustObjectiveNormalizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        median = {name: float(data["median"][name]) for name in OBJECTIVE_NAMES}
        iqr = {name: float(data["iqr"][name]) for name in OBJECTIVE_NAMES}
        return cls(
            median=median,
            iqr=iqr,
            valid_count={name: int(data.get("valid_count", {}).get(name, 0)) for name in OBJECTIVE_NAMES},
            raw_q25={name: float(data.get("raw_q25", median).get(name, median[name])) for name in OBJECTIVE_NAMES},
            raw_q50={name: float(data.get("raw_q50", median).get(name, median[name])) for name in OBJECTIVE_NAMES},
            raw_q75={name: float(data.get("raw_q75", median).get(name, median[name])) for name in OBJECTIVE_NAMES},
            zero_iqr_objectives=list(data.get("zero_iqr_objectives", [])),
            fit_input_files=list(data.get("fit_input_files", [])),
            clip=float(data.get("clip", 5.0)),
            version=data.get("version", "robust-objective-normalizer-v1"),
        )


def load_jsonl_records(paths: Iterable[str | Path]) -> List[Dict]:
    records = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records
