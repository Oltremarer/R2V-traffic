from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


_MISSING = object()


@dataclass(frozen=True)
class GenerativeScoreRow:
    transition_id: str
    rarity_score: float
    support_score: float
    repaired_transition: dict[str, Any] | None
    source: dict[str, Any]


def load_generative_score_artifact(
    path: str | Path,
    *,
    id_key: str = "transition_id",
    backend: str | None = None,
) -> tuple[dict[str, GenerativeScoreRow], dict[str, Any]]:
    path_obj = Path(path)
    rows = _load_json_rows(path_obj)
    score_rows: dict[str, GenerativeScoreRow] = {}
    for idx, row in enumerate(rows):
        transition_id = _required_string(row.get(id_key, _MISSING), field=id_key, row_idx=idx)
        if transition_id in score_rows:
            raise ValueError(f"duplicate score artifact {id_key} {transition_id!r} at row {idx}")
        rarity_score = _score_value(row, row_idx=idx)
        support_score = _support_value(row, rarity_score=rarity_score, row_idx=idx)
        score_rows[transition_id] = GenerativeScoreRow(
            transition_id=transition_id,
            rarity_score=rarity_score,
            support_score=support_score,
            repaired_transition=_repaired_transition_payload(row, row_idx=idx),
            source={
                "kind": "score_artifact",
                "backend": backend or row.get("backend") or row.get("candidate_model") or "generative",
                "artifact_path": str(path_obj),
                "artifact_row_index": idx,
                "model_checkpoint": row.get("model_checkpoint"),
                "config_hash": row.get("config_hash"),
                "normalization_id": row.get("normalization_id"),
                "adapter": row.get("adapter"),
                "paper_claim_eligible": row.get("paper_claim_eligible"),
            },
        )
    summary = {
        "kind": "score_artifact",
        "backend": backend or "generative",
        "artifact_path": str(path_obj),
        "id_key": id_key,
        "row_count": len(rows),
        "score_count": len(score_rows),
    }
    return score_rows, summary


def _repaired_transition_payload(row: dict[str, Any], *, row_idx: int) -> dict[str, Any] | None:
    if "repaired_transition" not in row:
        return None
    payload = row["repaired_transition"]
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"repaired_transition must be an object in score artifact row {row_idx}")
    for field in ("transition_id", "sample_id"):
        if field not in payload or payload[field] is None or str(payload[field]).strip() == "":
            raise ValueError(f"repaired_transition missing {field} in score artifact row {row_idx}")
    return dict(payload)


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        text = handle.read().strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"score artifact must be a JSON list or JSONL file: {path}")
        rows = data
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"score artifact rows must be JSON objects: {path}")
    return list(rows)


def _required_string(value: Any, *, field: str, row_idx: int) -> str:
    if value is _MISSING or value is None or str(value) == "":
        raise ValueError(f"missing {field} in score artifact row {row_idx}")
    return str(value)


def _score_value(row: dict[str, Any], *, row_idx: int) -> float:
    for key in ("rarity_score", "score", "detector_score", "normalized_detector_score"):
        if key in row:
            return _finite_float(row[key], field=key, row_idx=row_idx)
    raise ValueError(f"missing rarity_score in score artifact row {row_idx}")


def _support_value(row: dict[str, Any], *, rarity_score: float, row_idx: int) -> float:
    if "support_score" in row:
        support_score = _finite_float(row["support_score"], field="support_score", row_idx=row_idx)
    else:
        support_score = 1.0 / (1.0 + max(0.0, rarity_score))
    if support_score <= 0.0:
        raise ValueError(f"support_score must be positive in score artifact row {row_idx}")
    return float(support_score)


def _finite_float(value: Any, *, field: str, row_idx: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid {field} in score artifact row {row_idx}: {value!r}") from None
    if not math.isfinite(result):
        raise ValueError(f"invalid {field} in score artifact row {row_idx}: {value!r}")
    return float(result)
