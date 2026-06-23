from pathlib import Path
import json

import pytest

torch = pytest.importorskip("torch")

from pareto.data.offline_dataset import (
    dominance_pair_tensors,
    load_records_by_id,
    objective_pair_tensors,
    preference_pair_tensors,
    read_jsonl,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _record(sample_id: str, value: float) -> dict:
    objectives = {
        "efficiency": value,
        "safety": value + 1,
        "fairness": value + 2,
        "stability": value + 3,
    }
    return {
        "sample_id": sample_id,
        "obs_features": [value, value + 0.5],
        "objective_values_norm": objectives,
        "objective_valid_mask": {key: True for key in objectives},
    }


def test_read_jsonl_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"a": 1}) + "\n\n" + json.dumps({"b": 2}) + "\n", encoding="utf-8")
    assert read_jsonl(path) == [{"a": 1}, {"b": 2}]


def test_load_records_by_id_rejects_duplicate_sample_id(tmp_path: Path):
    path = tmp_path / "records.jsonl"
    _write_jsonl(path, [_record("same", 1.0), _record("same", 2.0)])

    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_records_by_id([path])


def test_pair_tensor_conversions_preserve_labels_and_orientation(tmp_path: Path):
    records = {"a": _record("a", 1.0), "b": _record("b", 2.0)}
    objective = objective_pair_tensors(
        [{"a_id": "a", "b_id": "b", "objective": "fairness", "label": 0}],
        records,
    )
    assert objective["x_a"].shape == (1, 2)
    assert objective["objective_idx"].tolist() == [2]
    assert objective["labels"].tolist() == [0.0]

    preference = preference_pair_tensors(
        [{"a_id": "a", "b_id": "b", "w": [0.25, 0.25, 0.25, 0.25], "label": 1}],
        records,
    )
    assert preference["w"].shape == (1, 4)
    assert preference["labels"].tolist() == [1.0]

    dominance = dominance_pair_tensors(
        [{"a_id": "a", "b_id": "b", "dominates": "b"}],
        records,
    )
    assert dominance["x_dom"].tolist() == [[2.0, 2.5]]
    assert dominance["x_sub"].tolist() == [[1.0, 1.5]]

