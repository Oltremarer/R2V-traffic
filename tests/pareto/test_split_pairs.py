import json
from pathlib import Path

from pareto.data.split_pairs import split_pair_rows, split_pairs_dir


def _pair(pair_id, a_id, b_id):
    return {
        "pair_id": pair_id,
        "split": "train",
        "scenario": "jinan",
        "a_id": a_id,
        "b_id": b_id,
        "objective": "efficiency",
        "label": 1,
        "is_tie": False,
        "margin_raw": 1.0,
        "margin_norm": 1.0,
        "source": "rule",
        "rule_version": "v1",
        "sampling_strategy": "objective_contrast",
    }


def test_split_pair_rows_keeps_sample_ids_disjoint():
    rows = [_pair("p0", "a", "b"), _pair("p1", "c", "d"), _pair("p2", "e", "f")]
    split_rows, report = split_pair_rows(rows, seed=0, train_ratio=0.34, val_ratio=0.33)

    seen = {}
    for row in split_rows:
        for sample_id in (row["a_id"], row["b_id"]):
            previous = seen.setdefault(sample_id, row["split"])
            assert previous == row["split"]

    assert report["sample_overlap"] == {}


def test_split_pairs_dir_rewrites_pair_files(tmp_path: Path):
    pairs_dir = tmp_path / "pairs"
    out_dir = tmp_path / "out"
    pairs_dir.mkdir()
    (pairs_dir / "objective_pairs.jsonl").write_text(
        "\n".join(json.dumps(row) for row in [_pair("p0", "a", "b"), _pair("p1", "c", "d")]) + "\n",
        encoding="utf-8",
    )

    report = split_pairs_dir(pairs_dir, out_dir, seed=0)

    assert (out_dir / "objective_pairs.jsonl").exists()
    assert report["files"]["objective_pairs.jsonl"]["written_count"] == 2
    assert report["sample_overlap"] == {}
