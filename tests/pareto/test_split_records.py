import json
from pathlib import Path

from pareto.data.split_records import split_records


def _record(sample_id, episode=0, step=0):
    return {
        "sample_id": sample_id,
        "run_id": "run0",
        "episode": episode,
        "step": step,
        "sim_time_sec": float(step * 30),
        "policy_id": "maxpressure",
    }


def _write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_split_records_writes_disjoint_train_val_test_files(tmp_path: Path):
    input_path = tmp_path / "records.jsonl"
    rows = [_record(f"s{i}", step=i) for i in range(30)]
    _write_jsonl(input_path, rows)

    report = split_records([input_path], tmp_path / "split", seed=0)

    split_ids = {
        split: {row["sample_id"] for row in _read_jsonl(tmp_path / "split" / f"{split}_raw.jsonl")}
        for split in ("train", "val", "test")
    }

    assert report["sample_overlap"] == {}
    assert split_ids["train"]
    assert split_ids["val"]
    assert split_ids["test"]
    assert split_ids["train"].isdisjoint(split_ids["val"])
    assert split_ids["train"].isdisjoint(split_ids["test"])
    assert split_ids["val"].isdisjoint(split_ids["test"])


def test_split_records_can_group_by_time_block(tmp_path: Path):
    input_path = tmp_path / "records.jsonl"
    rows = [_record(f"s{i}", step=i) for i in range(12)]
    _write_jsonl(input_path, rows)

    report = split_records(
        [input_path],
        tmp_path / "split",
        seed=0,
        group_key="time_block",
        time_block_size=120,
    )

    assert report["group_key"] == "time_block"
    assert report["sample_overlap"] == {}
