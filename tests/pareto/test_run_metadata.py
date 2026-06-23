from pathlib import Path

from pareto.common.run_metadata import stable_hash, write_run_metadata


def test_stable_hash_is_order_independent():
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})


def test_write_run_metadata_creates_command_and_json(tmp_path: Path):
    metadata = write_run_metadata(
        tmp_path,
        command="python scripts/smoke_env.py --scenario jinan",
        config={"seed": 0, "scenario": "jinan"},
    )
    assert (tmp_path / "command.txt").read_text().startswith("python scripts/smoke_env.py")
    assert (tmp_path / "metadata.json").exists()
    assert metadata["config_hash"] == stable_hash({"seed": 0, "scenario": "jinan"})
