from __future__ import annotations

import json
from pathlib import Path

import pytest

from pareto.rl.formal_result_field_inventory import (
    FORMAL_RESULT_FIELD_INVENTORY_APPROVAL_PHRASE,
    run_formal_result_field_inventory,
)
from pareto.rl.formal_result_field_inventory_validator import validate_field_inventory_packet


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fake_formal_root(tmp_path: Path) -> Path:
    root = tmp_path / "records" / "formal"
    run_dir = root / "seed0" / "env_reward"
    _write_json(
        run_dir / "metadata.json",
        {
            "method": "env_reward",
            "cityflow_seed": 0,
            "traffic_file": "anon_3_4_jinan_real.json",
            "hidden_metric_value": 12345.6789,
        },
    )
    _write_json(run_dir / "status.json", {"status": "FORMAL_JINAN_3SEED_RUN_DONE", "steps": 600})
    _write_jsonl(
        run_dir / "train_metrics.jsonl",
        [
            {
                "step": 0,
                "loss": 0.25,
                "avg_travel_time": 12345.6789,
                "queue_length": 77.7,
            }
        ],
    )
    _write_jsonl(
        run_dir / "reward_components.jsonl",
        [{"step": 0, "total_reward": -999.5, "component_keys": ["x"]}],
    )
    _write_jsonl(run_dir / "loss_debug.jsonl", [{"total_loss": 0.5, "grad_norm": 2.0}])
    return root


def test_field_inventory_writes_only_allowed_outputs_without_metric_values(tmp_path: Path):
    root = _fake_formal_root(tmp_path)
    out_dir = tmp_path / "out"

    report = run_formal_result_field_inventory(
        root=root,
        out_dir=out_dir,
        approval_phrase=FORMAL_RESULT_FIELD_INVENTORY_APPROVAL_PHRASE,
    )

    assert report["report_status"] == "FORMAL_JINAN_RESULT_FIELD_INVENTORY_PASS"
    assert sorted(path.name for path in out_dir.iterdir()) == [
        "formal_jinan_result_field_inventory.json",
        "formal_jinan_result_field_inventory_packet.md",
    ]
    inventory = json.loads((out_dir / "formal_jinan_result_field_inventory.json").read_text(encoding="utf-8"))
    assert inventory["files"]["train_metrics.jsonl"]["row_count"] == 1
    assert "avg_travel_time" in inventory["files"]["train_metrics.jsonl"]["keys"]
    assert inventory["field_categories"]["avg_travel_time"] == "traffic_like_forbidden"

    output_text = "\n".join(path.read_text(encoding="utf-8") for path in out_dir.iterdir())
    assert "12345.6789" not in output_text
    assert "-999.5" not in output_text
    assert "77.7" not in output_text

    validate_field_inventory_packet(out_dir)


def test_field_inventory_rejects_wrong_phrase(tmp_path: Path):
    with pytest.raises(ValueError, match="exact Pro approval phrase"):
        run_formal_result_field_inventory(root=_fake_formal_root(tmp_path), out_dir=tmp_path / "out", approval_phrase="wrong")


def test_field_inventory_validator_rejects_forbidden_numeric_samples(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_json(
        out_dir / "formal_jinan_result_field_inventory.json",
        {
            "report_status": "FORMAL_JINAN_RESULT_FIELD_INVENTORY_PASS",
            "scope": "field_inventory_only_no_metric_values",
            "files": {},
            "field_categories": {},
            "numeric_values": {"avg_travel_time": 12345.6789},
        },
    )
    (out_dir / "formal_jinan_result_field_inventory_packet.md").write_text("packet", encoding="utf-8")

    with pytest.raises(ValueError, match="numeric_values"):
        validate_field_inventory_packet(out_dir)
