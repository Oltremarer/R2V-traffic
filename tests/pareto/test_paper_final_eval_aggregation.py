from __future__ import annotations

import json
from pathlib import Path

from pareto.rl import paper_final_eval_aggregation as aggregation
from pareto.rl.paper_final_learned_eval_runner import PAPER_FINAL_LEARNED_EVAL_DONE


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metrics(offset: float = 0.0) -> dict:
    return {
        "test_reward_over": -10.0 - offset,
        "test_avg_queue_len_over": 20.0 + offset,
        "test_queuing_vehicle_num_over": 200.0 + offset,
        "test_avg_waiting_time_over": 3.0 + offset,
        "test_avg_travel_time_over": 100.0 + offset,
    }


def test_aggregation_collects_reference_and_learned_eval_with_same_metric_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregation, "ROOT", tmp_path)
    ref_root = Path("records/paper_final/train_20260602_v1")
    learned_root = Path("records/paper_final/eval_20260602_v1")
    ref_dir = tmp_path / ref_root / "jinan" / "anon_3_4_jinan_real" / "Random" / "seed0"
    learned_dir = (
        tmp_path
        / learned_root
        / "jinan"
        / "anon_3_4_jinan_real"
        / "Cond-Scalar-RL"
        / "seed0"
        / "balanced"
    )
    _write_json(ref_dir / "paper_final_reference_metrics.json", _metrics(0.0))
    _write_json(ref_dir / "paper_final_reference_status.json", {"status": "PAPER_FINAL_REFERENCE_RUN_DONE"})
    _write_json(learned_dir / "paper_final_learned_eval_metrics.json", _metrics(5.0))
    _write_json(learned_dir / "paper_final_learned_eval_status.json", {"status": PAPER_FINAL_LEARNED_EVAL_DONE})

    payload = aggregation.build_paper_final_eval_aggregation(
        reference_root=ref_root,
        learned_eval_root=learned_root,
        cities=("jinan",),
    )

    assert payload["same_metric_schema_for_reference_and_learned"] is True
    assert payload["counts"]["reference_observed"] == 1
    assert payload["counts"]["learned_observed"] == 1
    assert payload["counts"]["reference_expected"] == 35
    assert payload["counts"]["learned_expected"] == 35
    assert payload["aggregation_ready"] is False
    summary = {
        (row["phase"], row["city"], row["method"], row["preference_template"]): row
        for row in payload["summary_rows"]
    }
    assert summary[("reference", "jinan", "Random", "not_applicable")]["test_avg_travel_time_over_mean"] == 100.0
    assert summary[("learned", "jinan", "Cond-Scalar-RL", "balanced")]["test_avg_travel_time_over_mean"] == 105.0
    assert any(row["phase"] == "learned" and row["reason"] == "missing_learned_eval_metrics" for row in payload["missing_rows"])


def test_aggregation_writes_jsonl_summary_csv_and_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregation, "ROOT", tmp_path)
    payload = {
        "raw_rows": [
            {
                "phase": "reference",
                "city": "jinan",
                "method": "Random",
                "preference_template": "not_applicable",
                "seed": 0,
                **_metrics(0.0),
            }
        ],
        "summary_rows": [
            {
                "phase": "reference",
                "city": "jinan",
                "method": "Random",
                "preference_template": "not_applicable",
                "n": 1,
                "seeds": [0],
                **{f"{key}_mean": value for key, value in _metrics(0.0).items()},
                **{f"{key}_std": 0.0 for key in _metrics(0.0)},
            }
        ],
        "missing_rows": [],
        "aggregation_ready": True,
        "counts": {},
    }

    aggregation.write_paper_final_eval_aggregation_outputs(payload, "records/paper_final/aggregation_test")

    out = tmp_path / "records/paper_final/aggregation_test"
    assert (out / "paper_final_eval_raw_rows.jsonl").is_file()
    assert (out / "paper_final_eval_summary.json").is_file()
    assert (out / "paper_final_eval_summary.csv").is_file()
    assert (out / "paper_final_eval_missing_rows.json").is_file()
