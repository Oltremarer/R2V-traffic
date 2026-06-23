from __future__ import annotations

import json
from pathlib import Path

import pytest

from pareto.rl.formal_training_stability_sanity import (
    FORMAL_TRAINING_STABILITY_APPROVAL_PHRASE,
    run_training_stability_sanity,
)
from pareto.rl.formal_training_stability_sanity_validator import validate_training_stability_packet

EXPECTED_METHODS = ["film_scalar_potential", "weighted_proxy", "env_reward"]
EXPECTED_SEEDS = [0, 1, 2]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _loss_rows(count: int = 480) -> list[dict]:
    row = {
        "approx_kl": 0.01,
        "grad_norm": 0.25,
        "policy_loss": -0.1,
        "value_loss": 0.2,
        "total_loss": 0.3,
        "entropy_bonus": 0.01,
        "clip_fraction": 0.0,
        "ratio_min": 0.8,
        "ratio_mean": 1.0,
        "ratio_max": 1.2,
    }
    return [dict(row) for _ in range(count)]


def _valid_run(seed: int = 0, method: str = "film_scalar_potential") -> dict:
    return {
        "seed": seed,
        "method": method,
        "status": "FORMAL_JINAN_3SEED_RUN_DONE",
        "pass_fail": "PASS",
        "loss_debug_rows": 480,
        "observed_row_count": 480,
        "missing_allowed_field_count": 0,
        "nonfinite_count": 0,
        "threshold_violation_count": 0,
        "explosion_flag": False,
    }


def _valid_runs() -> list[dict]:
    return [_valid_run(seed=seed, method=method) for seed in EXPECTED_SEEDS for method in EXPECTED_METHODS]


def _valid_totals() -> dict:
    return {
        "run_count": 9,
        "failed_run_count": 0,
        "warn_run_count": 0,
        "nonfinite_count": 0,
        "threshold_violation_count": 0,
        "missing_allowed_field_count": 0,
    }


def _valid_packet_payload() -> dict:
    return {
        "report_status": "FORMAL_JINAN_TRAINING_STABILITY_SANITY_PASS",
        "scope": "training_stability_sanity_only_no_method_comparison",
        "totals": _valid_totals(),
        "runs": _valid_runs(),
    }


def _fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    for seed in EXPECTED_SEEDS:
        for method in EXPECTED_METHODS:
            run_dir = root / f"seed{seed}" / method
            _write_json(run_dir / "metadata.json", {"method": method, "cityflow_seed": seed, "formal_experiment": True})
            _write_json(run_dir / "status.json", {"status": "FORMAL_JINAN_3SEED_RUN_DONE", "policy_update_count": 2})
            _write_jsonl(run_dir / "loss_debug.jsonl", _loss_rows())
    return root


def test_training_stability_outputs_counts_not_raw_values(tmp_path: Path):
    root = _fake_root(tmp_path)
    out_dir = tmp_path / "out"

    report = run_training_stability_sanity(
        root=root,
        out_dir=out_dir,
        approval_phrase=FORMAL_TRAINING_STABILITY_APPROVAL_PHRASE,
    )

    assert report["report_status"] == "FORMAL_JINAN_TRAINING_STABILITY_SANITY_PASS"
    assert sorted(path.name for path in out_dir.iterdir()) == [
        "formal_jinan_training_stability_sanity.json",
        "formal_jinan_training_stability_sanity_packet.md",
    ]
    payload = json.loads((out_dir / "formal_jinan_training_stability_sanity.json").read_text(encoding="utf-8"))
    run = payload["runs"][0]
    assert run["loss_debug_rows"] == 480
    assert run["nonfinite_count"] == 0
    assert run["threshold_violation_count"] == 0

    output_text = "\n".join(path.read_text(encoding="utf-8") for path in out_dir.iterdir())
    assert "0.25" not in output_text
    assert "1.2" not in output_text
    assert "ratio_max" not in output_text
    validate_training_stability_packet(out_dir)


def test_training_stability_rejects_wrong_phrase(tmp_path: Path):
    with pytest.raises(ValueError, match="exact Pro approval phrase"):
        run_training_stability_sanity(root=_fake_root(tmp_path), out_dir=tmp_path / "out", approval_phrase="wrong")


def test_training_stability_validator_rejects_raw_value_carrier(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    payload = _valid_packet_payload()
    payload["raw_values"] = {"grad_norm": 12345.6789}
    _write_json(out_dir / "formal_jinan_training_stability_sanity.json", payload)
    (out_dir / "formal_jinan_training_stability_sanity_packet.md").write_text("packet", encoding="utf-8")

    with pytest.raises(ValueError, match="raw_values"):
        validate_training_stability_packet(out_dir)


def test_training_stability_validator_rejects_nonzero_guard_counts(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    payload = _valid_packet_payload()
    payload["totals"]["threshold_violation_count"] = 1
    _write_json(out_dir / "formal_jinan_training_stability_sanity.json", payload)
    (out_dir / "formal_jinan_training_stability_sanity_packet.md").write_text("packet", encoding="utf-8")

    with pytest.raises(ValueError, match="threshold_violation_count"):
        validate_training_stability_packet(out_dir)


def test_training_stability_report_status_fails_on_guard_counts(tmp_path: Path):
    root = _fake_root(tmp_path)
    loss_path = root / "seed0" / "film_scalar_potential" / "loss_debug.jsonl"
    rows = [json.loads(line) for line in loss_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["ratio_max"] = 999.9
    _write_jsonl(loss_path, rows)
    out_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="training-stability sanity did not pass"):
        run_training_stability_sanity(
            root=root,
            out_dir=out_dir,
            approval_phrase=FORMAL_TRAINING_STABILITY_APPROVAL_PHRASE,
        )

    payload = json.loads((out_dir / "formal_jinan_training_stability_sanity.json").read_text(encoding="utf-8"))
    assert payload["report_status"] == "FORMAL_JINAN_TRAINING_STABILITY_SANITY_FAIL"
    assert payload["totals"]["threshold_violation_count"] == 1


def test_training_stability_report_status_fails_on_missing_run_coverage(tmp_path: Path):
    root = _fake_root(tmp_path)
    run_dir = root / "seed2" / "env_reward"
    for path in run_dir.iterdir():
        path.unlink()
    run_dir.rmdir()
    out_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="training-stability sanity did not pass"):
        run_training_stability_sanity(
            root=root,
            out_dir=out_dir,
            approval_phrase=FORMAL_TRAINING_STABILITY_APPROVAL_PHRASE,
        )

    payload = json.loads((out_dir / "formal_jinan_training_stability_sanity.json").read_text(encoding="utf-8"))
    assert payload["report_status"] == "FORMAL_JINAN_TRAINING_STABILITY_SANITY_FAIL"
    assert payload["totals"]["run_count"] == 8


def test_training_stability_validator_rejects_non_pass_run(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    payload = _valid_packet_payload()
    payload["runs"][0]["pass_fail"] = "WARN"
    _write_json(out_dir / "formal_jinan_training_stability_sanity.json", payload)
    (out_dir / "formal_jinan_training_stability_sanity_packet.md").write_text("packet", encoding="utf-8")

    with pytest.raises(ValueError, match="pass_fail"):
        validate_training_stability_packet(out_dir)


def test_training_stability_validator_rejects_partial_run_coverage(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    payload = _valid_packet_payload()
    payload["runs"] = payload["runs"][:-1]
    payload["totals"]["run_count"] = 8
    _write_json(out_dir / "formal_jinan_training_stability_sanity.json", payload)
    (out_dir / "formal_jinan_training_stability_sanity_packet.md").write_text("packet", encoding="utf-8")

    with pytest.raises(ValueError, match="run_count"):
        validate_training_stability_packet(out_dir)


def test_training_stability_validator_rejects_duplicate_method_seed_pair(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    payload = _valid_packet_payload()
    payload["runs"][1] = dict(payload["runs"][0])
    _write_json(out_dir / "formal_jinan_training_stability_sanity.json", payload)
    (out_dir / "formal_jinan_training_stability_sanity_packet.md").write_text("packet", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        validate_training_stability_packet(out_dir)


def test_training_stability_validator_rejects_missing_method_or_seed(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    payload = _valid_packet_payload()
    payload["runs"][0].pop("method")
    _write_json(out_dir / "formal_jinan_training_stability_sanity.json", payload)
    (out_dir / "formal_jinan_training_stability_sanity_packet.md").write_text("packet", encoding="utf-8")

    with pytest.raises(ValueError, match="method"):
        validate_training_stability_packet(out_dir)


def test_training_stability_validator_rejects_wrong_loss_debug_rows(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    payload = _valid_packet_payload()
    payload["runs"][0]["loss_debug_rows"] = 479
    _write_json(out_dir / "formal_jinan_training_stability_sanity.json", payload)
    (out_dir / "formal_jinan_training_stability_sanity_packet.md").write_text("packet", encoding="utf-8")

    with pytest.raises(ValueError, match="loss_debug_rows"):
        validate_training_stability_packet(out_dir)
