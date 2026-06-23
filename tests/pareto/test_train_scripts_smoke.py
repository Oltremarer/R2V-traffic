from argparse import Namespace
from pathlib import Path
import json

import pytest

torch = pytest.importorskip("torch")

from pareto.train_conditioned_scalar import train as train_scalar
from pareto.train_vector_quality import train as train_vector
from pareto.eval.run_offline_diagnostics import run as run_offline_diagnostics


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _record(split: str, idx: int) -> dict:
    value = float(idx)
    objectives = {
        "efficiency": value,
        "safety": -value,
        "fairness": 1.0 if idx % 2 == 0 else -1.0,
        "stability": -1.0 if idx % 2 == 0 else 1.0,
    }
    return {
        "sample_id": f"{split}_{idx}",
        "scenario": "toy",
        "obs_features": [value, value * 0.5, 1.0],
        "objective_values_norm": objectives,
        "objective_valid_mask": {key: True for key in objectives},
    }


def _pairs(split: str) -> dict[str, list[dict]]:
    a, b = f"{split}_3", f"{split}_0"
    return {
        "objective_pairs.jsonl": [
            {
                "pair_id": f"{split}_obj_eff",
                "split": split,
                "scenario": "toy",
                "a_id": a,
                "b_id": b,
                "objective": "efficiency",
                "label": 1,
                "is_tie": False,
                "margin_raw": 3.0,
                "margin_norm": 3.0,
                "source": "toy",
                "rule_version": "toy",
                "sampling_strategy": "objective_contrast",
            },
            {
                "pair_id": f"{split}_obj_safety",
                "split": split,
                "scenario": "toy",
                "a_id": a,
                "b_id": b,
                "objective": "safety",
                "label": 0,
                "is_tie": False,
                "margin_raw": -3.0,
                "margin_norm": -3.0,
                "source": "toy",
                "rule_version": "toy",
                "sampling_strategy": "objective_contrast",
            },
        ],
        "preference_pairs.jsonl": [
            {
                "pair_id": f"{split}_pref",
                "split": split,
                "scenario": "toy",
                "a_id": a,
                "b_id": b,
                "w": [0.7, 0.1, 0.1, 0.1],
                "label": 1,
                "is_tie": False,
                "source": "toy",
                "rule_utility_a": 1.0,
                "rule_utility_b": 0.0,
                "rule_margin": 1.0,
                "sampling_strategy": "preference_efficiency",
            }
        ],
        "dominance_pairs.jsonl": [
            {
                "pair_id": f"{split}_dom",
                "split": split,
                "scenario": "toy",
                "a_id": f"{split}_2",
                "b_id": f"{split}_1",
                "dominates": "a",
                "objective_margins_norm": {
                    "efficiency": 1.0,
                    "safety": 1.0,
                    "fairness": 1.0,
                    "stability": 1.0,
                },
                "min_margin_norm": 1.0,
                "dominance_threshold": 0.1,
                "source": "toy",
            }
        ],
        "reversal_pairs.jsonl": [
            {
                "pair_id": f"{split}_rev",
                "split": split,
                "scenario": "toy",
                "a_id": a,
                "b_id": b,
                "w_1_name": "efficiency",
                "w_1": [0.7, 0.1, 0.1, 0.1],
                "label_1": 1,
                "margin_1": 1.0,
                "w_2_name": "safety",
                "w_2": [0.1, 0.7, 0.1, 0.1],
                "label_2": 0,
                "margin_2": -1.0,
                "sampling_strategy": "reversal",
            }
        ],
    }


def _make_fixture(root: Path) -> tuple[Path, Path]:
    records_root = root / "records"
    pairs_root = root / "pairs"
    for split in ("train", "val", "test"):
        _write_jsonl(records_root / f"{split}_raw.jsonl", [_record(split, idx) for idx in range(4)])
        for name, rows in _pairs(split).items():
            _write_jsonl(pairs_root / split / name, rows)
    return records_root, pairs_root


def _args(records_root: Path, pairs_root: Path, output_dir: Path) -> Namespace:
    return Namespace(
        records_root=str(records_root),
        pairs_root=str(pairs_root),
        output_dir=str(output_dir),
        epochs=1,
        batch_size=2,
        seed=0,
        device="cpu",
        hidden_dim=8,
        num_layers=2,
        dropout=0.0,
        lr=1e-3,
        objective_loss_weight=1.0,
        preference_loss_weight=1.0,
        dominance_loss_weight=0.2,
        calibration_loss_weight=0.01,
        dominance_margin=0.1,
        score_mode="linear",
        interaction_rank=4,
        interaction_beta=0.3,
        interaction_l2=0.0,
        isotonic_dominance_weight=0.0,
        isotonic_margin_floor=0.05,
        use_objective_margins_for_dominance=False,
    )


def test_vector_training_script_writes_expected_artifacts(tmp_path: Path):
    records_root, pairs_root = _make_fixture(tmp_path)
    out_dir = tmp_path / "vector"

    metadata = train_vector(_args(records_root, pairs_root, out_dir))

    assert metadata["model_type"] == "VectorQualityNet"
    assert metadata["param_count"] > 0
    assert metadata["score_mode"] == "linear"
    assert (out_dir / "model.pt").exists()
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "train_log.jsonl").exists()
    assert (out_dir / "diagnostics_val.json").exists()
    assert (out_dir / "diagnostics_test.json").exists()


def test_conditioned_scalar_training_script_writes_expected_artifacts(tmp_path: Path):
    records_root, pairs_root = _make_fixture(tmp_path)
    out_dir = tmp_path / "scalar"

    metadata = train_scalar(_args(records_root, pairs_root, out_dir))

    assert metadata["model_type"] == "ConditionedScalarQualityNet"
    assert metadata["param_count"] > 0
    assert (out_dir / "model.pt").exists()
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "train_log.jsonl").exists()
    assert (out_dir / "diagnostics_val.json").exists()
    assert (out_dir / "diagnostics_test.json").exists()


def test_offline_diagnostics_runner_compares_vector_and_scalar(tmp_path: Path):
    records_root, pairs_root = _make_fixture(tmp_path)
    vector_dir = tmp_path / "vector"
    scalar_dir = tmp_path / "scalar"
    train_vector(_args(records_root, pairs_root, vector_dir))
    train_scalar(_args(records_root, pairs_root, scalar_dir))
    out_path = tmp_path / "offline_diagnostics.json"

    payload = run_offline_diagnostics(Namespace(
        records_root=str(records_root),
        pairs_root=str(pairs_root),
        vector_model_dir=str(vector_dir),
        scalar_model_dir=str(scalar_dir),
        out=str(out_path),
        device="cpu",
    ))

    assert out_path.exists()
    assert "vector" in payload
    assert "cond_scalar" in payload
    assert "test" in payload["vector"]
    assert "decision" in payload
    assert payload["decision"]["recommended_next_step"] in {
        "debug_model",
        "run_more_offline",
        "stop_before_ppo",
    }


def test_vector_training_accepts_low_rank_scorer_and_isotonic_loss(tmp_path: Path):
    records_root, pairs_root = _make_fixture(tmp_path)
    out_dir = tmp_path / "vector_low_rank"
    args = _args(records_root, pairs_root, out_dir)
    args.score_mode = "low_rank_interaction"
    args.interaction_rank = 2
    args.interaction_beta = 0.3
    args.interaction_l2 = 0.0
    args.isotonic_dominance_weight = 0.25
    args.use_objective_margins_for_dominance = True

    metadata = train_vector(args)

    assert metadata["score_mode"] == "low_rank_interaction"
    assert metadata["scorer_param_count"] > 0
    checkpoint = torch.load(out_dir / "model.pt", map_location="cpu")
    assert checkpoint["scorer_config"]["score_mode"] == "low_rank_interaction"
    assert "scorer_state_dict" in checkpoint
