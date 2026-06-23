#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from pareto.eval.offline_metrics import evaluate_scalar_model, evaluate_vector_model, load_split_bundle
from pareto.models.conditioned_scalar import build_conditioned_scalar_model
from pareto.models.vector_quality import build_preference_scorer, build_vector_quality_model
from pareto.train_common import load_checkpoint, resolve_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records_root", required=True)
    parser.add_argument("--pairs_root", required=True)
    parser.add_argument("--vector_model_dir", required=True)
    parser.add_argument("--scalar_model_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_vector_model(model_dir: str | Path, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, dict]:
    checkpoint = load_checkpoint(Path(model_dir) / "model.pt", device)
    config = checkpoint["config"]
    model = build_vector_quality_model(
        config.get("architecture", "shared_mlp"),
        input_dim=config["input_dim"],
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 3),
        dropout=config.get("dropout", 0.0),
        trunk_layers=config.get("trunk_layers", 2),
        head_layers=config.get("head_layers", 2),
        tower_residual_alpha=config.get("tower_residual_alpha", 0.5),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    scorer_config = checkpoint.get("scorer_config", {
        "score_mode": config.get("score_mode", "linear"),
        "interaction_rank": config.get("interaction_rank", 4),
        "interaction_beta": config.get("interaction_beta", 0.3),
    })
    scorer = build_preference_scorer(
        scorer_config.get("score_mode", "linear"),
        rank=scorer_config.get("interaction_rank", 4),
        beta=scorer_config.get("interaction_beta", 0.3),
    )
    if "scorer_state_dict" in checkpoint:
        scorer.load_state_dict(checkpoint["scorer_state_dict"])
    scorer.to(device)
    scorer.eval()
    return model, scorer, config


def _load_scalar_model(model_dir: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = load_checkpoint(Path(model_dir) / "model.pt", device)
    config = checkpoint["config"]
    model = build_conditioned_scalar_model(
        config.get("architecture", "concat"),
        input_dim=config["input_dim"],
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 3),
        dropout=config.get("dropout", 0.0),
        preference_dim=config.get("preference_dim", 4),
        film_layers=config.get("film_layers", 2),
        head_layers=config.get("head_layers", 2),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, config


def _vector_gate(test_metrics: dict) -> str:
    std_ok = min(test_metrics.get("output_std_per_head", [0.0])) > 1e-3
    checks = [
        test_metrics.get("obj_acc_mean", 0.0) > 0.62,
        test_metrics.get("pref_acc", 0.0) > 0.65,
        test_metrics.get("rev_acc", 0.0) > 0.60,
        test_metrics.get("dpr_head", 0.0) > 0.85,
        test_metrics.get("dpr_utility", 0.0) > 0.85,
        test_metrics.get("head_leakage_diag_offdiag_gap", 0.0) >= 0.08,
        std_ok,
    ]
    return "pass" if all(checks) else "fail"


def _condscalar_danger(vector_test: dict, scalar_test: dict) -> str:
    pref_margin = scalar_test.get("pref_acc", 0.0) - vector_test.get("pref_acc", 0.0)
    rev_margin = scalar_test.get("rev_acc", 0.0) - vector_test.get("rev_acc", 0.0)
    if pref_margin >= 0.03 and rev_margin >= 0.03:
        return "fail"
    if pref_margin >= 0.01 and rev_margin >= 0.01:
        return "warn"
    return "pass"


def _next_step(vector_gate: str, cond_danger: str) -> str:
    if cond_danger == "fail":
        return "stop_before_ppo"
    if vector_gate == "pass":
        return "run_more_offline"
    return "debug_model"


def run(args: argparse.Namespace) -> dict:
    device = resolve_device(args.device)
    vector_model, vector_scorer, vector_config = _load_vector_model(args.vector_model_dir, device)
    scalar_model, scalar_config = _load_scalar_model(args.scalar_model_dir, device)

    vector_metrics = {}
    scalar_metrics = {}
    for split in ("train", "val", "test"):
        records, pairs = load_split_bundle(args.records_root, args.pairs_root, split)
        vector_metrics[split] = evaluate_vector_model(vector_model, records, pairs, device=device, scorer=vector_scorer)
        scalar_metrics[split] = evaluate_scalar_model(scalar_model, records, pairs, device=device)

    vector_gate = _vector_gate(vector_metrics["test"])
    cond_danger = _condscalar_danger(vector_metrics["test"], scalar_metrics["test"])
    payload = {
        "records_root": args.records_root,
        "pairs_root": args.pairs_root,
        "vector_model_dir": args.vector_model_dir,
        "scalar_model_dir": args.scalar_model_dir,
        "device": str(device),
        "vector_config": vector_config,
        "scalar_config": scalar_config,
        "vector": vector_metrics,
        "cond_scalar": scalar_metrics,
        "decision": {
            "vectorq_representation_gate": vector_gate,
            "condscalar_danger": cond_danger,
            "recommended_next_step": _next_step(vector_gate, cond_danger),
        },
    }
    write_json(args.out, payload)
    return payload


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
