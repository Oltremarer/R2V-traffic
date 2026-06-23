#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from pareto.constants import OBJECTIVE_NAMES
from pareto.data.offline_dataset import (
    build_reversal_training_pairs,
    dominance_pair_tensors,
    infer_input_dim,
    load_split_pairs,
    load_split_records,
    make_feature_tensor,
    objective_pair_tensors,
    preference_pair_tensors,
    reversal_pair_tensors,
)
from pareto.eval.offline_metrics import EVAL_PREFERENCES, evaluate_vector_model, load_split_bundle
from pareto.losses.preference_losses import (
    calibration_loss,
    dominance_loss,
    dominance_utility_loss,
    isotonic_dominance_loss,
    objective_pair_loss,
    preference_hinge_margin_loss,
    preference_margin_regression_loss,
    preference_pair_loss,
    reversal_hinge_margin_loss,
    reversal_margin_regression_loss,
    reversal_pair_loss,
)
from pareto.models.vector_quality import build_preference_scorer, build_vector_quality_model
from pareto.train_common import (
    append_jsonl,
    count_parameters,
    move_batch,
    resolve_device,
    set_seed,
    shuffled_batches,
    tensor_batch,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records_root", required=True)
    parser.add_argument("--pairs_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--architecture", choices=["shared_mlp", "per_head_tower", "residual_tower"], default="shared_mlp")
    parser.add_argument("--trunk_layers", type=int, default=2)
    parser.add_argument("--head_layers", type=int, default=2)
    parser.add_argument("--tower_residual_alpha", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--objective_loss_weight", type=float, default=1.0)
    parser.add_argument("--preference_loss_weight", type=float, default=1.0)
    parser.add_argument("--reversal_loss_weight", type=float, default=0.0)
    parser.add_argument("--dominance_loss_weight", type=float, default=None)
    parser.add_argument("--dominance_coord_loss_weight", type=float, default=0.2)
    parser.add_argument("--dominance_utility_loss_weight", type=float, default=0.0)
    parser.add_argument("--dominance_margin", type=float, default=0.1)
    parser.add_argument("--calibration_loss_weight", type=float, default=0.01)
    parser.add_argument("--objective_weights", default="")
    parser.add_argument("--training_schedule", choices=["sequential", "joint"], default="sequential")
    parser.add_argument("--reversal_min_margin", type=float, default=0.0)
    parser.add_argument("--reversal_sampler", choices=["uniform", "template_balanced"], default="uniform")
    parser.add_argument("--reversal_template_min_count", type=int, default=20)
    parser.add_argument("--pref_margin_loss_weight", type=float, default=0.0)
    parser.add_argument("--rev_margin_loss_weight", type=float, default=0.0)
    parser.add_argument("--pref_hinge_loss_weight", type=float, default=0.0)
    parser.add_argument("--rev_hinge_loss_weight", type=float, default=0.0)
    parser.add_argument("--classification_margin", type=float, default=0.5)
    parser.add_argument("--margin_clip", type=float, default=2.0)
    parser.add_argument("--score_mode", choices=["linear", "low_rank_interaction"], default="linear")
    parser.add_argument("--interaction_rank", type=int, default=4)
    parser.add_argument("--interaction_beta", type=float, default=0.3)
    parser.add_argument("--interaction_l2", type=float, default=0.0)
    parser.add_argument("--isotonic_dominance_weight", type=float, default=0.0)
    parser.add_argument("--isotonic_margin_floor", type=float, default=0.05)
    parser.add_argument("--use_objective_margins_for_dominance", action="store_true")
    return parser.parse_args()


def _filter_reversal_pairs(pairs: list[dict], min_margin: float) -> list[dict]:
    if min_margin <= 0:
        return pairs
    return [
        pair for pair in pairs
        if min(abs(float(pair.get("margin_1", 0.0))), abs(float(pair.get("margin_2", 0.0)))) >= min_margin
    ]


def _prepare_train_tensors(
    records_root: str,
    pairs_root: str,
    device: torch.device,
    reversal_min_margin: float = 0.0,
    reversal_sampler: str = "uniform",
    reversal_template_min_count: int = 20,
    seed: int = 0,
) -> tuple[dict, dict]:
    records = load_split_records(records_root, "train")
    pairs = load_split_pairs(pairs_root, "train")
    filtered_reversal_pairs = _filter_reversal_pairs(pairs["reversal"], reversal_min_margin)
    reversal_pairs, reversal_sampler_report = build_reversal_training_pairs(
        filtered_reversal_pairs,
        sampler=reversal_sampler,
        min_count=reversal_template_min_count,
        seed=seed,
    )
    tensors = {
        "objective": move_batch(objective_pair_tensors(pairs["objective"], records), device) if pairs["objective"] else None,
        "preference": move_batch(preference_pair_tensors(pairs["preference"], records), device) if pairs["preference"] else None,
        "dominance": move_batch(dominance_pair_tensors(pairs["dominance"], records), device) if pairs["dominance"] else None,
        "reversal": move_batch(reversal_pair_tensors(reversal_pairs, records), device) if reversal_pairs else None,
        "records_x": make_feature_tensor(records, list(records)).to(device),
        "reversal_pair_count_raw": len(pairs["reversal"]),
        "reversal_pair_count_filtered": len(filtered_reversal_pairs),
        "reversal_pair_count_used": len(reversal_pairs),
        "reversal_sampler_report": reversal_sampler_report,
    }
    return records, tensors


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _parse_objective_weights(payload: str, device: torch.device) -> torch.Tensor | None:
    if not payload:
        return None
    values = {name: 1.0 for name in OBJECTIVE_NAMES}
    for item in payload.split(","):
        if not item.strip():
            continue
        name, raw_value = item.split(":", 1)
        name = name.strip()
        if name not in values:
            raise ValueError(f"unknown objective weight key: {name}")
        values[name] = float(raw_value)
    return torch.tensor([values[name] for name in OBJECTIVE_NAMES], dtype=torch.float32, device=device)


def _batch_lists(tensors: dict, batch_size: int, device: torch.device) -> dict[str, list[torch.Tensor]]:
    batches = {}
    for name in ("objective", "preference", "reversal", "dominance"):
        batch = tensors.get(name)
        if batch is None:
            batches[name] = []
            continue
        first_tensor = next(iter(batch.values()))
        batches[name] = shuffled_batches(first_tensor.shape[0], batch_size, device)
    return batches


def _take_batch(tensors: dict, batches: dict[str, list[torch.Tensor]], name: str, step: int) -> dict[str, torch.Tensor] | None:
    if not batches[name]:
        return None
    return tensor_batch(tensors[name], batches[name][step % len(batches[name])])


def _module_l2(module: torch.nn.Module, device: torch.device) -> torch.Tensor:
    loss = torch.tensor(0.0, device=device)
    for parameter in module.parameters():
        loss = loss + parameter.pow(2).mean()
    return loss


def train(args: argparse.Namespace) -> dict:
    for key, value in {
        "objective_weights": "",
        "training_schedule": "sequential",
        "reversal_loss_weight": 0.0,
        "dominance_coord_loss_weight": None,
        "dominance_utility_loss_weight": 0.0,
        "dominance_margin": 0.1,
        "calibration_loss_weight": 0.01,
        "architecture": "shared_mlp",
        "trunk_layers": 2,
        "head_layers": 2,
        "tower_residual_alpha": 0.5,
        "reversal_min_margin": 0.0,
        "reversal_sampler": "uniform",
        "reversal_template_min_count": 20,
        "pref_margin_loss_weight": 0.0,
        "rev_margin_loss_weight": 0.0,
        "pref_hinge_loss_weight": 0.0,
        "rev_hinge_loss_weight": 0.0,
        "classification_margin": 0.5,
        "margin_clip": 2.0,
        "score_mode": "linear",
        "interaction_rank": 4,
        "interaction_beta": 0.3,
        "interaction_l2": 0.0,
        "isotonic_dominance_weight": 0.0,
        "isotonic_margin_floor": 0.05,
        "use_objective_margins_for_dominance": False,
    }.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_log = out_dir / "train_log.jsonl"
    if train_log.exists():
        train_log.unlink()

    train_records, tensors = _prepare_train_tensors(
        args.records_root,
        args.pairs_root,
        device,
        args.reversal_min_margin,
        args.reversal_sampler,
        args.reversal_template_min_count,
        args.seed,
    )
    input_dim = infer_input_dim(train_records)
    config = {
        "model_type": "VectorQualityNet",
        "architecture": args.architecture,
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "trunk_layers": args.trunk_layers,
        "head_layers": args.head_layers,
        "tower_residual_alpha": args.tower_residual_alpha,
        "dropout": args.dropout,
        "objective_order": list(OBJECTIVE_NAMES),
        "score_mode": args.score_mode,
        "interaction_rank": args.interaction_rank,
        "interaction_beta": args.interaction_beta,
    }
    model = build_vector_quality_model(
        args.architecture,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        trunk_layers=args.trunk_layers,
        head_layers=args.head_layers,
        tower_residual_alpha=args.tower_residual_alpha,
    ).to(device)
    scorer = build_preference_scorer(
        args.score_mode,
        rank=args.interaction_rank,
        beta=args.interaction_beta,
        num_objectives=len(OBJECTIVE_NAMES),
    ).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(scorer.parameters()), lr=args.lr)
    objective_weights = _parse_objective_weights(getattr(args, "objective_weights", ""), device)
    coord_weight = getattr(args, "dominance_coord_loss_weight", None)
    if coord_weight is None:
        coord_weight = getattr(args, "dominance_loss_weight", 0.2)
    if getattr(args, "dominance_loss_weight", None) is not None:
        coord_weight = args.dominance_loss_weight
    eval_preferences = EVAL_PREFERENCES.to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = {
            "objective": [],
            "preference": [],
            "preference_margin": [],
            "preference_hinge": [],
            "reversal": [],
            "reversal_margin": [],
            "reversal_hinge": [],
            "dominance_coord": [],
            "dominance_utility": [],
            "isotonic_dominance": [],
            "calibration": [],
        }

        if getattr(args, "training_schedule", "sequential") == "joint":
            batches = _batch_lists(tensors, args.batch_size, device)
            steps_per_epoch = max([len(items) for items in batches.values()] + [1])
            for step in range(steps_per_epoch):
                optimizer.zero_grad()
                loss = torch.tensor(0.0, device=device)
                q_cal_parts = []

                batch = _take_batch(tensors, batches, "objective", step)
                if batch is not None:
                    q_a = model(batch["x_a"])
                    q_b = model(batch["x_b"])
                    obj_loss = objective_pair_loss(q_a, q_b, batch["objective_idx"], batch["labels"], objective_weights)
                    loss = loss + args.objective_loss_weight * obj_loss
                    losses["objective"].append(float(obj_loss.detach().cpu()))
                    q_cal_parts.extend([q_a, q_b])

                batch = _take_batch(tensors, batches, "preference", step)
                if batch is not None:
                    q_a = model(batch["x_a"])
                    q_b = model(batch["x_b"])
                    pref_loss = preference_pair_loss(q_a, q_b, batch["w"], batch["labels"], scorer=scorer)
                    loss = loss + args.preference_loss_weight * pref_loss
                    losses["preference"].append(float(pref_loss.detach().cpu()))
                    if args.pref_margin_loss_weight > 0:
                        margin_loss = preference_margin_regression_loss(
                            q_a,
                            q_b,
                            batch["w"],
                            batch["rule_margin"],
                            clip=args.margin_clip,
                            scorer=scorer,
                        )
                        loss = loss + args.pref_margin_loss_weight * margin_loss
                        losses["preference_margin"].append(float(margin_loss.detach().cpu()))
                    if args.pref_hinge_loss_weight > 0:
                        hinge_loss = preference_hinge_margin_loss(
                            q_a,
                            q_b,
                            batch["w"],
                            batch["labels"],
                            margin=args.classification_margin,
                            scorer=scorer,
                        )
                        loss = loss + args.pref_hinge_loss_weight * hinge_loss
                        losses["preference_hinge"].append(float(hinge_loss.detach().cpu()))
                    q_cal_parts.extend([q_a, q_b])

                batch = _take_batch(tensors, batches, "reversal", step)
                if batch is not None and args.reversal_loss_weight > 0:
                    q_a = model(batch["x_a"])
                    q_b = model(batch["x_b"])
                    rev_loss = reversal_pair_loss(
                        q_a,
                        q_b,
                        batch["w_1"],
                        batch["w_2"],
                        batch["labels_1"],
                        batch["labels_2"],
                        scorer=scorer,
                    )
                    loss = loss + args.reversal_loss_weight * rev_loss
                    losses["reversal"].append(float(rev_loss.detach().cpu()))
                    if args.rev_margin_loss_weight > 0:
                        margin_loss = reversal_margin_regression_loss(
                            q_a,
                            q_b,
                            batch["w_1"],
                            batch["w_2"],
                            batch["margin_1"],
                            batch["margin_2"],
                            clip=args.margin_clip,
                            scorer=scorer,
                        )
                        loss = loss + args.rev_margin_loss_weight * margin_loss
                        losses["reversal_margin"].append(float(margin_loss.detach().cpu()))
                    if args.rev_hinge_loss_weight > 0:
                        hinge_loss = reversal_hinge_margin_loss(
                            q_a,
                            q_b,
                            batch["w_1"],
                            batch["w_2"],
                            batch["labels_1"],
                            batch["labels_2"],
                            margin=args.classification_margin,
                            scorer=scorer,
                        )
                        loss = loss + args.rev_hinge_loss_weight * hinge_loss
                        losses["reversal_hinge"].append(float(hinge_loss.detach().cpu()))
                    q_cal_parts.extend([q_a, q_b])

                batch = _take_batch(tensors, batches, "dominance", step)
                if batch is not None:
                    q_dom = model(batch["x_dom"])
                    q_sub = model(batch["x_sub"])
                    if coord_weight > 0:
                        dom_coord = dominance_loss(q_dom, q_sub, margin=args.dominance_margin)
                        loss = loss + coord_weight * dom_coord
                        losses["dominance_coord"].append(float(dom_coord.detach().cpu()))
                    if args.dominance_utility_loss_weight > 0:
                        dom_util = dominance_utility_loss(q_dom, q_sub, eval_preferences, margin=args.dominance_margin)
                        loss = loss + args.dominance_utility_loss_weight * dom_util
                        losses["dominance_utility"].append(float(dom_util.detach().cpu()))
                    if args.isotonic_dominance_weight > 0:
                        margins = batch["objective_margins"] if args.use_objective_margins_for_dominance else None
                        iso_loss = isotonic_dominance_loss(
                            q_dom,
                            q_sub,
                            margins,
                            margin_floor=args.isotonic_margin_floor,
                        )
                        loss = loss + args.isotonic_dominance_weight * iso_loss
                        losses["isotonic_dominance"].append(float(iso_loss.detach().cpu()))
                    q_cal_parts.extend([q_dom, q_sub])

                if q_cal_parts and args.calibration_loss_weight > 0:
                    cal_loss = calibration_loss(torch.cat(q_cal_parts, dim=0))
                    loss = loss + args.calibration_loss_weight * cal_loss
                    losses["calibration"].append(float(cal_loss.detach().cpu()))
                if args.interaction_l2 > 0 and count_parameters(scorer) > 0:
                    loss = loss + args.interaction_l2 * _module_l2(scorer, device)
                loss.backward()
                optimizer.step()
        else:
            objective = tensors["objective"]
            if objective is not None:
                for indices in shuffled_batches(objective["labels"].shape[0], args.batch_size, device):
                    batch = tensor_batch(objective, indices)
                    q_a = model(batch["x_a"])
                    q_b = model(batch["x_b"])
                    obj_loss = objective_pair_loss(q_a, q_b, batch["objective_idx"], batch["labels"], objective_weights)
                    cal_loss = calibration_loss(torch.cat([q_a, q_b], dim=0))
                    loss = args.objective_loss_weight * obj_loss + args.calibration_loss_weight * cal_loss
                    if args.interaction_l2 > 0 and count_parameters(scorer) > 0:
                        loss = loss + args.interaction_l2 * _module_l2(scorer, device)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses["objective"].append(float(obj_loss.detach().cpu()))
                    losses["calibration"].append(float(cal_loss.detach().cpu()))

            preference = tensors["preference"]
            if preference is not None:
                for indices in shuffled_batches(preference["labels"].shape[0], args.batch_size, device):
                    batch = tensor_batch(preference, indices)
                    q_a = model(batch["x_a"])
                    q_b = model(batch["x_b"])
                    pref_loss = preference_pair_loss(q_a, q_b, batch["w"], batch["labels"], scorer=scorer)
                    loss = args.preference_loss_weight * pref_loss
                    if args.pref_margin_loss_weight > 0:
                        margin_loss = preference_margin_regression_loss(
                            q_a,
                            q_b,
                            batch["w"],
                            batch["rule_margin"],
                            clip=args.margin_clip,
                            scorer=scorer,
                        )
                        loss = loss + args.pref_margin_loss_weight * margin_loss
                        losses["preference_margin"].append(float(margin_loss.detach().cpu()))
                    if args.pref_hinge_loss_weight > 0:
                        hinge_loss = preference_hinge_margin_loss(
                            q_a,
                            q_b,
                            batch["w"],
                            batch["labels"],
                            margin=args.classification_margin,
                            scorer=scorer,
                        )
                        loss = loss + args.pref_hinge_loss_weight * hinge_loss
                        losses["preference_hinge"].append(float(hinge_loss.detach().cpu()))
                    if args.interaction_l2 > 0 and count_parameters(scorer) > 0:
                        loss = loss + args.interaction_l2 * _module_l2(scorer, device)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses["preference"].append(float(pref_loss.detach().cpu()))

            reversal = tensors["reversal"]
            if reversal is not None and args.reversal_loss_weight > 0:
                for indices in shuffled_batches(reversal["labels_1"].shape[0], args.batch_size, device):
                    batch = tensor_batch(reversal, indices)
                    q_a = model(batch["x_a"])
                    q_b = model(batch["x_b"])
                    rev_loss = reversal_pair_loss(
                        q_a,
                        q_b,
                        batch["w_1"],
                        batch["w_2"],
                        batch["labels_1"],
                        batch["labels_2"],
                        scorer=scorer,
                    )
                    loss = args.reversal_loss_weight * rev_loss
                    if args.rev_margin_loss_weight > 0:
                        margin_loss = reversal_margin_regression_loss(
                            q_a,
                            q_b,
                            batch["w_1"],
                            batch["w_2"],
                            batch["margin_1"],
                            batch["margin_2"],
                            clip=args.margin_clip,
                            scorer=scorer,
                        )
                        loss = loss + args.rev_margin_loss_weight * margin_loss
                        losses["reversal_margin"].append(float(margin_loss.detach().cpu()))
                    if args.rev_hinge_loss_weight > 0:
                        hinge_loss = reversal_hinge_margin_loss(
                            q_a,
                            q_b,
                            batch["w_1"],
                            batch["w_2"],
                            batch["labels_1"],
                            batch["labels_2"],
                            margin=args.classification_margin,
                            scorer=scorer,
                        )
                        loss = loss + args.rev_hinge_loss_weight * hinge_loss
                        losses["reversal_hinge"].append(float(hinge_loss.detach().cpu()))
                    if args.interaction_l2 > 0 and count_parameters(scorer) > 0:
                        loss = loss + args.interaction_l2 * _module_l2(scorer, device)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses["reversal"].append(float(rev_loss.detach().cpu()))

            dominance = tensors["dominance"]
            if dominance is not None:
                for indices in shuffled_batches(dominance["x_dom"].shape[0], args.batch_size, device):
                    batch = tensor_batch(dominance, indices)
                    q_dom = model(batch["x_dom"])
                    q_sub = model(batch["x_sub"])
                    loss = torch.tensor(0.0, device=device)
                    if coord_weight > 0:
                        dom_coord = dominance_loss(q_dom, q_sub, margin=args.dominance_margin)
                        loss = loss + coord_weight * dom_coord
                        losses["dominance_coord"].append(float(dom_coord.detach().cpu()))
                    if args.dominance_utility_loss_weight > 0:
                        dom_util = dominance_utility_loss(q_dom, q_sub, eval_preferences, margin=args.dominance_margin)
                        loss = loss + args.dominance_utility_loss_weight * dom_util
                        losses["dominance_utility"].append(float(dom_util.detach().cpu()))
                    if args.isotonic_dominance_weight > 0:
                        margins = batch["objective_margins"] if args.use_objective_margins_for_dominance else None
                        iso_loss = isotonic_dominance_loss(
                            q_dom,
                            q_sub,
                            margins,
                            margin_floor=args.isotonic_margin_floor,
                        )
                        loss = loss + args.isotonic_dominance_weight * iso_loss
                        losses["isotonic_dominance"].append(float(iso_loss.detach().cpu()))
                    if args.interaction_l2 > 0 and count_parameters(scorer) > 0:
                        loss = loss + args.interaction_l2 * _module_l2(scorer, device)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        row = {
            "epoch": epoch,
            "objective_loss": _mean(losses["objective"]),
            "preference_loss": _mean(losses["preference"]),
            "preference_margin_loss": _mean(losses["preference_margin"]),
            "preference_hinge_loss": _mean(losses["preference_hinge"]),
            "reversal_loss": _mean(losses["reversal"]),
            "reversal_margin_loss": _mean(losses["reversal_margin"]),
            "reversal_hinge_loss": _mean(losses["reversal_hinge"]),
            "dominance_coord_loss": _mean(losses["dominance_coord"]),
            "dominance_utility_loss": _mean(losses["dominance_utility"]),
            "isotonic_dominance_loss": _mean(losses["isotonic_dominance"]),
            "calibration_loss": _mean(losses["calibration"]),
        }
        append_jsonl(train_log, row)

    metadata = {
        "model_type": "VectorQualityNet",
        "architecture": args.architecture,
        "param_count": count_parameters(model) + count_parameters(scorer),
        "model_param_count": count_parameters(model),
        "scorer_param_count": count_parameters(scorer),
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "trunk_layers": args.trunk_layers,
        "head_layers": args.head_layers,
        "tower_residual_alpha": args.tower_residual_alpha,
        "dropout": args.dropout,
        "objective_order": list(OBJECTIVE_NAMES),
        "score_mode": args.score_mode,
        "interaction_rank": args.interaction_rank,
        "interaction_beta": args.interaction_beta,
        "interaction_l2": args.interaction_l2,
        "records_root": args.records_root,
        "pairs_root": args.pairs_root,
        "seed": args.seed,
        "requested_device": args.device,
        "actual_device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "objective_weights": args.objective_weights,
        "training_schedule": args.training_schedule,
        "reversal_min_margin": args.reversal_min_margin,
        "reversal_sampler": args.reversal_sampler,
        "reversal_template_min_count": args.reversal_template_min_count,
        "reversal_pair_count_raw": tensors["reversal_pair_count_raw"],
        "reversal_pair_count_filtered": tensors["reversal_pair_count_filtered"],
        "reversal_pair_count_used": tensors["reversal_pair_count_used"],
        "reversal_sampler_report": tensors["reversal_sampler_report"],
        "reversal_loss_weight": args.reversal_loss_weight,
        "pref_margin_loss_weight": args.pref_margin_loss_weight,
        "rev_margin_loss_weight": args.rev_margin_loss_weight,
        "pref_hinge_loss_weight": args.pref_hinge_loss_weight,
        "rev_hinge_loss_weight": args.rev_hinge_loss_weight,
        "classification_margin": args.classification_margin,
        "margin_clip": args.margin_clip,
        "dominance_coord_loss_weight": coord_weight,
        "dominance_utility_loss_weight": args.dominance_utility_loss_weight,
        "isotonic_dominance_weight": args.isotonic_dominance_weight,
        "isotonic_margin_floor": args.isotonic_margin_floor,
        "use_objective_margins_for_dominance": args.use_objective_margins_for_dominance,
        "dominance_margin": args.dominance_margin,
        "calibration_loss_weight": args.calibration_loss_weight,
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "scorer_state_dict": scorer.state_dict(),
            "scorer_config": {
                "score_mode": args.score_mode,
                "interaction_rank": args.interaction_rank,
                "interaction_beta": args.interaction_beta,
                "interaction_l2": args.interaction_l2,
            },
        },
        out_dir / "model.pt",
    )
    write_json(out_dir / "config.json", config)
    write_json(out_dir / "metadata.json", metadata)

    for split in ("val", "test"):
        records, pairs = load_split_bundle(args.records_root, args.pairs_root, split)
        diagnostics = evaluate_vector_model(model, records, pairs, device=device, scorer=scorer)
        write_json(out_dir / f"diagnostics_{split}.json", diagnostics)

    return metadata


def main() -> None:
    metadata = train(parse_args())
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
