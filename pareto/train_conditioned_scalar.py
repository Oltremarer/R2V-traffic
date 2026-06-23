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
import torch.nn.functional as F

from pareto.constants import OBJECTIVE_NAMES
from pareto.data.offline_dataset import (
    build_reversal_training_pairs,
    dominance_pair_tensors,
    infer_input_dim,
    load_split_pairs,
    load_split_records,
    preference_pair_tensors,
    reversal_pair_tensors,
)
from pareto.eval.offline_metrics import EVAL_PREFERENCES, evaluate_scalar_model, load_split_bundle
from pareto.models.conditioned_scalar import build_conditioned_scalar_model
from pareto.train_common import (
    append_jsonl,
    count_parameters,
    move_batch,
    resolve_device,
    save_model_checkpoint,
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
    parser.add_argument("--architecture", choices=["concat", "film"], default="concat")
    parser.add_argument("--film_layers", type=int, default=2)
    parser.add_argument("--head_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--preference_loss_weight", type=float, default=1.0)
    parser.add_argument("--reversal_loss_weight", type=float, default=0.0)
    parser.add_argument("--dominance_loss_weight", type=float, default=0.2)
    parser.add_argument("--dominance_margin", type=float, default=0.1)
    parser.add_argument("--training_schedule", choices=["sequential", "joint"], default="sequential")
    parser.add_argument("--reversal_sampler", choices=["uniform", "template_balanced"], default="uniform")
    parser.add_argument("--reversal_template_min_count", type=int, default=20)
    parser.add_argument("--pref_margin_loss_weight", type=float, default=0.0)
    parser.add_argument("--rev_margin_loss_weight", type=float, default=0.0)
    parser.add_argument("--pref_hinge_loss_weight", type=float, default=0.0)
    parser.add_argument("--rev_hinge_loss_weight", type=float, default=0.0)
    parser.add_argument("--classification_margin", type=float, default=0.5)
    parser.add_argument("--margin_clip", type=float, default=2.0)
    return parser.parse_args()


def _prepare_train_tensors(
    records_root: str,
    pairs_root: str,
    device: torch.device,
    reversal_sampler: str = "uniform",
    reversal_template_min_count: int = 20,
    seed: int = 0,
) -> tuple[dict, dict]:
    records = load_split_records(records_root, "train")
    pairs = load_split_pairs(pairs_root, "train")
    reversal_pairs, reversal_sampler_report = build_reversal_training_pairs(
        pairs["reversal"],
        sampler=reversal_sampler,
        min_count=reversal_template_min_count,
        seed=seed,
    )
    tensors = {
        "preference": move_batch(preference_pair_tensors(pairs["preference"], records), device) if pairs["preference"] else None,
        "reversal": move_batch(reversal_pair_tensors(reversal_pairs, records), device) if reversal_pairs else None,
        "dominance": move_batch(dominance_pair_tensors(pairs["dominance"], records), device) if pairs["dominance"] else None,
        "reversal_pair_count_raw": len(pairs["reversal"]),
        "reversal_pair_count_used": len(reversal_pairs),
        "reversal_sampler_report": reversal_sampler_report,
    }
    return records, tensors


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _scalar_dominance_loss(
    model: torch.nn.Module,
    x_dom: torch.Tensor,
    x_sub: torch.Tensor,
    w_eval: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    losses = []
    for w in w_eval:
        w_batch = w.reshape(1, -1).repeat(x_dom.shape[0], 1)
        diff = model(x_dom, w_batch) - model(x_sub, w_batch)
        losses.append(torch.relu(float(margin) - diff).pow(2).mean())
    return torch.stack(losses).mean()


def _scalar_reversal_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    logits_1 = model(batch["x_a"], batch["w_1"]) - model(batch["x_b"], batch["w_1"])
    logits_2 = model(batch["x_a"], batch["w_2"]) - model(batch["x_b"], batch["w_2"])
    loss_1 = F.binary_cross_entropy_with_logits(logits_1, batch["labels_1"])
    loss_2 = F.binary_cross_entropy_with_logits(logits_2, batch["labels_2"])
    return loss_1 + loss_2


def _scalar_preference_margin_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    clip: float,
) -> torch.Tensor:
    pred = model(batch["x_a"], batch["w"]) - model(batch["x_b"], batch["w"])
    target = batch["rule_margin"].float().to(pred.device).clamp(-float(clip), float(clip))
    return F.smooth_l1_loss(pred, target)


def _scalar_reversal_margin_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    clip: float,
) -> torch.Tensor:
    pred_1 = model(batch["x_a"], batch["w_1"]) - model(batch["x_b"], batch["w_1"])
    pred_2 = model(batch["x_a"], batch["w_2"]) - model(batch["x_b"], batch["w_2"])
    target_1 = batch["margin_1"].float().to(pred_1.device).clamp(-float(clip), float(clip))
    target_2 = batch["margin_2"].float().to(pred_2.device).clamp(-float(clip), float(clip))
    return F.smooth_l1_loss(pred_1, target_1) + F.smooth_l1_loss(pred_2, target_2)


def _scalar_preference_hinge_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    margin: float,
) -> torch.Tensor:
    logits = model(batch["x_a"], batch["w"]) - model(batch["x_b"], batch["w"])
    signed = (2.0 * batch["labels"].float().to(logits.device) - 1.0) * logits
    return torch.relu(float(margin) - signed).pow(2).mean()


def _scalar_reversal_hinge_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    margin: float,
) -> torch.Tensor:
    logits_1 = model(batch["x_a"], batch["w_1"]) - model(batch["x_b"], batch["w_1"])
    logits_2 = model(batch["x_a"], batch["w_2"]) - model(batch["x_b"], batch["w_2"])
    signed_1 = (2.0 * batch["labels_1"].float().to(logits_1.device) - 1.0) * logits_1
    signed_2 = (2.0 * batch["labels_2"].float().to(logits_2.device) - 1.0) * logits_2
    return torch.relu(float(margin) - signed_1).pow(2).mean() + torch.relu(float(margin) - signed_2).pow(2).mean()


def _batch_lists(tensors: dict, batch_size: int, device: torch.device) -> dict[str, list[torch.Tensor]]:
    batches = {}
    for name in ("preference", "reversal", "dominance"):
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


def train(args: argparse.Namespace) -> dict:
    for key, value in {
        "reversal_loss_weight": 0.0,
        "dominance_loss_weight": 0.2,
        "dominance_margin": 0.1,
        "training_schedule": "sequential",
        "architecture": "concat",
        "film_layers": 2,
        "head_layers": 2,
        "reversal_sampler": "uniform",
        "reversal_template_min_count": 20,
        "pref_margin_loss_weight": 0.0,
        "rev_margin_loss_weight": 0.0,
        "pref_hinge_loss_weight": 0.0,
        "rev_hinge_loss_weight": 0.0,
        "classification_margin": 0.5,
        "margin_clip": 2.0,
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
        args.reversal_sampler,
        args.reversal_template_min_count,
        args.seed,
    )
    input_dim = infer_input_dim(train_records)
    config = {
        "model_type": "ConditionedScalarQualityNet",
        "architecture": args.architecture,
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "film_layers": args.film_layers,
        "head_layers": args.head_layers,
        "dropout": args.dropout,
        "preference_dim": len(OBJECTIVE_NAMES),
        "objective_order": list(OBJECTIVE_NAMES),
    }
    model = build_conditioned_scalar_model(
        args.architecture,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        preference_dim=len(OBJECTIVE_NAMES),
        film_layers=args.film_layers,
        head_layers=args.head_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    eval_preferences = EVAL_PREFERENCES.to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = {
            "preference": [],
            "preference_margin": [],
            "preference_hinge": [],
            "reversal": [],
            "reversal_margin": [],
            "reversal_hinge": [],
            "dominance": [],
        }

        if args.training_schedule == "joint":
            batches = _batch_lists(tensors, args.batch_size, device)
            steps_per_epoch = max([len(items) for items in batches.values()] + [1])
            for step in range(steps_per_epoch):
                optimizer.zero_grad()
                loss = torch.tensor(0.0, device=device)

                batch = _take_batch(tensors, batches, "preference", step)
                if batch is not None:
                    logits = model(batch["x_a"], batch["w"]) - model(batch["x_b"], batch["w"])
                    pref_loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
                    loss = loss + args.preference_loss_weight * pref_loss
                    losses["preference"].append(float(pref_loss.detach().cpu()))
                    if args.pref_margin_loss_weight > 0:
                        margin_loss = _scalar_preference_margin_loss(model, batch, args.margin_clip)
                        loss = loss + args.pref_margin_loss_weight * margin_loss
                        losses["preference_margin"].append(float(margin_loss.detach().cpu()))
                    if args.pref_hinge_loss_weight > 0:
                        hinge_loss = _scalar_preference_hinge_loss(model, batch, args.classification_margin)
                        loss = loss + args.pref_hinge_loss_weight * hinge_loss
                        losses["preference_hinge"].append(float(hinge_loss.detach().cpu()))

                batch = _take_batch(tensors, batches, "reversal", step)
                if batch is not None and args.reversal_loss_weight > 0:
                    rev_loss = _scalar_reversal_loss(model, batch)
                    loss = loss + args.reversal_loss_weight * rev_loss
                    losses["reversal"].append(float(rev_loss.detach().cpu()))
                    if args.rev_margin_loss_weight > 0:
                        margin_loss = _scalar_reversal_margin_loss(model, batch, args.margin_clip)
                        loss = loss + args.rev_margin_loss_weight * margin_loss
                        losses["reversal_margin"].append(float(margin_loss.detach().cpu()))
                    if args.rev_hinge_loss_weight > 0:
                        hinge_loss = _scalar_reversal_hinge_loss(model, batch, args.classification_margin)
                        loss = loss + args.rev_hinge_loss_weight * hinge_loss
                        losses["reversal_hinge"].append(float(hinge_loss.detach().cpu()))

                batch = _take_batch(tensors, batches, "dominance", step)
                if batch is not None and args.dominance_loss_weight > 0:
                    dom_loss = _scalar_dominance_loss(
                        model,
                        batch["x_dom"],
                        batch["x_sub"],
                        eval_preferences,
                        margin=args.dominance_margin,
                    )
                    loss = loss + args.dominance_loss_weight * dom_loss
                    losses["dominance"].append(float(dom_loss.detach().cpu()))

                loss.backward()
                optimizer.step()
        else:
            preference = tensors["preference"]
            if preference is not None:
                for indices in shuffled_batches(preference["labels"].shape[0], args.batch_size, device):
                    batch = tensor_batch(preference, indices)
                    logits = model(batch["x_a"], batch["w"]) - model(batch["x_b"], batch["w"])
                    pref_loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
                    loss = args.preference_loss_weight * pref_loss
                    if args.pref_margin_loss_weight > 0:
                        margin_loss = _scalar_preference_margin_loss(model, batch, args.margin_clip)
                        loss = loss + args.pref_margin_loss_weight * margin_loss
                        losses["preference_margin"].append(float(margin_loss.detach().cpu()))
                    if args.pref_hinge_loss_weight > 0:
                        hinge_loss = _scalar_preference_hinge_loss(model, batch, args.classification_margin)
                        loss = loss + args.pref_hinge_loss_weight * hinge_loss
                        losses["preference_hinge"].append(float(hinge_loss.detach().cpu()))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses["preference"].append(float(pref_loss.detach().cpu()))

            reversal = tensors["reversal"]
            if reversal is not None and args.reversal_loss_weight > 0:
                for indices in shuffled_batches(reversal["labels_1"].shape[0], args.batch_size, device):
                    batch = tensor_batch(reversal, indices)
                    rev_loss = _scalar_reversal_loss(model, batch)
                    loss = args.reversal_loss_weight * rev_loss
                    if args.rev_margin_loss_weight > 0:
                        margin_loss = _scalar_reversal_margin_loss(model, batch, args.margin_clip)
                        loss = loss + args.rev_margin_loss_weight * margin_loss
                        losses["reversal_margin"].append(float(margin_loss.detach().cpu()))
                    if args.rev_hinge_loss_weight > 0:
                        hinge_loss = _scalar_reversal_hinge_loss(model, batch, args.classification_margin)
                        loss = loss + args.rev_hinge_loss_weight * hinge_loss
                        losses["reversal_hinge"].append(float(hinge_loss.detach().cpu()))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses["reversal"].append(float(rev_loss.detach().cpu()))

            dominance = tensors["dominance"]
            if dominance is not None:
                for indices in shuffled_batches(dominance["x_dom"].shape[0], args.batch_size, device):
                    batch = tensor_batch(dominance, indices)
                    dom_loss = _scalar_dominance_loss(
                        model,
                        batch["x_dom"],
                        batch["x_sub"],
                        eval_preferences,
                        margin=args.dominance_margin,
                    )
                    loss = args.dominance_loss_weight * dom_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses["dominance"].append(float(dom_loss.detach().cpu()))

        append_jsonl(train_log, {
            "epoch": epoch,
            "preference_loss": _mean(losses["preference"]),
            "preference_margin_loss": _mean(losses["preference_margin"]),
            "preference_hinge_loss": _mean(losses["preference_hinge"]),
            "reversal_loss": _mean(losses["reversal"]),
            "reversal_margin_loss": _mean(losses["reversal_margin"]),
            "reversal_hinge_loss": _mean(losses["reversal_hinge"]),
            "dominance_loss": _mean(losses["dominance"]),
        })

    metadata = {
        "model_type": "ConditionedScalarQualityNet",
        "architecture": args.architecture,
        "param_count": count_parameters(model),
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "film_layers": args.film_layers,
        "head_layers": args.head_layers,
        "dropout": args.dropout,
        "objective_order": list(OBJECTIVE_NAMES),
        "records_root": args.records_root,
        "pairs_root": args.pairs_root,
        "seed": args.seed,
        "requested_device": args.device,
        "actual_device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "reversal_loss_weight": args.reversal_loss_weight,
        "reversal_sampler": args.reversal_sampler,
        "reversal_template_min_count": args.reversal_template_min_count,
        "reversal_pair_count_raw": tensors["reversal_pair_count_raw"],
        "reversal_pair_count_used": tensors["reversal_pair_count_used"],
        "reversal_sampler_report": tensors["reversal_sampler_report"],
        "pref_margin_loss_weight": args.pref_margin_loss_weight,
        "rev_margin_loss_weight": args.rev_margin_loss_weight,
        "pref_hinge_loss_weight": args.pref_hinge_loss_weight,
        "rev_hinge_loss_weight": args.rev_hinge_loss_weight,
        "classification_margin": args.classification_margin,
        "margin_clip": args.margin_clip,
        "dominance_loss_weight": args.dominance_loss_weight,
        "dominance_margin": args.dominance_margin,
        "training_schedule": args.training_schedule,
    }
    save_model_checkpoint(out_dir / "model.pt", model, config)
    write_json(out_dir / "config.json", config)
    write_json(out_dir / "metadata.json", metadata)

    for split in ("val", "test"):
        records, pairs = load_split_bundle(args.records_root, args.pairs_root, split)
        diagnostics = evaluate_scalar_model(model, records, pairs, device=device)
        write_json(out_dir / f"diagnostics_{split}.json", diagnostics)

    return metadata


def main() -> None:
    metadata = train(parse_args())
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
