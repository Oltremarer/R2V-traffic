from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import torch

from pareto.constants import OBJECTIVE_INDEX, OBJECTIVE_NAMES


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_records_by_id(record_paths: Iterable[str | Path]) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in record_paths:
        for row in read_jsonl(path):
            sample_id = row.get("sample_id")
            if not sample_id:
                raise ValueError(f"record without sample_id in {path}")
            if sample_id in records:
                raise ValueError(f"duplicate sample_id: {sample_id}")
            records[sample_id] = row
    return records


def load_split_records(records_root: str | Path, split: str) -> dict[str, dict]:
    return load_records_by_id([Path(records_root) / f"{split}_raw.jsonl"])


def load_pair_file(path: str | Path) -> list[dict]:
    return read_jsonl(path)


def load_split_pairs(pairs_root: str | Path, split: str) -> dict[str, list[dict]]:
    root = Path(pairs_root) / split
    return {
        "objective": load_pair_file(root / "objective_pairs.jsonl"),
        "preference": load_pair_file(root / "preference_pairs.jsonl"),
        "dominance": load_pair_file(root / "dominance_pairs.jsonl"),
        "reversal": load_pair_file(root / "reversal_pairs.jsonl"),
    }


def infer_input_dim(records_by_id: dict[str, dict]) -> int:
    if not records_by_id:
        raise ValueError("cannot infer input_dim from empty records")
    first = next(iter(records_by_id.values()))
    return len(first["obs_features"])


def feature_tensor(record: dict) -> torch.Tensor:
    return torch.tensor(record["obs_features"], dtype=torch.float32)


def make_feature_tensor(records_by_id: dict[str, dict], ids: list[str]) -> torch.Tensor:
    try:
        tensors = [feature_tensor(records_by_id[sample_id]) for sample_id in ids]
    except KeyError as exc:
        raise KeyError(f"pair references unknown sample_id: {exc.args[0]}") from exc
    return torch.stack(tensors, dim=0)


def objective_target_tensor(records_by_id: dict[str, dict], ids: list[str]) -> torch.Tensor:
    values = []
    for sample_id in ids:
        record = records_by_id[sample_id]
        values.append([float(record["objective_values_norm"][name]) for name in OBJECTIVE_NAMES])
    return torch.tensor(values, dtype=torch.float32)


def objective_pair_tensors(pairs: list[dict], records_by_id: dict[str, dict]) -> dict[str, torch.Tensor]:
    return {
        "x_a": make_feature_tensor(records_by_id, [pair["a_id"] for pair in pairs]),
        "x_b": make_feature_tensor(records_by_id, [pair["b_id"] for pair in pairs]),
        "objective_idx": torch.tensor([OBJECTIVE_INDEX[pair["objective"]] for pair in pairs], dtype=torch.long),
        "labels": torch.tensor([float(pair["label"]) for pair in pairs], dtype=torch.float32),
    }


def preference_pair_tensors(pairs: list[dict], records_by_id: dict[str, dict]) -> dict[str, torch.Tensor]:
    return {
        "x_a": make_feature_tensor(records_by_id, [pair["a_id"] for pair in pairs]),
        "x_b": make_feature_tensor(records_by_id, [pair["b_id"] for pair in pairs]),
        "w": torch.tensor([pair["w"] for pair in pairs], dtype=torch.float32),
        "labels": torch.tensor([float(pair["label"]) for pair in pairs], dtype=torch.float32),
        "rule_margin": torch.tensor([float(pair.get("rule_margin", 0.0)) for pair in pairs], dtype=torch.float32),
    }


def dominance_pair_tensors(pairs: list[dict], records_by_id: dict[str, dict]) -> dict[str, torch.Tensor]:
    dom_ids: list[str] = []
    sub_ids: list[str] = []
    objective_margins = []
    for pair in pairs:
        margins = [float(pair.get("objective_margins_norm", {}).get(name, 0.0)) for name in OBJECTIVE_NAMES]
        if pair["dominates"] == "a":
            dom_ids.append(pair["a_id"])
            sub_ids.append(pair["b_id"])
            objective_margins.append(margins)
        elif pair["dominates"] == "b":
            dom_ids.append(pair["b_id"])
            sub_ids.append(pair["a_id"])
            objective_margins.append([-value for value in margins])
        else:
            raise ValueError(f"invalid dominates field: {pair['dominates']}")
    return {
        "x_dom": make_feature_tensor(records_by_id, dom_ids),
        "x_sub": make_feature_tensor(records_by_id, sub_ids),
        "objective_margins": torch.tensor(objective_margins, dtype=torch.float32),
    }


def reversal_pair_tensors(pairs: list[dict], records_by_id: dict[str, dict]) -> dict[str, torch.Tensor]:
    return {
        "x_a": make_feature_tensor(records_by_id, [pair["a_id"] for pair in pairs]),
        "x_b": make_feature_tensor(records_by_id, [pair["b_id"] for pair in pairs]),
        "w_1": torch.tensor([pair["w_1"] for pair in pairs], dtype=torch.float32),
        "w_2": torch.tensor([pair["w_2"] for pair in pairs], dtype=torch.float32),
        "labels_1": torch.tensor([float(pair["label_1"]) for pair in pairs], dtype=torch.float32),
        "labels_2": torch.tensor([float(pair["label_2"]) for pair in pairs], dtype=torch.float32),
        "margin_1": torch.tensor([float(pair.get("margin_1", 0.0)) for pair in pairs], dtype=torch.float32),
        "margin_2": torch.tensor([float(pair.get("margin_2", 0.0)) for pair in pairs], dtype=torch.float32),
    }


def reversal_template_key(pair: dict) -> str:
    return f"{pair.get('w_1_name', 'unknown')}__{pair.get('w_2_name', 'unknown')}"


def build_reversal_training_pairs(
    pairs: list[dict],
    sampler: str = "uniform",
    min_count: int = 20,
    seed: int = 0,
) -> tuple[list[dict], dict]:
    template_counts: dict[str, int] = {}
    groups: dict[str, list[dict]] = {}
    for pair in pairs:
        key = reversal_template_key(pair)
        groups.setdefault(key, []).append(pair)
        template_counts[key] = template_counts.get(key, 0) + 1
    report = {
        "sampler": sampler,
        "template_counts": dict(sorted(template_counts.items())),
        "enabled_templates": [],
        "underpowered_templates": [],
        "sampled_template_counts": {},
        "oversampling_ratio_by_template": {},
    }
    if sampler == "uniform":
        report["enabled_templates"] = sorted(groups)
        report["sampled_template_counts"] = dict(sorted(template_counts.items()))
        report["oversampling_ratio_by_template"] = {key: 1.0 for key in sorted(groups)}
        return pairs, report
    if sampler != "template_balanced":
        raise ValueError(f"unknown reversal sampler: {sampler}")

    min_count = max(1, int(min_count))
    enabled = {key: rows for key, rows in groups.items() if len(rows) >= min_count}
    underpowered = sorted(key for key, rows in groups.items() if len(rows) < min_count)
    report["enabled_templates"] = sorted(enabled)
    report["underpowered_templates"] = underpowered
    if not enabled:
        return [], report

    target = max(len(rows) for rows in enabled.values())
    rng = random.Random(seed)
    sampled: list[dict] = []
    sampled_counts: dict[str, int] = {}
    oversampling: dict[str, float] = {}
    for key in sorted(enabled):
        rows = list(enabled[key])
        chosen = list(rows)
        while len(chosen) < target:
            chosen.append(rng.choice(rows))
        rng.shuffle(chosen)
        sampled.extend(chosen)
        sampled_counts[key] = len(chosen)
        oversampling[key] = float(len(chosen) / len(rows))
    rng.shuffle(sampled)
    report["sampled_template_counts"] = sampled_counts
    report["oversampling_ratio_by_template"] = oversampling
    return sampled, report


def split_tensor_batch(tensors: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value[indices] for key, value in tensors.items()}
