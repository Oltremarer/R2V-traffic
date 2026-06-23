#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from pareto.constants import OBJECTIVE_NAMES
from pareto.data.offline_dataset import load_split_records
from pareto.eval.run_offline_diagnostics import _load_scalar_model, _load_vector_model
from pareto.rl.reward_adapter import ScalarQualityRewardAdapter, VectorQRewardAdapter, WeightedProxyRewardAdapter
from pareto.train_common import append_jsonl, resolve_device, write_json


DEFAULT_W = [0.25, 0.25, 0.25, 0.25]


def _load_gate(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _record_to_obs(record: dict) -> torch.Tensor:
    return torch.tensor(record["obs_features"], dtype=torch.float32)


def _record_objectives(record: dict) -> dict[str, float]:
    values = record.get("objective_values_norm", {})
    return {name: float(values.get(name, 0.0)) for name in OBJECTIVE_NAMES}


def _build_adapters(args: argparse.Namespace, device: torch.device) -> list[Any]:
    adapters: list[Any] = [WeightedProxyRewardAdapter(DEFAULT_W)]
    if args.vector_model_dir:
        vector_model, vector_scorer, _ = _load_vector_model(args.vector_model_dir, device)
        adapters.append(VectorQRewardAdapter(vector_model, DEFAULT_W, scorer=vector_scorer, gamma=args.gamma, device=device))
    if args.scalar_model_dir:
        scalar_model, _ = _load_scalar_model(args.scalar_model_dir, device)
        adapters.append(ScalarQualityRewardAdapter(scalar_model, DEFAULT_W, gamma=args.gamma, device=device))
    return adapters


def run_wiring_smoke(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gate = _load_gate(args.formal_gate_decision)
    if not gate.get("wiring_smoke_allowed", False):
        raise ValueError("formal gate does not allow wiring-only PPO smoke")
    if gate.get("ppo_formal_allowed", False):
        raise ValueError("this smoke script is only for pre-formal wiring; use formal PPO tooling after gate pass")

    device = resolve_device(args.device)
    records = list(load_split_records(args.records_root, args.split).values())
    if len(records) < 2:
        raise ValueError("wiring smoke requires at least two records")
    adapters = _build_adapters(args, device)
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    steps = min(int(args.max_transitions), len(records) - 1)
    adapter_names = [adapter.name for adapter in adapters]
    for idx in range(steps):
        current = records[idx]
        nxt = records[idx + 1]
        obs_t = _record_to_obs(current)
        obs_tp1 = _record_to_obs(nxt)
        objectives_t = _record_objectives(current)
        objectives_tp1 = _record_objectives(nxt)
        for adapter in adapters:
            reward, debug = adapter.compute(
                obs_t=obs_t,
                obs_tp1=obs_tp1,
                objectives_t=objectives_t,
                objectives_tp1=objectives_tp1,
                done=False,
            )
            append_jsonl(metrics_path, {
                "wiring_only": True,
                "performance_claim": False,
                "step": idx,
                "sample_id": current.get("sample_id"),
                "next_sample_id": nxt.get("sample_id"),
                "adapter": adapter.name,
                "reward": reward,
                "debug": debug,
            })

    metadata = {
        "wiring_only": True,
        "performance_claim": False,
        "ppo_formal_allowed": False,
        "ppo_training": False,
        "policy_update": False,
        "transition_source": "adjacent_split_records_not_env_rollout",
        "records_root": str(args.records_root),
        "split": args.split,
        "steps": steps,
        "adapter_names": adapter_names,
        "formal_gate_decision": gate,
        "vector_model_dir": args.vector_model_dir,
        "scalar_model_dir": args.scalar_model_dir,
        "outputs": {
            "metrics_jsonl": str(metrics_path),
            "metadata_json": str(out_dir / "metadata.json"),
            "formal_gate_decision_json": str(out_dir / "formal_gate_decision.json"),
        },
    }
    write_json(out_dir / "metadata.json", metadata)
    write_json(out_dir / "formal_gate_decision.json", gate)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records_root", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--formal_gate_decision", required=True)
    parser.add_argument("--vector_model_dir")
    parser.add_argument("--scalar_model_dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_transitions", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    return parser.parse_args()


def main() -> None:
    payload = run_wiring_smoke(parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
