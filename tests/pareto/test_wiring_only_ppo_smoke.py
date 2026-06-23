from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import torch

from pareto.rl.ppo_wiring_smoke import run_wiring_smoke
from pareto.rl.reward_adapter import WeightedProxyRewardAdapter


def test_weighted_proxy_reward_adapter_logs_components():
    adapter = WeightedProxyRewardAdapter([0.5, 0.5, 0.0, 0.0])
    reward, debug = adapter.compute(
        obs_t=torch.zeros(4),
        obs_tp1=torch.ones(4),
        objectives_t={"efficiency": 1.0, "safety": 0.0, "fairness": 0.0, "stability": 0.0},
        objectives_tp1={"efficiency": 2.0, "safety": 1.0, "fairness": 0.0, "stability": 0.0},
        done=False,
    )

    assert reward == 1.5
    assert debug["adapter"] == "weighted_proxy"
    assert debug["w"] == [0.5, 0.5, 0.0, 0.0]
    assert debug["weighted_proxy_reward"] == 1.5


def test_wiring_only_smoke_writes_no_performance_claim(tmp_path: Path):
    records_root = tmp_path / "records"
    records_root.mkdir()
    rows = [
        {
            "sample_id": "s0",
            "obs_features": [0.0, 0.0, 0.0, 0.0],
            "objective_values_norm": {
                "efficiency": 1.0,
                "safety": 0.0,
                "fairness": 0.0,
                "stability": 0.0,
            },
        },
        {
            "sample_id": "s1",
            "obs_features": [1.0, 1.0, 1.0, 1.0],
            "objective_values_norm": {
                "efficiency": 2.0,
                "safety": 1.0,
                "fairness": 0.0,
                "stability": 0.0,
            },
        },
    ]
    (records_root / "test_raw.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    gate_path = tmp_path / "formal_gate.json"
    gate_path.write_text(
        json.dumps({
            "wiring_smoke_allowed": True,
            "ppo_formal_allowed": False,
            "claim_mode": "diagnostics_only",
        }),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    payload = run_wiring_smoke(Namespace(
        records_root=str(records_root),
        split="test",
        out_dir=str(out_dir),
        formal_gate_decision=str(gate_path),
        vector_model_dir=None,
        scalar_model_dir=None,
        device="cpu",
        max_transitions=1,
        gamma=0.99,
    ))

    assert payload["wiring_only"] is True
    assert payload["performance_claim"] is False
    assert payload["ppo_training"] is False
    assert payload["policy_update"] is False
    assert payload["transition_source"] == "adjacent_split_records_not_env_rollout"
    assert payload["formal_gate_decision"]["wiring_smoke_allowed"] is True
    assert (out_dir / "metrics.jsonl").exists()
    assert (out_dir / "metadata.json").exists()
