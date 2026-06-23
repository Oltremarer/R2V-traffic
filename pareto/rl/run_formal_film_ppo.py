#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pareto.rl.formal_experiment_spec import load_formal_experiment_spec, write_formal_experiment_spec
from pareto.train_common import write_json


def _load_gate(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _adapter_plan(spec) -> dict[str, Any]:
    return {
        "reward_adapter": spec.reward_adapter,
        "reward_scale": spec.reward_scale,
        "reward_clip": spec.reward_clip,
        "reward_normalization": spec.reward_normalization,
        "potential_gamma": spec.potential_gamma,
        "mix_env_reward": spec.mix_env_reward,
        "film_model_dir": spec.film_model_dir,
        "film_model_hash": spec.film_model_hash,
        "vectorq_diagnostic_only": spec.reward_adapter == "vectorq_diagnostic_potential",
    }


def run_dry_run(spec_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    spec = load_formal_experiment_spec(spec_path)
    gate = _load_gate(spec.formal_gate_decision_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "dry_run": True,
        "env_rollout": False,
        "ppo_training": False,
        "policy_update": False,
        "performance_claim": False,
        "approved_formal_spec": spec.approved_formal_spec,
        "formal_gate_decision": gate,
        "spec_hash": spec.spec_hash(),
        "method": spec.reward_adapter,
        "scenario": spec.scenario,
        "traffic_file": spec.traffic_file,
        "cityflow_seed": spec.cityflow_seed,
        "policy_seed": spec.policy_seed,
        "model_seed": spec.model_seed,
        "state_encoder_hash": spec.state_encoder_hash,
        "objective_normalizer_hash": spec.objective_normalizer_hash,
        "policy_conditioned_on_w": spec.policy_conditioned_on_w,
        "critic_conditioned_on_w": spec.critic_conditioned_on_w,
        "min_action_time": spec.min_action_time,
        "forbid_inner_step_direct_call": spec.forbid_inner_step_direct_call,
        "adapter_plan": _adapter_plan(spec),
    }
    write_json(out_dir / "metadata.json", payload)
    write_formal_experiment_spec(out_dir / "formal_experiment_spec.json", spec)
    shutil.copyfile(spec.formal_gate_decision_path, out_dir / "formal_gate_decision.json")
    write_json(out_dir / "status.json", {"status": "DRY_RUN_DONE", "env_rollout": False, "ppo_training": False})
    (out_dir / "DRY_RUN_DONE").write_text("dry run completed; no env rollout or PPO training\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        raise SystemExit("formal training is not implemented or allowed; pass --dry_run")
    payload = run_dry_run(args.spec, args.out_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
