from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import torch

from pareto.common.artifact_guard import assert_no_forbidden_performance_artifacts
from pareto.common.io import append_jsonl, write_json
from pareto.rl.formal_ppo_config import FormalPPODryRunConfig
from pareto.rl.ppo_actor_critic import (
    PreferenceConditionedActorCritic,
    load_actor_critic_checkpoint,
    load_training_checkpoint,
    save_actor_critic_checkpoint,
    save_training_checkpoint,
)
from pareto.rl.ppo_buffer import PPORolloutBuffer
from pareto.rl.ppo_loss import clipped_ppo_loss


PREFERENCE_TEMPLATES = [
    torch.tensor([1.0, 0.0, 0.0, 0.0]),
    torch.tensor([0.0, 1.0, 0.0, 0.0]),
    torch.tensor([0.0, 0.0, 1.0, 0.0]),
    torch.tensor([0.0, 0.0, 0.0, 1.0]),
    torch.tensor([0.25, 0.25, 0.25, 0.25]),
]


def _metadata(config: FormalPPODryRunConfig, method: str) -> dict[str, Any]:
    return {
        "dry_run": True,
        "env_rollout": False,
        "env_training": False,
        "ppo_training": False,
        "ppo_training_on_env": False,
        "synthetic_ppo_update": True,
        "synthetic_update_only": True,
        "policy_update": True,
        "performance_claim": False,
        "pilot_execution": False,
        "formal_experiment": False,
        "formal_ppo_dryrun_spec_snapshot": True,
        "formal_experiment_spec_is_snapshot_only": True,
        "spec_snapshot_name": "formal_ppo_dryrun_spec.json",
        "not_for_main_results": True,
        "method": method,
        "ppo_config_hash": config.ppo_config_hash(),
        "algorithm_label": config.ppo["algorithm_label"],
    }


def _synthetic_buffer(model: PreferenceConditionedActorCritic, config: FormalPPODryRunConfig) -> PPORolloutBuffer:
    buffer = PPORolloutBuffer()
    rollout_steps = int(config.ppo["rollout_steps"])
    obs_dim = int(config.model["obs_dim"])
    action_dim = int(config.model["action_dim"])
    del action_dim
    for idx in range(rollout_steps):
        obs = torch.randn(obs_dim)
        w = PREFERENCE_TEMPLATES[idx % len(PREFERENCE_TEMPLATES)]
        with torch.no_grad():
            actions, log_probs, values = model.act(obs.unsqueeze(0), w.unsqueeze(0))
        reward = float(torch.tanh(obs.mean() + 0.1 * (idx % 3)).item())
        buffer.add(
            obs=obs,
            w=w,
            action=int(actions[0].item()),
            reward=reward,
            done=idx == rollout_steps - 1,
            value=float(values[0].item()),
            log_prob=float(log_probs[0].item()),
        )
    return buffer


def run_synthetic_ppo_dry_run(config: FormalPPODryRunConfig, method: str, out_dir: str | Path) -> dict[str, Any]:
    if method not in config.pilot["methods"]:
        raise ValueError(f"method {method} is not in pilot methods")
    random.seed(int(config.pilot.get("model_seed", 0)))
    torch.manual_seed(int(config.pilot.get("model_seed", 0)))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "loss_debug.jsonl",
        "metadata.json",
        "ppo_config.json",
        "formal_ppo_dryrun_spec.json",
        "formal_experiment_spec.json",
        "formal_gate_decision.json",
        "pilot_spec.json",
        "checkpoint_last.pt",
        "training_checkpoint_last.pt",
        "status.json",
        "DRY_RUN_DONE",
    ):
        stale_path = out / stale_name
        if stale_path.exists():
            stale_path.unlink()
    metadata = _metadata(config, method)
    write_json(out / "metadata.json", metadata)
    write_json(out / "ppo_config.json", {"ppo": config.ppo, "ppo_config_hash": config.ppo_config_hash()})
    write_json(out / "formal_ppo_dryrun_spec.json", config.to_dict())
    write_json(out / "formal_experiment_spec.json", config.to_dict())
    gate_path = Path(config.pilot["formal_gate_decision_path"])
    pilot_spec = Path(config.pilot["pilot_spec_path"])
    if gate_path.exists():
        shutil.copyfile(gate_path, out / "formal_gate_decision.json")
    else:
        write_json(out / "formal_gate_decision.json", {"missing_source": str(gate_path), "ppo_formal_allowed": False})
    if pilot_spec.exists():
        shutil.copyfile(pilot_spec, out / "pilot_spec.json")
    else:
        (out / "pilot_spec.json").write_text(json.dumps({"missing_source": str(pilot_spec)}), encoding="utf-8")

    model = PreferenceConditionedActorCritic(
        obs_dim=int(config.model["obs_dim"]),
        preference_dim=int(config.model["preference_dim"]),
        action_dim=int(config.model["action_dim"]),
        hidden_dim=int(config.model["hidden_dim"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.ppo["lr"]))
    buffer = _synthetic_buffer(model, config)
    batch = buffer.compute_returns_and_advantages(
        last_value=0.0,
        gamma=float(config.ppo["gamma"]),
        gae_lambda=float(config.ppo["gae_lambda"]),
        normalize_advantages=bool(config.ppo["normalize_advantages"]),
    )
    batch_size = int(config.ppo["minibatch_size"])
    num_items = len(buffer)
    losses: list[dict[str, float]] = []
    for epoch in range(int(config.ppo["update_epochs"])):
        order = torch.randperm(num_items)
        for start in range(0, num_items, batch_size):
            idx = order[start:start + batch_size]
            new_log_probs, entropy, values = model.evaluate_actions(
                batch["obs"][idx],
                batch["w"][idx],
                batch["actions"][idx],
            )
            loss = clipped_ppo_loss(
                new_log_probs=new_log_probs,
                old_log_probs=batch["old_log_probs"][idx],
                advantages=batch["advantages"][idx],
                values=values,
                returns=batch["returns"][idx],
                entropy=entropy,
                clip_eps=float(config.ppo["clip_eps"]),
                value_loss_coef=float(config.ppo["value_loss_coef"]),
                entropy_coef=float(config.ppo["entropy_coef"]),
            )
            optimizer.zero_grad()
            loss["total_loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.ppo["max_grad_norm"]))
            optimizer.step()
            row = {
                "epoch": epoch,
                "policy_loss": float(loss["policy_loss"].detach().item()),
                "value_loss": float(loss["value_loss"].detach().item()),
                "entropy_bonus": float(loss["entropy_bonus"].detach().item()),
                "approx_kl": float(loss["approx_kl"].detach().item()),
                "clip_fraction": float(loss["clip_fraction"].detach().item()),
                "ratio_mean": float(loss["ratio_mean"].detach().item()),
                "ratio_min": float(loss["ratio_min"].detach().item()),
                "ratio_max": float(loss["ratio_max"].detach().item()),
                "total_loss": float(loss["total_loss"].detach().item()),
                "grad_norm": float(grad_norm),
            }
            if not all(torch.isfinite(torch.tensor(value)) for value in row.values()):
                raise ValueError("non-finite synthetic PPO dry-run loss")
            losses.append(row)
            append_jsonl(out / "loss_debug.jsonl", [row])
    checkpoint = out / "checkpoint_last.pt"
    save_actor_critic_checkpoint(checkpoint, model, metadata)
    save_training_checkpoint(
        out / "training_checkpoint_last.pt",
        model,
        optimizer,
        metadata,
        step=len(losses),
        episode=0,
        global_update=len(losses),
    )
    loaded, _ = load_actor_critic_checkpoint(checkpoint)
    _, loaded_optimizer, training_payload = load_training_checkpoint(out / "training_checkpoint_last.pt")
    if not loaded_optimizer.state_dict()["state"] or training_payload["metadata"]["global_update"] != len(losses):
        raise ValueError("optimizer checkpoint roundtrip failed")
    logits, values = loaded(torch.zeros(2, int(config.model["obs_dim"])), torch.ones(2, 4) / 4.0)
    if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
        raise ValueError("non-finite logits after checkpoint roundtrip")
    write_json(out / "status.json", {"status": "DRY_RUN_DONE", "loss_rows": len(losses), **metadata})
    (out / "DRY_RUN_DONE").write_text("synthetic PPO dry-run completed; no env rollout\n", encoding="utf-8")
    assert_no_forbidden_performance_artifacts(out)
    return {"status": "DRY_RUN_DONE", "loss_rows": len(losses), **metadata}
