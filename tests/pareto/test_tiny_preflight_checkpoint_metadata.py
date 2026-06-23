from __future__ import annotations

import torch

from pareto.rl.formal_pilot_runner import _save_tiny_training_checkpoint
from pareto.rl.ppo_actor_critic import PreferenceConditionedActorCritic, load_training_checkpoint


def test_tiny_preflight_checkpoint_persists_guard_metadata(tmp_path):
    model = PreferenceConditionedActorCritic(obs_dim=5, preference_dim=4, action_dim=3, hidden_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    obs = torch.randn(4, 5)
    w = torch.ones(4, 4) / 4.0
    logits, values = model(obs, w)
    loss = logits.mean() + values.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    metadata = {
        "method": "film_scalar_potential",
        "reward_adapter": "film_scalar_potential",
        "tiny_real_ppo_update_preflight": True,
        "tiny_preflight_not_pilot": True,
        "exclude_from_analysis": True,
        "checkpoint_use": "preflight_resume_test_only",
        "pilot_execution": False,
        "formal_experiment": False,
        "performance_claim": False,
        "ppo_config_hash": "ppo_hash",
        "pilot_spec_hash": "pilot_hash",
        "objective_normalizer_hash": "norm_hash",
        "film_model_hash": "film_hash",
    }
    path = tmp_path / "training_checkpoint_last.pt"

    _save_tiny_training_checkpoint(
        path,
        model,
        optimizer,
        metadata,
        total_steps=2,
        episodes=1,
        max_policy_updates=1,
    )

    _, loaded_optimizer, payload = load_training_checkpoint(path)
    checkpoint_metadata = payload["metadata"]
    assert loaded_optimizer.state_dict()["state"]
    assert checkpoint_metadata["step"] == 2
    assert checkpoint_metadata["episode"] == 1
    assert checkpoint_metadata["global_update"] == 1
    assert checkpoint_metadata["method"] == "film_scalar_potential"
    assert checkpoint_metadata["reward_adapter"] == "film_scalar_potential"
    assert checkpoint_metadata["tiny_preflight_not_pilot"] is True
    assert checkpoint_metadata["exclude_from_analysis"] is True
    assert checkpoint_metadata["checkpoint_use"] == "preflight_resume_test_only"
    assert checkpoint_metadata["pilot_execution"] is False
    assert checkpoint_metadata["formal_experiment"] is False
    assert checkpoint_metadata["performance_claim"] is False
    assert checkpoint_metadata["ppo_config_hash"] == "ppo_hash"
    assert checkpoint_metadata["pilot_spec_hash"] == "pilot_hash"
    assert checkpoint_metadata["objective_normalizer_hash"] == "norm_hash"
    assert checkpoint_metadata["film_model_hash"] == "film_hash"
