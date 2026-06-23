from __future__ import annotations

from pathlib import Path


def test_formal_ppo_dry_run_entrypoint_does_not_reference_cityflow_or_env_step():
    root = Path(__file__).resolve().parents[2]
    sources = [
        root / "pareto/rl/run_formal_ppo_dry_run.py",
        root / "pareto/rl/formal_ppo_trainer.py",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in sources)

    assert "CityFlowEnv" not in combined
    assert "env.step" not in combined
    assert "env.step(" not in combined
    assert "cityflow_env" not in combined
