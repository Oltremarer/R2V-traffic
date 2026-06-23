from __future__ import annotations

from pathlib import Path
from typing import Any

from pareto.rl.paper_final_experiment_manifest import REQUIRED_PAPER_BASELINES
from pareto.rl.paper_final_root_policy import build_paper_final_roots


PAPER_BASELINE_COMMANDS: dict[str, dict[str, Any]] = {
    "Random": {"status": "implemented", "script": "run_random.py", "method_id": "random"},
    "FixedTime": {"status": "implemented", "script": "run_fixedtime.py", "method_id": "fixed_time"},
    "MaxPressure": {"status": "implemented", "script": "run_maxpressure.py", "method_id": "maxpressure"},
    "PressLight": {"status": "implemented", "script": "run_presslight.py", "method_id": "presslight"},
    "MPLight": {"status": "implemented", "script": "run_mplight.py", "method_id": "mplight"},
    "CoLight": {"status": "implemented", "script": "run_colight.py", "method_id": "colight"},
    "Advanced-Co": {"status": "implemented", "script": "run_advanced_colight.py", "method_id": "advanced_colight"},
    "C2T-scalar": {
        "status": "missing_blocker",
        "script": None,
        "method_id": "c2t_scalar",
        "blocker": "unresolved_c2t_scalar_command",
    },
    "Cond-Scalar-RL": {"status": "implemented", "script": "pareto/rl/formal_pilot_runner.py", "method_id": "film_scalar_potential"},
    "Weighted-RL": {
        "status": "missing_blocker",
        "script": "pareto/rl/formal_pilot_runner.py",
        "method_id": "weighted_proxy",
        "requires_reviewer_mapping_approval": True,
        "blocker": "Weighted-RL to weighted_proxy mapping approval required",
    },
    "VectorQ-PPO": {"status": "implemented", "script": "pareto/rl/formal_pilot_runner.py", "method_id": "vector_quality_potential"},
}


def validate_paper_baseline_commands(commands: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    missing = sorted(set(REQUIRED_PAPER_BASELINES) - set(commands))
    extra = sorted(set(commands) - set(REQUIRED_PAPER_BASELINES))
    if missing:
        raise ValueError(f"missing paper baseline commands: {missing}")
    if extra:
        if any("EnvReward" in item or item == "env_reward" for item in extra):
            raise ValueError("Stage-A env_reward diagnostic is not a paper baseline")
        raise ValueError(f"not a paper baseline: {extra}")
    for name, row in commands.items():
        status = row.get("status")
        if status not in {"implemented", "missing_blocker"}:
            raise ValueError(f"unknown command status for {name}: {status}")
        script = row.get("script")
        if status == "implemented" and not script:
            raise ValueError(f"implemented baseline {name} missing script")
        if script and not Path(script).exists():
            raise ValueError(f"baseline {name} command path missing: {script}")
    return {name: dict(row) for name, row in commands.items()}


def baseline_command_blockers(commands: dict[str, dict[str, Any]]) -> list[str]:
    validated = validate_paper_baseline_commands(commands)
    return [
        f"{name}: {row.get('blocker') or 'command unresolved'}"
        for name, row in validated.items()
        if row.get("status") == "missing_blocker" or row.get("requires_reviewer_mapping_approval") is True
    ]


def build_baseline_command_preview(
    baseline: str,
    *,
    city: str,
    traffic_file: str,
    seed: int,
) -> dict[str, Any]:
    commands = validate_paper_baseline_commands(PAPER_BASELINE_COMMANDS)
    if baseline not in commands:
        raise ValueError(f"unknown paper baseline: {baseline}")
    row = commands[baseline]
    if row.get("status") != "implemented":
        raise ValueError(f"baseline {baseline} is not executable: {row.get('blocker')}")
    roots = build_paper_final_roots(
        city=city,
        traffic_file=traffic_file,
        method=baseline,
        seed=int(seed),
        preference_id="smoke",
    )
    out_dir = roots.preflight / city / baseline / f"seed{int(seed)}"
    command = (
        f"{row['script']} --memo paper_final_smoke --traffic_file {traffic_file} "
        f"--seed {int(seed)} --out_dir {out_dir}"
    )
    return {
        "baseline": baseline,
        "method_id": row["method_id"],
        "city": city,
        "traffic_file": traffic_file,
        "seed": int(seed),
        "out_dir": str(out_dir),
        "command": command,
        "executes_now": False,
        "reads_result_values": False,
        "generates_ranking": False,
    }
