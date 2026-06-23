from __future__ import annotations

import pytest

from pareto.rl.paper_baseline_commands import (
    PAPER_BASELINE_COMMANDS,
    build_baseline_command_preview,
    baseline_command_blockers,
    validate_paper_baseline_commands,
)


def test_baseline_commands_cover_required_scripts_and_aliases():
    commands = validate_paper_baseline_commands(PAPER_BASELINE_COMMANDS)

    assert commands["Advanced-Co"]["script"] == "run_advanced_colight.py"
    assert commands["Advanced-Co"]["method_id"] == "advanced_colight"
    assert commands["Weighted-RL"]["requires_reviewer_mapping_approval"] is True
    assert commands["C2T-scalar"]["status"] == "missing_blocker"


def test_baseline_command_preview_is_non_executing_and_rooted_in_preflight():
    preview = build_baseline_command_preview(
        "Random",
        city="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        seed=3,
    )

    assert preview["executes_now"] is False
    assert "run_random.py" in preview["command"]
    assert "records/paper_final/preflight_20260602_v1" in preview["out_dir"]
    assert preview["reads_result_values"] is False


def test_baseline_command_blockers_keep_unresolved_rows_explicit():
    blockers = baseline_command_blockers(PAPER_BASELINE_COMMANDS)

    assert any("C2T-scalar" in item for item in blockers)
    assert any("Weighted-RL" in item for item in blockers)


def test_baseline_commands_reject_stage_a_diagnostic_method():
    bad = dict(PAPER_BASELINE_COMMANDS)
    bad["EnvReward-QueuePenalty-PPO"] = {
        "status": "implemented",
        "script": "pareto/rl/formal_pilot_runner.py",
        "method_id": "env_reward",
    }

    with pytest.raises(ValueError, match="not a paper baseline"):
        validate_paper_baseline_commands(bad)
