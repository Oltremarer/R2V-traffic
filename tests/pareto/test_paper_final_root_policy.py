from __future__ import annotations

from pathlib import Path

import pytest

from pareto.rl.paper_final_root_policy import (
    build_paper_final_roots,
    ensure_paper_final_root_empty,
    traffic_slug,
    validate_not_stage_a_root,
)


def test_build_paper_final_roots_include_city_traffic_method_seed_and_preference():
    roots = build_paper_final_roots(
        city="newyork_28x7",
        traffic_file="anon_28_7_newyork_real_double.json",
        method="VectorQ-PPO",
        seed=4,
        preference_id="balanced",
    )

    assert roots.train == Path("records/paper_final/train_20260602_v1/newyork_28x7/anon_28_7_newyork_real_double/VectorQ-PPO/seed4")
    assert roots.eval == Path("records/paper_final/eval_20260602_v1/newyork_28x7/anon_28_7_newyork_real_double/VectorQ-PPO/seed4/balanced")
    assert roots.diagnostics == Path("records/paper_final/diagnostics_20260602_v1/newyork_28x7/anon_28_7_newyork_real_double/VectorQ-PPO/seed4")


def test_non_balanced_preference_gets_distinct_train_root():
    roots = build_paper_final_roots(
        city="jinan",
        traffic_file="anon_3_4_jinan_real.json",
        method="Weighted-RL",
        seed=0,
        preference_id="efficiency_focused",
    )

    assert roots.train == Path(
        "records/paper_final/train_20260602_v1/jinan/anon_3_4_jinan_real/Weighted-RL/seed0/efficiency_focused"
    )
    assert roots.diagnostics == Path(
        "records/paper_final/diagnostics_20260602_v1/jinan/anon_3_4_jinan_real/Weighted-RL/seed0/efficiency_focused"
    )
    assert roots.eval == Path(
        "records/paper_final/eval_20260602_v1/jinan/anon_3_4_jinan_real/Weighted-RL/seed0/efficiency_focused"
    )


def test_traffic_slug_removes_json_suffix_only():
    assert traffic_slug("anon_4_4_hangzhou_real.json") == "anon_4_4_hangzhou_real"


def test_root_policy_rejects_stage_a_roots():
    with pytest.raises(ValueError, match="Stage-A root"):
        validate_not_stage_a_root(Path("records/formal_jinan_3seed_guarded_20260602_stageA_revised_small_v1"))


def test_root_policy_rejects_non_empty_root(tmp_path: Path):
    root = tmp_path / "records" / "paper_final" / "train_20260602_v1"
    root.mkdir(parents=True)
    (root / "metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="must be empty"):
        ensure_paper_final_root_empty(root)


def test_root_policy_allows_absent_or_empty_root(tmp_path: Path):
    absent = tmp_path / "absent"
    ensure_paper_final_root_empty(absent)

    empty = tmp_path / "empty"
    empty.mkdir()
    ensure_paper_final_root_empty(empty)
