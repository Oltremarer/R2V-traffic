from pareto.data.objective_sanity import strict_failures, summarize_records


def test_objective_sanity_reports_local_safety_variance():
    records = [
        {
            "sim_time_sec": 0.0,
            "objective_values_raw": {"safety": 0.0, "efficiency": 0.0, "fairness": 0.0, "stability": 0.0},
            "objective_valid_mask": {"safety": True, "efficiency": True, "fairness": True, "stability": False},
            "metadata": {"objective_debug": {"local_ttc_pair_count": 0}},
        },
        {
            "sim_time_sec": 0.0,
            "objective_values_raw": {"safety": -1.0, "efficiency": -1.0, "fairness": -0.1, "stability": -0.2},
            "objective_valid_mask": {"safety": True, "efficiency": True, "fairness": True, "stability": True},
            "metadata": {"objective_debug": {"local_ttc_pair_count": 3}},
        },
    ]

    report = summarize_records(records)

    assert report["safety_valid_rate"] == 1.0
    assert report["same_time_cross_intersection_safety_std_mean"] > 0


def test_objective_sanity_strict_failures_flag_unusable_safety():
    report = {
        "safety_valid_rate": 0.1,
        "ttc_pair_count_zero_rate": 0.9,
        "same_time_cross_intersection_safety_std_mean": 0.0,
        "objective_correlations": {
            "efficiency__fairness": 0.99,
            "efficiency__stability": 0.2,
        },
    }

    failures = strict_failures(report)

    assert any("safety_valid_rate" in failure for failure in failures)
    assert any("ttc_pair_count_zero_rate" in failure for failure in failures)
    assert any("efficiency__fairness" in failure for failure in failures)
