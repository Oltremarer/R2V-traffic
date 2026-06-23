from pareto.eval.offline_ci_report import build_offline_ci_report


def test_offline_ci_report_uses_pair_counts():
    diagnostics = {
        "pref_acc": 0.75,
        "rev_acc": 0.5,
        "dpr_head": 0.8,
        "dpr_utility": 0.9,
        "obj_acc": {
            "efficiency": 0.8,
            "safety": 0.6,
            "fairness": 0.7,
            "stability": 0.5,
        },
    }
    pair_report = {
        "preference_pairs": 20,
        "reversal_pairs": 10,
        "dominance_pairs": 5,
        "objective_counts": {
            "efficiency": 10,
            "safety": 10,
            "fairness": 10,
            "stability": 10,
        },
    }

    report = build_offline_ci_report(diagnostics, pair_report, n_boot=20, seed=0)

    assert report["metrics"]["pref_acc"]["n"] == 20
    assert report["metrics"]["dpr_utility"]["n"] == 25
    assert "safety" in report["metrics"]["obj_acc"]
