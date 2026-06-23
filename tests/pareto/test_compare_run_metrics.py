from pareto.data.compare_run_metrics import metrics_match


def test_metrics_match_detects_identical_and_different_files(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    c = tmp_path / "c.csv"
    a.write_text("step,value\n0,1\n", encoding="utf-8")
    b.write_text("step,value\n0,1\n", encoding="utf-8")
    c.write_text("step,value\n0,2\n", encoding="utf-8")

    assert metrics_match(a, b)
    assert not metrics_match(a, c)
