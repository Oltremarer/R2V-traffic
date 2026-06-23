import json
import importlib.util
from pathlib import Path

from pareto.eval.formal_gate import evaluate_formal_gate


def _load_check_formal_gate_main():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "check_formal_gate.py"
    spec = importlib.util.spec_from_file_location("check_formal_gate", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.main


def test_formal_gate_blocks_current_vectorq_and_allows_only_wiring_smoke():
    vector_metrics = {
        "rev_acc": 0.565,
        "dpr_head": 0.644,
        "dpr_utility": 0.922,
        "pref_acc": 0.767,
        "obj_acc_mean": 0.791,
        "head_leakage_diag_offdiag_gap": 0.300,
    }
    film_metrics = {
        "rev_acc": 0.675,
        "pref_acc": 0.758,
        "dpr_utility": 0.886,
    }

    decision = evaluate_formal_gate(vector_metrics, film_metrics)

    assert decision["representation_gate_pass"] is False
    assert decision["ppo_formal_allowed"] is False
    assert decision["wiring_smoke_allowed"] is True
    assert decision["claim_mode"] == "film_scalar_candidate"
    assert "rev_acc below formal threshold" in decision["failed_reasons"]
    assert "film condscalar clearly stronger on rev_acc" in decision["failed_reasons"]


def test_check_formal_gate_script_writes_decision(tmp_path: Path, monkeypatch):
    check_formal_gate_main = _load_check_formal_gate_main()
    vector_path = tmp_path / "vector.json"
    film_path = tmp_path / "film.json"
    out_path = tmp_path / "decision.json"
    vector_path.write_text(json.dumps({"test": {
        "rev_acc": 0.61,
        "dpr_head": 0.76,
        "dpr_utility": 0.90,
        "pref_acc": 0.72,
        "obj_acc_mean": 0.70,
        "head_leakage_diag_offdiag_gap": 0.20,
    }}), encoding="utf-8")
    film_path.write_text(json.dumps({"test": {"rev_acc": 0.60}}), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_formal_gate.py",
            "--vector_metrics",
            str(vector_path),
            "--film_metrics",
            str(film_path),
            "--out",
            str(out_path),
        ],
    )

    check_formal_gate_main()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["ppo_formal_allowed"] is True
    assert payload["claim_mode"] == "vector_superiority"
