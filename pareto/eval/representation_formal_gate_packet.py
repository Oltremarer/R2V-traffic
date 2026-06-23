from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pareto.eval.formal_gate import FORMAL_THRESHOLDS


PACKET_VERSION = "representation-formal-gate-packet-v1"
APPROVAL_PHRASE = "PARETO PPO OFFLINE REPRESENTATION FORMAL-GATE REMEDIATION GO"
OBJECTIVES = ("efficiency", "safety", "fairness", "stability")
ALLOWED_PACKET_CITIES = {"jinan", "hangzhou", "newyork_28x7"}
DEFAULT_VECTOR_RUN_ID = "lowrank_iso_w100_coord050"
DEFAULT_SCALAR_RUN_ID = "film_rich_v2"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _metric_status(value: float, threshold: float) -> dict[str, Any]:
    return {
        "value": float(value),
        "threshold": float(threshold),
        "pass": float(value) >= float(threshold),
    }


def _bootstrap_metric(bootstrap: dict[str, Any], section: str, key: str) -> dict[str, Any] | None:
    metric = bootstrap.get(section, {}).get("metrics", {}).get(key)
    if not isinstance(metric, dict):
        return None
    return {
        "mean": metric.get("mean"),
        "low": metric.get("low"),
        "high": metric.get("high"),
        "n": metric.get("n"),
        "n_boot": metric.get("n_boot"),
    }


def _split_check(split_report: dict[str, Any]) -> dict[str, Any]:
    split_counts = split_report.get("split_counts", {})
    sample_overlap = split_report.get("sample_overlap", {})
    required_splits_present = all(int(split_counts.get(name, 0)) > 0 for name in ("train", "val", "test"))
    pass_check = (
        split_report.get("group_key") == "time_block"
        and int(split_report.get("group_count", 0)) >= 8
        and required_splits_present
        and sample_overlap == {}
    )
    return {
        "pass": pass_check,
        "group_key": split_report.get("group_key"),
        "group_count": split_report.get("group_count"),
        "time_block_size": split_report.get("time_block_size"),
        "split_counts": split_counts,
        "sample_overlap": sample_overlap,
    }


def _normalizer_check(normalizer: dict[str, Any]) -> dict[str, Any]:
    fit_inputs = list(normalizer.get("fit_input_files", []))
    train_only = bool(fit_inputs) and all("train_raw.jsonl" in str(path) for path in fit_inputs)
    no_eval_inputs = not any(
        token in str(path)
        for path in fit_inputs
        for token in ("val_raw.jsonl", "test_raw.jsonl")
    )
    zero_iqr = list(normalizer.get("zero_iqr_objectives", []))
    valid_count = dict(normalizer.get("valid_count", {}))
    valid_counts_ok = all(int(valid_count.get(name, 0)) > 0 for name in OBJECTIVES)
    return {
        "pass": train_only and no_eval_inputs and not zero_iqr and valid_counts_ok,
        "fit_input_files": fit_inputs,
        "train_only": train_only,
        "no_eval_inputs": no_eval_inputs,
        "zero_iqr_objectives": zero_iqr,
        "valid_count": valid_count,
        "normalizer_hash": normalizer.get("hash"),
    }


def _pair_report_check(pair_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    split_status: dict[str, Any] = {}
    all_pass = True
    for split, report in pair_reports.items():
        counts = report.get("counts", {})
        objective_counts = report.get("objective_counts", {})
        strategy_counts = report.get("sampling_strategy_counts", {})
        reversal_by_template_pair = report.get("reversal_by_template_pair", {})
        split_pass = (
            int(report.get("error_count", 0)) == 0
            and int(report.get("serialized_tie_count", 0)) == 0
            and int(report.get("invalid_objective_pair_count", 0)) == 0
            and all(int(objective_counts.get(name, 0)) > 0 for name in OBJECTIVES)
            and int(counts.get("preference_pairs", 0)) > 0
            and int(counts.get("dominance_pairs", 0)) > 0
            and int(counts.get("reversal_pairs", 0)) > 0
            and int(strategy_counts.get("eff_controlled_fairness", 0)) > 0
            and int(strategy_counts.get("eff_controlled_stability", 0)) > 0
            and int(strategy_counts.get("efficiency_safety_conflict", 0)) > 0
            and int(reversal_by_template_pair.get("efficiency__stability", 0)) > 0
        )
        all_pass = all_pass and split_pass
        split_status[split] = {
            "pass": split_pass,
            "counts": counts,
            "objective_counts": objective_counts,
            "positive_ratio_by_objective": report.get("positive_ratio_by_objective", {}),
            "positive_ratio_by_strategy": report.get("positive_ratio_by_strategy", {}),
            "sampling_strategy_counts": strategy_counts,
            "reversal_by_template_pair": reversal_by_template_pair,
            "serialized_tie_count": report.get("serialized_tie_count"),
            "invalid_objective_pair_count": report.get("invalid_objective_pair_count"),
            "error_count": report.get("error_count"),
        }
    return {"pass": all_pass, "splits": split_status}


def _calibration_status(metrics: dict[str, Any]) -> dict[str, Any]:
    stds = [float(value) for value in metrics.get("output_std_per_head", [])]
    finite_positive = bool(stds) and all(value > 1e-3 for value in stds)
    spread = max(stds) / min(stds) if finite_positive else None
    return {
        "pass": finite_positive,
        "output_std_per_head": stds,
        "max_min_std_ratio": spread,
        "interpretation": "nonzero head outputs; ratio is reported as calibration context only",
    }


def _condscalar_status(
    vector_metrics: dict[str, Any],
    vector_metadata: dict[str, Any],
    scalar_metadata: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    vector_param_count = int(vector_metadata.get("param_count", 0))
    scalar_param_count = int(scalar_metadata.get("param_count", 0))
    relative_gap = (
        abs(vector_param_count - scalar_param_count) / max(vector_param_count, scalar_param_count)
        if max(vector_param_count, scalar_param_count) > 0
        else None
    )
    vector_rev = float(vector_metrics.get("rev_acc", 0.0))
    scalar_rev = float(bootstrap.get("cond_scalar", {}).get("metrics", {}).get("rev_acc", {}).get("mean", 0.0))
    return {
        "pass": relative_gap is not None and relative_gap <= 0.50 and scalar_rev <= vector_rev + 0.05,
        "vector_param_count": vector_param_count,
        "condscalar_param_count": scalar_param_count,
        "relative_param_gap": relative_gap,
        "condscalar_architecture": scalar_metadata.get("architecture"),
        "vector_rev_acc": vector_rev,
        "condscalar_rev_acc": scalar_rev,
        "condscalar_not_clearly_stronger": scalar_rev <= vector_rev + 0.05,
    }


def _bootstrap_lower_status(threshold_checks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    status: dict[str, Any] = {}
    for key, check in threshold_checks.items():
        bootstrap = check.get("bootstrap") or {}
        low = bootstrap.get("low")
        threshold = check.get("threshold")
        status[key] = {
            "low": low,
            "threshold": threshold,
            "pass": low is not None and threshold is not None and float(low) >= float(threshold),
        }
    return status


def _bootstrap_pair_count_consistency_check(
    threshold_checks: dict[str, dict[str, Any]],
    pair_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    test_counts = pair_reports.get("test", {}).get("counts", {})
    expected = {
        "rev_acc": int(test_counts.get("reversal_pairs", 0)),
        "dpr_head": int(test_counts.get("dominance_pairs", 0)),
        "dpr_utility": int(test_counts.get("dominance_pairs", 0)) * 5,
        "pref_acc": int(test_counts.get("preference_pairs", 0)),
        "obj_acc_mean": int(test_counts.get("objective_pairs", 0)),
        "head_leakage_diag_offdiag_gap": int(test_counts.get("objective_pairs", 0)),
    }
    metrics: dict[str, dict[str, Any]] = {}
    all_pass = True
    for key, expected_n in expected.items():
        bootstrap = threshold_checks.get(key, {}).get("bootstrap") or {}
        observed_n = bootstrap.get("n")
        passed = observed_n is not None and int(observed_n) == expected_n
        metrics[key] = {
            "observed_n": observed_n,
            "expected_n": expected_n,
            "pass": passed,
        }
        all_pass = all_pass and passed
    return {
        "pass": all_pass,
        "test_pair_counts": dict(test_counts),
        "metrics": metrics,
        "interpretation": (
            "Each formal-gate bootstrap sample count must match the same test pair set reported by "
            "pair_coverage_check. DPR_utility uses five fixed preference templates per dominance pair."
        ),
    }


def _dpr_head_margin_status(threshold_checks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    check = threshold_checks["dpr_head"]
    bootstrap = check.get("bootstrap") or {}
    value = float(check["value"])
    threshold = float(check["threshold"])
    low = bootstrap.get("low")
    return {
        "value": value,
        "threshold": threshold,
        "point_margin": value - threshold,
        "bootstrap_low": low,
        "bootstrap_low_margin": (float(low) - threshold) if low is not None else None,
        "bootstrap_low_pass": low is not None and float(low) >= threshold,
    }


def _boundary_note(threshold_checks: dict[str, dict[str, Any]]) -> str:
    dpr = _dpr_head_margin_status(threshold_checks)
    if abs(dpr["point_margin"]) < 1e-9:
        return (
            "DPR_head equals the formal threshold in the selected test point estimate; "
            "fresh external-reviewer approval is required before any formal experiment."
        )
    if dpr["point_margin"] > 0 and dpr["bootstrap_low_pass"]:
        return (
            "DPR_head point estimate is above the formal threshold and its bootstrap lower bound clears "
            "the threshold; fresh external-reviewer approval is still required before any formal experiment."
        )
    if dpr["point_margin"] > 0:
        return (
            "DPR_head point estimate is above the formal threshold, but its bootstrap lower bound remains "
            "below threshold; formal experiment remains blocked without further review."
        )
    return "DPR_head remains below the formal threshold; formal experiment remains blocked."


def build_packet(
    evidence_dir: str | Path,
    vector_run_id: str = DEFAULT_VECTOR_RUN_ID,
    scalar_run_id: str = DEFAULT_SCALAR_RUN_ID,
    vector_model_dir: str | None = None,
    scalar_model_dir: str | None = None,
    approval_phrase: str = APPROVAL_PHRASE,
    city: str = "jinan",
    normalizer_filename: str = "objective_norm_smoke3600.json",
) -> dict[str, Any]:
    if city not in ALLOWED_PACKET_CITIES:
        raise ValueError(f"unknown representation city: {city}")
    evidence = Path(evidence_dir)
    vector_test = _load_json(evidence / f"{vector_run_id}_diagnostics_test.json")
    vector_val = _load_json(evidence / f"{vector_run_id}_diagnostics_val.json")
    formal_gate = _load_json(evidence / f"{vector_run_id}_formal_gate_decision.json")
    bootstrap = _load_json(evidence / f"{vector_run_id}_pair_bootstrap_test.json")
    dominance_audit = _load_json(evidence / f"{vector_run_id}_dominance_error_audit.json")
    split_report = _load_json(evidence / "split_records_report.json")
    normalizer = _load_json(evidence / normalizer_filename)
    objective_sanity = _load_json(evidence / "objective_sanity_v4_train.json")
    vector_metadata = _load_json(evidence / f"{vector_run_id}_metadata.json")
    scalar_metadata = _load_json(evidence / f"{scalar_run_id}_metadata.json")
    pair_reports = {
        "train": _load_json(evidence / "pair_report_train.json"),
        "val": _load_json(evidence / "pair_report_val.json"),
        "test": _load_json(evidence / "pair_report_test.json"),
    }

    threshold_checks = {
        key: {
            **_metric_status(float(vector_test.get(key, 0.0)), threshold),
            "bootstrap": _bootstrap_metric(bootstrap, "vector", key),
        }
        for key, threshold in FORMAL_THRESHOLDS.items()
    }
    bootstrap_lower_checks = _bootstrap_lower_status(threshold_checks)
    bootstrap_pair_count_consistency = _bootstrap_pair_count_consistency_check(threshold_checks, pair_reports)
    formal_representation_pass = bool(formal_gate.get("representation_gate_pass", False)) and all(
        item["pass"] for item in threshold_checks.values()
    ) and all(item["pass"] for item in bootstrap_lower_checks.values()) and bootstrap_pair_count_consistency["pass"]
    dpr_head_margin = _dpr_head_margin_status(threshold_checks)

    packet = {
        "packet_version": PACKET_VERSION,
        "pro_approval_phrase": approval_phrase,
        "scope": {
            "city": f"{city} paper-final offline records only"
            if normalizer_filename == "objective_norm_paper_final.json"
            else "Jinan offline records only",
            "seed": 0,
            "formal_experiment_executed": False,
            "new_cityflow_ppo_run_executed": False,
            "multi_seed_executed": False,
            "city_expansion_executed": False,
            "traffic_result_value_reading_executed": False,
            "method_ranking_executed": False,
            "paper_result_claim": False,
        },
        "selected_vector_run": {
            "run_id": vector_run_id,
            "model_dir": vector_model_dir
            or f"model_weights/pareto_quality/jinan/formal_gate_remediation/{vector_run_id}",
            "architecture": vector_metadata.get("architecture"),
            "score_mode": vector_metadata.get("score_mode"),
            "isotonic_dominance_weight": vector_metadata.get("isotonic_dominance_weight"),
            "dominance_coord_loss_weight": vector_metadata.get("dominance_coord_loss_weight"),
            "dominance_utility_loss_weight": vector_metadata.get("dominance_utility_loss_weight"),
            "metadata": vector_metadata,
        },
        "dangerous_scalar_baseline": {
            "run_id": scalar_run_id,
            "model_dir": scalar_model_dir or f"model_weights/cond_scalar/jinan/preformal_final/{scalar_run_id}",
            "metadata": scalar_metadata,
        },
        "threshold_checks": threshold_checks,
        "bootstrap_lower_bound_checks": bootstrap_lower_checks,
        "bootstrap_pair_count_consistency_check": bootstrap_pair_count_consistency,
        "dpr_head_margin_status": dpr_head_margin,
        "formal_gate_decision": formal_gate,
        "formal_representation_pass": formal_representation_pass,
        "formal_experiment_requires_new_pro_approval": True,
        "split_leakage_check": _split_check(split_report),
        "train_only_normalizer_check": _normalizer_check(normalizer),
        "pair_coverage_check": _pair_report_check(pair_reports),
        "condscalar_baseline_fairness_status": _condscalar_status(
            vector_test,
            vector_metadata,
            scalar_metadata,
            bootstrap,
        ),
        "calibration_status": _calibration_status(vector_test),
        "dominance_audit_summary": {
            "DPR_head": dominance_audit.get("audit", {}).get("DPR_head"),
            "DPR_head_by_objective": dominance_audit.get("audit", {}).get("DPR_head_by_objective"),
            "DPR_utility_all_templates": dominance_audit.get("audit", {}).get("DPR_utility_all_templates"),
            "violation_by_margin_bin": dominance_audit.get("audit", {}).get("violation_by_margin_bin"),
            "violation_rate_by_objective": dominance_audit.get("audit", {}).get("violation_rate_by_objective"),
        },
        "objective_sanity_status": {
            "pass": objective_sanity.get("strict_failures") == [],
            "strict_failures": objective_sanity.get("strict_failures"),
            "warnings": objective_sanity.get("warnings"),
            "safety_valid_rate": objective_sanity.get("safety_valid_rate"),
            "objective_correlations": objective_sanity.get("objective_correlations"),
        },
        "vector_metrics": {
            "val": vector_val,
            "test": vector_test,
            "test_bootstrap": bootstrap.get("vector", {}).get("metrics", {}),
        },
        "condscalar_metrics": {
            "test_bootstrap": bootstrap.get("cond_scalar", {}).get("metrics", {}),
        },
        "interpretation": {
            "representation_gate_point_estimate": "pass" if formal_representation_pass else "fail",
            "dpr_head_margin_status": dpr_head_margin,
            "boundary_note": _boundary_note(threshold_checks),
            "allowed_next_action_requested": "external-reviewer review of representation packet only",
        },
    }
    return packet


def validate_packet(packet: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if packet.get("packet_version") != PACKET_VERSION:
        failures.append("packet version mismatch")
    if not packet.get("pro_approval_phrase"):
        failures.append("approval phrase missing")
    scope = packet.get("scope", {})
    for key in (
        "formal_experiment_executed",
        "new_cityflow_ppo_run_executed",
        "multi_seed_executed",
        "city_expansion_executed",
        "traffic_result_value_reading_executed",
        "method_ranking_executed",
        "paper_result_claim",
    ):
        if scope.get(key) is not False:
            failures.append(f"scope flag must be false: {key}")
    for key, status in packet.get("threshold_checks", {}).items():
        if "value" not in status or "threshold" not in status or "pass" not in status:
            failures.append(f"incomplete threshold status: {key}")
    for key in (
        "split_leakage_check",
        "train_only_normalizer_check",
        "pair_coverage_check",
        "bootstrap_pair_count_consistency_check",
        "condscalar_baseline_fairness_status",
        "calibration_status",
    ):
        if "pass" not in packet.get(key, {}):
            failures.append(f"missing pass status: {key}")
    if packet.get("formal_experiment_requires_new_pro_approval") is not True:
        failures.append("packet must require fresh Pro approval before formal experiment")
    return failures


def write_markdown(packet: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checks = packet["threshold_checks"]
    vector = packet["vector_metrics"]["test"]
    bootstrap = packet["vector_metrics"]["test_bootstrap"]
    cond = packet["condscalar_metrics"]["test_bootstrap"]
    lines = [
        "# Pareto Offline Representation Formal-Gate Packet",
        "",
        "Scope: offline representation gate remediation only. No new CityFlow PPO run, formal PPO run, multi-seed run, city expansion, traffic result value reading, ranking, or paper-result claim was performed.",
        "",
        "## Formal Gate Status",
        "",
        f"- formal_representation_pass: `{str(packet['formal_representation_pass']).lower()}`",
        f"- formal experiment still requires new Pro approval: `{str(packet['formal_experiment_requires_new_pro_approval']).lower()}`",
        f"- gate decision claim mode: `{packet['formal_gate_decision'].get('claim_mode')}`",
        f"- failed reasons: `{packet['formal_gate_decision'].get('failed_reasons')}`",
        "",
        "## Selected VectorQ Run",
        "",
        f"- run_id: `{packet['selected_vector_run']['run_id']}`",
        f"- architecture: `{packet['selected_vector_run']['architecture']}`",
        f"- score_mode: `{packet['selected_vector_run']['score_mode']}`",
        f"- isotonic_dominance_weight: `{packet['selected_vector_run']['isotonic_dominance_weight']}`",
        f"- dominance_coord_loss_weight: `{packet['selected_vector_run']['dominance_coord_loss_weight']}`",
        f"- dominance_utility_loss_weight: `{packet['selected_vector_run']['dominance_utility_loss_weight']}`",
        "",
        "## Threshold Checks",
        "",
        "| Metric | Test Value | Threshold | Status | Bootstrap 95% CI |",
        "|---|---:|---:|---|---|",
    ]
    for key in ("rev_acc", "dpr_head", "dpr_utility", "pref_acc", "obj_acc_mean", "head_leakage_diag_offdiag_gap"):
        status = checks[key]
        ci = status.get("bootstrap") or {}
        ci_text = ""
        if ci:
            ci_text = f"[{ci.get('low'):.4f}, {ci.get('high'):.4f}], n={ci.get('n')}"
        lines.append(
            f"| `{key}` | {status['value']:.4f} | {status['threshold']:.4f} | {'PASS' if status['pass'] else 'FAIL'} | {ci_text} |"
        )
    lines.extend([
        "",
        "## Baseline And Guard Checks",
        "",
        f"- split leakage check: `{'PASS' if packet['split_leakage_check']['pass'] else 'FAIL'}`; group_key=`{packet['split_leakage_check']['group_key']}`, sample_overlap=`{packet['split_leakage_check']['sample_overlap']}`",
        f"- train-only normalizer check: `{'PASS' if packet['train_only_normalizer_check']['pass'] else 'FAIL'}`; fit_input_files=`{packet['train_only_normalizer_check']['fit_input_files']}`",
        f"- pair coverage check: `{'PASS' if packet['pair_coverage_check']['pass'] else 'FAIL'}`",
        f"- bootstrap/pair-count consistency check: `{'PASS' if packet['bootstrap_pair_count_consistency_check']['pass'] else 'FAIL'}`; metrics=`{packet['bootstrap_pair_count_consistency_check']['metrics']}`",
        f"- CondScalar fairness status: `{'PASS' if packet['condscalar_baseline_fairness_status']['pass'] else 'FAIL'}`; vector_param_count=`{packet['condscalar_baseline_fairness_status']['vector_param_count']}`, condscalar_param_count=`{packet['condscalar_baseline_fairness_status']['condscalar_param_count']}`",
        f"- calibration status: `{'PASS' if packet['calibration_status']['pass'] else 'FAIL'}`; output_std_per_head=`{packet['calibration_status']['output_std_per_head']}`",
        f"- objective sanity strict status: `{'PASS' if packet['objective_sanity_status']['pass'] else 'FAIL'}`",
        "",
        "## Key Diagnostics",
        "",
        f"- VectorQ test Rev.Acc: `{vector.get('rev_acc')}`; bootstrap: `{bootstrap.get('rev_acc')}`",
        f"- VectorQ test DPR_head: `{vector.get('dpr_head')}`; bootstrap: `{bootstrap.get('dpr_head')}`",
        f"- VectorQ test DPR_utility: `{vector.get('dpr_utility')}`; bootstrap: `{bootstrap.get('dpr_utility')}`",
        f"- FiLM CondScalar test Rev.Acc bootstrap: `{cond.get('rev_acc')}`",
        f"- dominance audit by objective: `{packet['dominance_audit_summary']['DPR_head_by_objective']}`",
        f"- DPR_head margin status: `{packet['dpr_head_margin_status']}`",
        "",
        "## Interpretation Boundary",
        "",
        packet["interpretation"]["boundary_note"]
        + " Codex has not run any formal experiment from this packet.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--vector_run_id", default=DEFAULT_VECTOR_RUN_ID)
    parser.add_argument("--scalar_run_id", default=DEFAULT_SCALAR_RUN_ID)
    parser.add_argument("--vector_model_dir")
    parser.add_argument("--scalar_model_dir")
    parser.add_argument("--approval_phrase", default=APPROVAL_PHRASE)
    parser.add_argument("--city", default="jinan", choices=sorted(ALLOWED_PACKET_CITIES))
    parser.add_argument("--normalizer_filename", default="objective_norm_smoke3600.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet = build_packet(
        args.evidence_dir,
        vector_run_id=args.vector_run_id,
        scalar_run_id=args.scalar_run_id,
        vector_model_dir=args.vector_model_dir,
        scalar_model_dir=args.scalar_model_dir,
        approval_phrase=args.approval_phrase,
        city=args.city,
        normalizer_filename=args.normalizer_filename,
    )
    failures = validate_packet(packet)
    if failures:
        raise SystemExit("representation packet validation failed: " + "; ".join(failures))
    out_dir = Path(args.out_dir)
    _write_json(out_dir / "representation_formal_gate_packet.json", packet)
    write_markdown(packet, out_dir / "representation_formal_gate_packet.md")


if __name__ == "__main__":
    main()
