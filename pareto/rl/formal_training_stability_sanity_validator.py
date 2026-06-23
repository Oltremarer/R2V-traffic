from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TRAINING_STABILITY_PASS = "FORMAL_JINAN_TRAINING_STABILITY_SANITY_PASS"
TRAINING_STABILITY_FAIL = "FORMAL_JINAN_TRAINING_STABILITY_SANITY_FAIL"
EXPECTED_TRAINING_STABILITY_METHODS = frozenset({"film_scalar_potential", "weighted_proxy", "env_reward"})
EXPECTED_TRAINING_STABILITY_SEEDS = frozenset({0, 1, 2})
EXPECTED_TRAINING_STABILITY_RUN_COUNT = len(EXPECTED_TRAINING_STABILITY_METHODS) * len(EXPECTED_TRAINING_STABILITY_SEEDS)
EXPECTED_TRAINING_STABILITY_LOSS_DEBUG_ROWS = 480
REQUIRED_ZERO_TOTAL_KEYS = {
    "failed_run_count",
    "warn_run_count",
    "nonfinite_count",
    "threshold_violation_count",
    "missing_allowed_field_count",
}
ALLOWED_TRAINING_STABILITY_OUTPUTS = {
    "formal_jinan_training_stability_sanity.json",
    "formal_jinan_training_stability_sanity_packet.md",
}
FORBIDDEN_VALUE_KEYS = {
    "field_values",
    "max_values",
    "mean_values",
    "metric_values",
    "min_values",
    "raw_values",
    "sample_values",
    "std_values",
    "values",
}
FORBIDDEN_PACKET_WORDING = {
    "best method",
    "beats",
    "outperforms",
    "ranked",
    "leaderboard",
    "traffic improvement",
    "state-of-the-art",
    "better than",
    "wins",
    "performance gain",
    "main result",
    "paper result",
}


def _walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(_walk_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.extend(_walk_keys(item))
    return keys


def training_stability_coverage_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    totals = payload.get("totals")
    if isinstance(totals, dict):
        run_count = totals.get("run_count")
        if run_count != EXPECTED_TRAINING_STABILITY_RUN_COUNT:
            errors.append(
                f"run_count must be {EXPECTED_TRAINING_STABILITY_RUN_COUNT}, got {run_count}"
            )

    runs = payload.get("runs")
    if not isinstance(runs, list):
        errors.append("runs must be a list")
        return errors
    if len(runs) != EXPECTED_TRAINING_STABILITY_RUN_COUNT:
        errors.append(
            f"runs length must be {EXPECTED_TRAINING_STABILITY_RUN_COUNT}, got {len(runs)}"
        )

    seen_pairs: set[tuple[int, str]] = set()
    for idx, run in enumerate(runs):
        if not isinstance(run, dict):
            errors.append(f"run {idx} is not an object")
            continue
        method = run.get("method")
        seed = run.get("seed")
        if method is None:
            errors.append(f"run {idx} missing method")
        elif method not in EXPECTED_TRAINING_STABILITY_METHODS:
            errors.append(f"run {idx} unexpected method {method!r}")
        if seed is None:
            errors.append(f"run {idx} missing seed")
        elif not isinstance(seed, int) or isinstance(seed, bool):
            errors.append(f"run {idx} seed must be an integer")
        elif seed not in EXPECTED_TRAINING_STABILITY_SEEDS:
            errors.append(f"run {idx} unexpected seed {seed!r}")

        if method in EXPECTED_TRAINING_STABILITY_METHODS and isinstance(seed, int) and not isinstance(seed, bool):
            pair = (seed, method)
            if pair in seen_pairs:
                errors.append(f"duplicate method/seed pair: seed={seed}, method={method}")
            else:
                seen_pairs.add(pair)

        for field in ("loss_debug_rows", "observed_row_count"):
            value = run.get(field)
            if value != EXPECTED_TRAINING_STABILITY_LOSS_DEBUG_ROWS or isinstance(value, bool):
                errors.append(
                    f"run {idx} {field} must be {EXPECTED_TRAINING_STABILITY_LOSS_DEBUG_ROWS}, got {value}"
                )

    expected_pairs = {
        (seed, method)
        for seed in EXPECTED_TRAINING_STABILITY_SEEDS
        for method in EXPECTED_TRAINING_STABILITY_METHODS
    }
    missing_pairs = sorted(expected_pairs - seen_pairs)
    if missing_pairs:
        errors.append(f"missing method/seed pairs: {missing_pairs}")
    return errors


def validate_training_stability_packet(out_dir: str | Path) -> None:
    root = Path(out_dir)
    existing = {path.name for path in root.iterdir() if path.is_file()}
    unexpected = sorted(existing - ALLOWED_TRAINING_STABILITY_OUTPUTS)
    if unexpected:
        raise ValueError(f"unexpected training-stability outputs: {unexpected}")
    missing = sorted(ALLOWED_TRAINING_STABILITY_OUTPUTS - existing)
    if missing:
        raise ValueError(f"missing training-stability outputs: {missing}")

    payload = json.loads((root / "formal_jinan_training_stability_sanity.json").read_text(encoding="utf-8"))
    if payload.get("report_status") != TRAINING_STABILITY_PASS:
        raise ValueError("training-stability sanity did not pass")
    if payload.get("scope") != "training_stability_sanity_only_no_method_comparison":
        raise ValueError("training-stability sanity scope mismatch")
    totals = payload.get("totals")
    if not isinstance(totals, dict):
        raise ValueError("training-stability sanity missing totals")
    for key in sorted(REQUIRED_ZERO_TOTAL_KEYS):
        if key not in totals:
            raise ValueError(f"training-stability totals missing {key}")
        value = totals[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"training-stability total {key} must be an integer")
        if value != 0:
            raise ValueError(f"training-stability total {key} must be zero")
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("training-stability sanity missing runs")
    coverage_errors = training_stability_coverage_errors(payload)
    if coverage_errors:
        raise ValueError(f"training-stability coverage mismatch: {coverage_errors}")
    for idx, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"training-stability run {idx} is not an object")
        if run.get("pass_fail") != "PASS":
            raise ValueError(f"training-stability run {idx} pass_fail must be PASS")
    leaked_keys = sorted({key for key in _walk_keys(payload) if key.lower() in FORBIDDEN_VALUE_KEYS})
    if leaked_keys:
        raise ValueError(f"training-stability output contains forbidden value carrier keys: {leaked_keys}")

    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in root.iterdir() if path.is_file())
    wording_hits = sorted(word for word in FORBIDDEN_PACKET_WORDING if word in text)
    if wording_hits:
        raise ValueError(f"training-stability packet contains forbidden wording: {wording_hits}")
