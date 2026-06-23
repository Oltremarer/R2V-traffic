from __future__ import annotations

from pathlib import Path
from typing import Any

from pareto.rl.paper_final_experiment_manifest import REQUIRED_CITY_TRAFFIC


REQUIRED_SPLITS = ("train", "val", "test")
REQUIRED_PAIR_FILES = (
    "objective_pairs.jsonl",
    "preference_pairs.jsonl",
    "dominance_pairs.jsonl",
    "reversal_pairs.jsonl",
)
RECORDS_PREFIX = "data/pareto_records_split_norm"
PAIRS_PREFIX = "data/pareto_pairs"
ALLOWED_DATA_ROOT_STATUSES = {"complete", "missing_blocker"}
ALLOWED_DATA_ROW_STATUSES = {"ready", "missing_blocker"}


def _required_files(records_root: str, pairs_root: str) -> list[str]:
    files = [f"{records_root}/{split}_raw.jsonl" for split in REQUIRED_SPLITS]
    for split in REQUIRED_SPLITS:
        files.extend(f"{pairs_root}/{split}/{filename}" for filename in REQUIRED_PAIR_FILES)
    return files


def _city_row(root: Path, city: str) -> dict[str, Any]:
    records_root = f"{RECORDS_PREFIX}/{city}/paper_final"
    pairs_root = f"{PAIRS_PREFIX}/{city}/paper_final"
    missing = [path for path in _required_files(records_root, pairs_root) if not (root / path).exists()]
    status = "missing_blocker" if missing else "ready"
    row = {
        "city": city,
        "records_root": records_root,
        "pairs_root": pairs_root,
        "status": status,
        "missing_files": missing,
        "reads_final_traffic_result_values": False,
        "creates_files": False,
    }
    if missing:
        row["blocker"] = "paper_final records/pairs roots incomplete"
    return row


def build_paper_final_data_root_audit(root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    rows = [_city_row(root_path, city) for city in REQUIRED_CITY_TRAFFIC]
    audit = {
        "packet_type": "paper_final_data_root_audit",
        "status": "missing_blocker" if any(row["status"] == "missing_blocker" for row in rows) else "complete",
        "root": root_path.as_posix(),
        "rows": rows,
        "reads_final_traffic_result_values": False,
        "creates_files": False,
    }
    return validate_paper_final_data_root_audit(audit)


def validate_paper_final_data_root_audit(audit: dict[str, Any]) -> dict[str, Any]:
    if audit.get("status") not in ALLOWED_DATA_ROOT_STATUSES:
        raise ValueError(f"unknown paper-final data root audit status: {audit.get('status')}")
    if audit.get("reads_final_traffic_result_values") is not False:
        raise ValueError("paper-final data root audit must not read final traffic result values")
    if audit.get("creates_files") is not False:
        raise ValueError("paper-final data root audit must not create files")
    rows = audit.get("rows") or []
    observed = {row.get("city") for row in rows}
    missing_cities = sorted(set(REQUIRED_CITY_TRAFFIC) - observed)
    if missing_cities:
        raise ValueError(f"paper-final data root audit missing cities: {missing_cities}")
    for row in rows:
        city = row.get("city")
        if city not in REQUIRED_CITY_TRAFFIC:
            raise ValueError(f"unknown paper-final data root city: {city}")
        if row.get("status") not in ALLOWED_DATA_ROW_STATUSES:
            raise ValueError(f"unknown paper-final data root row status: {row.get('status')}")
        records_root = str(row.get("records_root") or "")
        pairs_root = str(row.get("pairs_root") or "")
        if not records_root.startswith(f"{RECORDS_PREFIX}/{city}/"):
            raise ValueError("records_root must be under data/pareto_records_split_norm")
        if not pairs_root.startswith(f"{PAIRS_PREFIX}/{city}/"):
            raise ValueError("pairs_root must be under data/pareto_pairs")
        if row.get("reads_final_traffic_result_values") is not False:
            raise ValueError("paper-final data root row must not read final traffic result values")
        if row.get("creates_files") is not False:
            raise ValueError("paper-final data root row must not create files")
        if row["status"] == "missing_blocker" and not row.get("blocker"):
            raise ValueError("missing paper-final data root row must include blocker")
    return dict(audit)


def data_root_blockers(audit: dict[str, Any]) -> list[str]:
    validated = validate_paper_final_data_root_audit(audit)
    blockers: list[str] = []
    for row in validated["rows"]:
        if row["status"] == "missing_blocker":
            blockers.append(f"{row['city']}: {row['blocker']}")
    return blockers
