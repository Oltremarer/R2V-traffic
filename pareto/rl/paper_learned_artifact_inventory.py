from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pareto.rl.paper_final_experiment_manifest import REQUIRED_CITY_TRAFFIC


REQUIRED_LEARNED_ARTIFACT_BASELINES = ("Cond-Scalar-RL", "VectorQ-PPO")
LEARNED_ARTIFACT_FAMILIES = {
    "Cond-Scalar-RL": "cond_scalar",
    "VectorQ-PPO": "pareto_quality",
}
NORMALIZER_FILENAMES = (
    "objective_normalizer.json",
    "objective_normalizer.pt",
    "objective_normalizer.pkl",
    "objective_normalizer.npz",
)
ALL_PAPER_CITIES = tuple(REQUIRED_CITY_TRAFFIC)
ALLOWED_LEARNED_ARTIFACT_STATUSES = {"implemented_guarded_preview", "missing_blocker"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _objective_normalizer_hash(path: Path) -> str | None:
    if path.suffix != ".json":
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    value = payload.get("hash")
    return str(value) if value is not None else None


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _normalizer_for(run_dir: Path) -> Path | None:
    for filename in NORMALIZER_FILENAMES:
        candidate = run_dir / filename
        if candidate.exists():
            return candidate
    return None


def _missing_row(baseline: str, city: str, blocker: str, required_glob: str) -> dict[str, Any]:
    return validate_learned_artifact_row(
        {
            "baseline": baseline,
            "city": city,
            "status": "missing_blocker",
            "blocker": blocker,
            "required_glob": required_glob,
            "executes_training_now": False,
        }
    )


def _inventory_city(root: Path, baseline: str, city: str) -> dict[str, Any]:
    family = LEARNED_ARTIFACT_FAMILIES[baseline]
    paper_final_root = root / "model_weights" / family / city / "paper_final"
    required_glob = f"model_weights/{family}/{city}/paper_final/<run_id>/model.pt"
    model_paths = sorted(paper_final_root.glob("*/model.pt"))
    if not model_paths:
        return _missing_row(baseline, city, "model hash missing", required_glob)

    artifacts: list[dict[str, Any]] = []
    missing_normalizer = False
    for model_path in model_paths:
        normalizer_path = _normalizer_for(model_path.parent)
        artifact = {
            "run_id": model_path.parent.name,
            "model_path": _rel(model_path, root),
            "model_hash": _sha256(model_path),
        }
        if normalizer_path is None:
            missing_normalizer = True
        else:
            normalizer_hash = _objective_normalizer_hash(normalizer_path)
            artifact["objective_normalizer_path"] = _rel(normalizer_path, root)
            artifact["objective_normalizer_file_sha256"] = _sha256(normalizer_path)
            if normalizer_hash is None:
                missing_normalizer = True
            else:
                artifact["objective_normalizer_hash"] = normalizer_hash
        artifacts.append(artifact)

    first = artifacts[0]
    if missing_normalizer:
        row = {
            "baseline": baseline,
            "city": city,
            "status": "missing_blocker",
            "blocker": "objective normalizer hash missing",
            "model_path": first["model_path"],
            "model_hash": first["model_hash"],
            "objective_normalizer_path": first.get("objective_normalizer_path"),
            "objective_normalizer_file_sha256": first.get("objective_normalizer_file_sha256"),
            "artifacts": artifacts,
            "required_glob": required_glob,
            "executes_training_now": False,
        }
        return validate_learned_artifact_row(row)

    row = {
        "baseline": baseline,
        "city": city,
        "status": "implemented_guarded_preview",
        "model_path": first["model_path"],
        "model_hash": first["model_hash"],
        "objective_normalizer_path": first["objective_normalizer_path"],
        "objective_normalizer_hash": first["objective_normalizer_hash"],
        "objective_normalizer_file_sha256": first["objective_normalizer_file_sha256"],
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "executes_training_now": False,
    }
    return validate_learned_artifact_row(row)


def inventory_learned_artifacts(root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    rows = [
        _inventory_city(root_path, baseline, city)
        for baseline in REQUIRED_LEARNED_ARTIFACT_BASELINES
        for city in ALL_PAPER_CITIES
    ]
    audit = {
        "packet_type": "paper_learned_artifact_inventory",
        "root": root_path.as_posix(),
        "required_baselines": list(REQUIRED_LEARNED_ARTIFACT_BASELINES),
        "required_cities": list(ALL_PAPER_CITIES),
        "rows": rows,
        "executes_training_now": False,
    }
    audit["coverage_status"] = "missing_blocker" if learned_artifact_blockers(audit) else "complete"
    return audit


def validate_learned_artifact_row(row: dict[str, Any]) -> dict[str, Any]:
    baseline = row.get("baseline")
    city = row.get("city")
    status = row.get("status")
    if baseline not in REQUIRED_LEARNED_ARTIFACT_BASELINES:
        raise ValueError(f"unknown learned artifact baseline: {baseline}")
    if city not in ALL_PAPER_CITIES:
        raise ValueError(f"unknown learned artifact city: {city}")
    if status not in ALLOWED_LEARNED_ARTIFACT_STATUSES:
        raise ValueError(f"unknown learned artifact status: {status}")
    if row.get("executes_training_now") is not False:
        raise ValueError("learned artifact inventory must be non-executing")
    family = LEARNED_ARTIFACT_FAMILIES[str(baseline)]
    expected_prefix = f"model_weights/{family}/{city}/paper_final/"
    if status == "implemented_guarded_preview":
        model_path = str(row.get("model_path") or "")
        normalizer_path = str(row.get("objective_normalizer_path") or "")
        if not model_path.startswith(expected_prefix) or not model_path.endswith("/model.pt"):
            raise ValueError("learned artifact model path must be under paper_final root")
        if not normalizer_path.startswith(expected_prefix):
            raise ValueError("learned artifact objective normalizer path must be under paper_final root")
        if len(str(row.get("model_hash") or "")) != 64:
            raise ValueError("implemented learned artifact missing model hash")
        if not str(row.get("objective_normalizer_hash") or ""):
            raise ValueError("implemented learned artifact missing objective normalizer hash")
        if len(str(row.get("objective_normalizer_file_sha256") or "")) != 64:
            raise ValueError("implemented learned artifact missing objective normalizer file sha256")
    if status == "missing_blocker" and not row.get("blocker"):
        raise ValueError("missing learned artifact row must include blocker")
    return dict(row)


def learned_artifact_blockers(audit: dict[str, Any]) -> list[str]:
    rows = audit.get("rows") or []
    blockers: list[str] = []
    for row in rows:
        validated = validate_learned_artifact_row(row)
        if validated["status"] == "missing_blocker":
            blockers.append(f"{validated['baseline']} {validated['city']}: {validated['blocker']}")
    return blockers
