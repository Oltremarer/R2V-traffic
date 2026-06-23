from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict

from pareto.common.io import write_json


def stable_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def write_run_metadata(run_dir: str | Path, command: str, config: Dict[str, Any]) -> Dict[str, Any]:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "command.txt").write_text(command + "\n", encoding="utf-8")
    metadata = {
        "command": command,
        "config_hash": stable_hash(config),
        "git_commit": git_commit(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }
    write_json(run_path / "metadata.json", metadata)
    return metadata
