import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_validate_schema_file_entrypoint_imports_package():
    result = subprocess.run(
        [sys.executable, str(ROOT / "pareto/data/validate_schema.py"), "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
    assert "--input" in result.stdout
