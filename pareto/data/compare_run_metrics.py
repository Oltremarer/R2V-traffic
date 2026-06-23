#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics_match(a: str | Path, b: str | Path) -> bool:
    return file_hash(a) == file_hash(b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--must_match", action="store_true")
    args = parser.parse_args()

    match = metrics_match(args.a, args.b)
    print({"match": match, "a_hash": file_hash(args.a), "b_hash": file_hash(args.b)})
    if args.must_match and not match:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
