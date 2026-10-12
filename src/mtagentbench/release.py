#!/usr/bin/env python3
"""Build the registry for canonical frozen public dataset splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mtagentbench.batch_evaluation import read_json_or_jsonl
from mtagentbench.dataset_builders.common import (
    CANONICAL_SPLITS,
    sha256_file,
    write_json,
)


FROZEN_SPLITS = CANONICAL_SPLITS


def build_registry(root: Path) -> dict[str, Any]:
    splits: list[dict[str, Any]] = []
    for spec in FROZEN_SPLITS:
        directory = root / spec.public_directory
        public_path = directory / "public_cases.jsonl"
        rows = read_json_or_jsonl(public_path)
        turns = {len(row.get("turns") or []) for row in rows}
        if turns != {7}:
            raise ValueError(
                f"{spec.public_directory}: expected exactly seven turns per case"
            )
        splits.append({
            "domain": spec.domain,
            "split": spec.split,
            "rows": len(rows),
            "turns_per_case": 7,
            "public_path": public_path.relative_to(root).as_posix(),
            "public_sha256": sha256_file(public_path),
        })
    return {
        "schema_version": 1,
        "registry": "GitHarness frozen public datasets",
        "splits": splits,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("datasets/registry.json"))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    write_json(output, build_registry(root))
    print(json.dumps({"output": str(output), "splits": len(FROZEN_SPLITS)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
