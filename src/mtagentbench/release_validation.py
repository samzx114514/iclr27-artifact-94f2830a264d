#!/usr/bin/env python3
"""Validate the registry and all canonical frozen dataset releases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from mtagentbench.dataset_builders import code as code_builder
from mtagentbench.dataset_builders import math as math_builder
from mtagentbench.dataset_builders import research as research_builder
from mtagentbench.dataset_builders import search as search_builder
from mtagentbench.dataset_builders import sql as sql_builder
from mtagentbench.dataset_builders.common import (
    CANONICAL_SPLITS,
    read_json_object,
    validate_public_split,
)
from mtagentbench.validation import validate_dataset
from .release import build_registry


def _scenario_checks(
    root: Path, rows_by_split: Mapping[tuple[str, str], list[Mapping[str, Any]]],
    errors: list[str],
) -> None:
    errors.extend(math_builder.validate_release(
        root,
        rows_by_split[("math", "train")],
        rows_by_split[("math", "validation")],
        rows_by_split[("math", "test")],
    ))
    errors.extend(sql_builder.validate_release(
        root,
        rows_by_split[("sql", "train")],
        rows_by_split[("sql", "validation")],
        rows_by_split[("sql", "test")],
    ))
    errors.extend(search_builder.validate_release(root, rows_by_split[("search", "test")]))
    errors.extend(research_builder.validate_release(
        root, rows_by_split[("research", "test")],
    ))
    errors.extend(code_builder.validate_release(root, rows_by_split[("code", "test")]))


def validate_release(
    root: Path, *, registry_path: Path, require_private: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    registry = read_json_object(registry_path)
    expected_registry = build_registry(root)
    if registry != expected_registry:
        errors.append("registry differs from canonical files or generated metadata")
    entries = registry.get("splits")
    if not isinstance(entries, list):
        return {"valid": False, "errors": ["registry splits must be a list"], "warnings": []}
    specs = {spec.key: spec for spec in CANONICAL_SPLITS}
    expected_keys = set(specs)
    actual_keys = {(entry.get("domain"), entry.get("split")) for entry in entries}
    if actual_keys != expected_keys or len(entries) != len(expected_keys):
        errors.append("registry split population differs from the canonical splits")
    registry_text = registry_path.read_text(encoding="utf-8")
    if ".local" in registry_text or "private_digest" in registry_text:
        errors.append("registry must describe public artifacts only")
    rows_by_split: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    private_validated = 0
    for entry in entries:
        key = (str(entry.get("domain")), str(entry.get("split")))
        if key not in expected_keys:
            continue
        rows, split_errors = validate_public_split(root, specs[key], entry)
        rows_by_split[key] = rows
        errors.extend(split_errors)
        private_path = root / specs[key].private_path
        if private_path.exists():
            result = validate_dataset(
                root / specs[key].public_path, private_path,
            )
            if not result["valid"]:
                errors.append(f"{key[0]}/{key[1]}: public/private pairing failed")
            else:
                private_validated += 1
        elif require_private:
            errors.append(f"{key[0]}/{key[1]}: required private evaluator is missing")
        else:
            warnings.append(f"{key[0]}/{key[1]}: private evaluator not locally available")
    if set(rows_by_split) == expected_keys:
        _scenario_checks(root, rows_by_split, errors)
    return {
        "schema_version": 1,
        "valid": not errors,
        "registered_splits": len(entries),
        "public_records": sum(len(rows) for rows in rows_by_split.values()),
        "private_splits_validated": private_validated,
        "errors": errors,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--registry", type=Path, default=Path("datasets/registry.json"))
    parser.add_argument("--require-private", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    registry_path = args.registry if args.registry.is_absolute() else root / args.registry
    result = validate_release(
        root, registry_path=registry_path, require_private=args.require_private,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
