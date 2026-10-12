from __future__ import annotations

import json
from pathlib import Path

from mtagentbench.release import FROZEN_SPLITS, build_registry
from mtagentbench.release_validation import validate_release


ROOT = Path(__file__).resolve().parents[1]


def test_registry_is_exactly_reproducible_from_canonical_public_splits() -> None:
    committed = json.loads((ROOT / "datasets/registry.json").read_text(encoding="utf-8"))
    assert committed == build_registry(ROOT)
    assert len(committed["splits"]) == len(FROZEN_SPLITS) == 9
    serialized = json.dumps(committed, sort_keys=True)
    assert ".local" not in serialized
    assert "private_digest" not in serialized
    assert "source_digest" not in serialized
    assert "manifest" not in serialized


def test_release_gate_passes_all_committed_splits() -> None:
    result = validate_release(
        ROOT, registry_path=ROOT / "datasets/registry.json", require_private=False,
    )
    assert result["valid"], result["errors"]
    assert result["registered_splits"] == 9
    assert result["public_records"] == 900


def test_release_gate_rejects_registry_digest_drift(tmp_path: Path) -> None:
    registry = build_registry(ROOT)
    registry["splits"][0]["public_sha256"] = "0" * 64
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    result = validate_release(ROOT, registry_path=path, require_private=False)
    assert result["valid"] is False
    assert "registry differs" in result["errors"][0]
