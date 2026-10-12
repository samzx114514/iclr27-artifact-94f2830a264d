#!/usr/bin/env python3
"""Download pinned Search assets or inventory an existing local snapshot.

This script never downloads by default: pass ``--download`` explicitly. Large
artifacts remain under the ignored ``.local`` directory and are not vendored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "datasets" / "search" / "retrieval_assets.json"


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("corpus", "index", "embedding_model"):
        item = payload.get(key)
        if not isinstance(item, dict) or not all(item.get(name) for name in ("repository", "repository_type", "revision")):
            raise ValueError(f"invalid {key} asset specification")
    if payload.get("corpus", {}).get("expected_documents") != 100195:
        raise ValueError("Search corpus manifest must freeze 100,195 documents")
    return payload


def download_command(item: dict[str, Any], destination: Path) -> list[str]:
    command = [
        "hf",
        "download",
        str(item["repository"]),
        "--repo-type",
        str(item["repository_type"]),
        "--revision",
        str(item["revision"]),
        "--local-dir",
        str(destination),
    ]
    if item.get("subdirectory"):
        command.extend(["--include", f"{item['subdirectory']}/**"])
    return command


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path, *, hash_files: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        raise FileNotFoundError(f"asset directory does not exist: {root}")
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if ".cache" in path.parts:
            continue
        row: dict[str, Any] = {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size}
        if hash_files:
            row["sha256"] = _sha256(path)
        rows.append(row)
    if not rows:
        raise ValueError(f"asset directory contains no files: {root}")
    return rows


def prepare_assets(
    manifest_path: Path,
    output: Path,
    *,
    download: bool = False,
    hash_files: bool = False,
) -> Path:
    """Download or inventory the pinned assets and write a local receipt."""
    manifest = load_manifest(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "assets": {},
    }
    for key in ("corpus", "index", "embedding_model"):
        target = output / key
        command = download_command(manifest[key], target)
        if download:
            if shutil.which("hf") is None:
                raise SystemExit("missing `hf` CLI; install huggingface_hub first")
            subprocess.run(command, check=True)
        files = inventory(target, hash_files=hash_files)
        receipt["assets"][key] = {
            "repository": manifest[key]["repository"],
            "revision": manifest[key]["revision"],
            "files": files,
            "total_bytes": sum(row["bytes"] for row in files),
        }
    receipt_path = output / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return receipt_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / ".local" / "search-assets")
    parser.add_argument("--download", action="store_true", help="download the pinned large assets")
    parser.add_argument("--hash-files", action="store_true", help="SHA-256 every downloaded file (slow)")
    args = parser.parse_args(argv)

    receipt_path = prepare_assets(
        args.manifest, args.output,
        download=args.download, hash_files=args.hash_files,
    )
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
