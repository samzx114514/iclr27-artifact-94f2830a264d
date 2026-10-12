from __future__ import annotations

from pathlib import Path

from mtagentbench import search_assets

ROOT = Path(__file__).parents[1]


def test_search_asset_manifest_is_pinned_to_public_protocol() -> None:
    manifest = search_assets.load_manifest(ROOT / "datasets" / "search" / "retrieval_assets.json")
    assert manifest["corpus"]["expected_documents"] == 100195
    assert len(manifest["corpus"]["revision"]) == 40
    assert len(manifest["index"]["revision"]) == 40
    assert len(manifest["embedding_model"]["revision"]) == 40
    assert manifest["embedding_model"]["dimensions"] == 4096
    assert manifest["retrieval"]["snippet_max_tokens"] == 512


def test_download_commands_pin_revision_and_keep_assets_local(tmp_path: Path) -> None:
    manifest = search_assets.load_manifest(ROOT / "datasets" / "search" / "retrieval_assets.json")
    command = search_assets.download_command(manifest["index"], tmp_path / "index")
    assert command[:3] == ["hf", "download", "Tevatron/browsecomp-plus-indexes"]
    assert command[command.index("--revision") + 1] == manifest["index"]["revision"]
    assert command[-2:] == ["--include", "qwen3-embedding-8b/**"]
    assert not any("123." in value or "API_KEY" in value for value in command)


def test_inventory_can_write_reproducible_file_hashes(tmp_path: Path) -> None:
    (tmp_path / "asset.bin").write_bytes(b"fixed")
    assert search_assets.inventory(tmp_path, hash_files=True) == [
        {
            "path": "asset.bin",
            "bytes": 5,
            "sha256": "992a93455c71fedd36ac9bbc439952c041cf61445958472af479269b8d873513",
        }
    ]
