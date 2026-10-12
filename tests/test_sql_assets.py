from __future__ import annotations

import sqlite3

import pytest

from mtagentbench.sql_assets import enrich_public_schemas, find_database, schema_text


def test_extracts_stable_public_schema_and_samples(tmp_path) -> None:
    root = tmp_path / "databases"
    directory = root / "tiny"
    directory.mkdir(parents=True)
    database = directory / "tiny.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
    connection.executemany("INSERT INTO items VALUES (?, ?)", [(1, "alpha"), (2, "beta")])
    connection.commit()
    connection.close()

    assert find_database(root, "tiny") == database
    rendered = schema_text(database)
    assert "CREATE TABLE items" in rendered
    assert "-- Columns: id, name" in rendered
    assert "(1, 'alpha')" in rendered


def test_missing_database_is_an_infrastructure_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="db_id=missing"):
        find_database(tmp_path, "missing")


def test_enriches_only_missing_schema_without_mutating_source(tmp_path) -> None:
    database = tmp_path / "tiny.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE items (value INTEGER)")
    connection.close()
    source = [{
        "task_id": "sql-1",
        "metadata": {"db_id": "tiny", "gold_sql": "SELECT value FROM items"},
    }]
    enriched = enrich_public_schemas(source, tmp_path)
    assert "CREATE TABLE items" in enriched[0]["metadata"]["schema"]
    assert "schema" not in source[0]["metadata"]


def test_missing_schema_requires_database_root() -> None:
    with pytest.raises(RuntimeError, match="pass --db-root"):
        enrich_public_schemas([{"task_id": "sql-1", "metadata": {"db_id": "tiny"}}], None)
