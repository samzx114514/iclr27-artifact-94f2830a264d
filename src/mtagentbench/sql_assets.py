"""Deterministic public schema extraction from external BIRD SQLite assets."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any, Mapping


def find_database(root: Path, db_id: str) -> Path:
    candidates = (
        root / db_id / f"{db_id}.sqlite",
        root / f"{db_id}.sqlite",
        root / f"{db_id}.db",
    )
    try:
        return next(path for path in candidates if path.is_file())
    except StopIteration as error:
        raise FileNotFoundError(
            f"BIRD database not found for db_id={db_id} under {root}"
        ) from error


def schema_text(db_path: Path) -> str:
    """Match the public BIRD prompt schema format without model calls."""
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        parts: list[str] = []
        for (table,) in cursor.fetchall():
            cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
            value = cursor.fetchone()
            create = value[0] if value and value[0] else f"-- No CREATE statement for {table}"
            parts.append(f"{create};")
            try:
                quoted_table = str(table).replace('"', '""')
                cursor.execute(f'SELECT * FROM "{quoted_table}" LIMIT 3')
                sample_rows = cursor.fetchall()
                columns = [item[0] for item in cursor.description or ()]
                if sample_rows:
                    parts.append(f"-- Sample rows from {table}:")
                    parts.append(f"-- Columns: {', '.join(columns)}")
                    parts.extend(f"--   {row}" for row in sample_rows)
            except sqlite3.Error:
                pass
            parts.append("")
        schema = "\n".join(parts).strip()
    finally:
        connection.close()
    if not schema:
        raise RuntimeError(f"no public schema could be extracted from {db_path}")
    return schema


def enrich_public_schemas(
    rows: list[Mapping[str, Any]], db_root: Path | None
) -> list[dict[str, Any]]:
    """Fill only absent public schemas while leaving evaluator fields untouched."""
    missing = [
        row
        for row in rows
        if not str((row.get("metadata") or {}).get("schema") or row.get("schema") or "").strip()
    ]
    if missing and db_root is None:
        raise RuntimeError(
            f"the SQL source has {len(missing)} rows without embedded schemas; pass --db-root "
            "pointing to the official BIRD dev_databases directory"
        )
    schemas: dict[str, str] = {}
    enriched: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        metadata = dict(row.get("metadata") or {})
        if not str(metadata.get("schema") or row.get("schema") or "").strip():
            db_id = metadata.get("db_id") or row.get("db_id")
            if not isinstance(db_id, str) or not db_id:
                raise RuntimeError(f"{row.get('task_id')}: SQL row has no db_id")
            if db_id not in schemas:
                assert db_root is not None
                schemas[db_id] = schema_text(find_database(db_root, db_id))
            metadata["schema"] = schemas[db_id]
        copied["metadata"] = metadata
        enriched.append(copied)
    return enriched
