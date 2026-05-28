"""
DuckDB query layer.

We don't actually ingest into DuckDB tables — that doubles storage. Instead
we register a set of views that read the parquet partitions in place. Hive
partitioning is auto-detected by DuckDB's `read_parquet(hive_partitioning=true)`.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

from src.utils.logging import logger


def open_duckdb(db_path: str | Path = ":memory:") -> duckdb.DuckDBPyConnection:
    """Open (or create) a DuckDB connection."""
    return duckdb.connect(str(db_path))


def register_views(con: duckdb.DuckDBPyConnection, data_root: str | Path) -> None:
    """
    Create views over the on-disk parquet hierarchy. Call after data has been
    written; safe to call repeatedly.
    """
    data_root = Path(data_root).resolve()
    views = {
        "raw_messages": data_root / "raw" / "**" / "part-*.parquet",
        "book_events": data_root / "normalized" / "**" / "part-*.parquet",
        "snapshots":   data_root / "snapshots" / "**" / "part-*.parquet",
        "features":    data_root / "features"  / "**" / "part-*.parquet",
        "labels":      data_root / "labels"    / "**" / "part-*.parquet",
    }
    for name, glob in views.items():
        try:
            con.execute(
                f"CREATE OR REPLACE VIEW {name} AS "
                f"SELECT * FROM read_parquet('{glob}', hive_partitioning=true)"
            )
            logger.debug(f"[duckdb] view {name} -> {glob}")
        except Exception as e:
            # If no files exist yet, DuckDB raises; create an empty stub view.
            logger.debug(f"[duckdb] view {name} skipped ({e})")
