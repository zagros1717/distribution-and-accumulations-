"""
Parquet storage layer.

Layout:

  data/raw/<exchange>/<symbol>/date=YYYY-MM-DD/raw_messages.parquet
  data/normalized/exchange=<x>/symbol=<s>/date=YYYY-MM-DD/book_events.parquet
  data/snapshots/exchange=<x>/symbol=<s>/date=YYYY-MM-DD/interval_ms=<n>/...
  data/features/interval_ms=<n>/exchange=<x>/symbol=<s>/date=YYYY-MM-DD/...
  data/labels/interval_ms=<n>/horizon_s=<h>/exchange=<x>/symbol=<s>/date=YYYY-MM-DD/...

Two writer classes:

  ParquetWriter            : single-directory buffered writer
  DatePartitionedWriter    : multi-day writer that rotates the output
                             directory whenever the *UTC date* of an incoming
                             row changes. This is what the recorder uses so
                             that data crossing midnight UTC ends up in the
                             correct date partition automatically.

Both writers respect `flush_rows` and `flush_seconds`. The recorder turns
`storage.rotate_files_every_minutes` (config) into `flush_seconds=that*60`.
DuckDB and PyArrow are happy reading a directory of parquet parts, so we
do not need a daily compaction step.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from src.schema import BookEvent, BOOK_EVENT_SCHEMA, RAW_MESSAGE_SCHEMA
from src.utils.logging import logger


# --------------------------------------------------------------------------
# Single-directory buffered writer
# --------------------------------------------------------------------------

class ParquetWriter:
    """
    Buffered writer that flushes to a part-file on size or time threshold.

    Each call to .write(records) appends to an in-memory buffer; when buffer
    crosses `flush_rows` or `flush_seconds` elapsed, it rolls to disk and
    starts a new part file.
    """

    def __init__(
        self,
        out_dir: str | Path,
        schema: pa.Schema,
        flush_rows: int = 50_000,
        flush_seconds: int = 60,
        compression: str = "zstd",
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self.flush_rows = flush_rows
        self.flush_seconds = flush_seconds
        self.compression = compression

        self._buf: List[dict] = []
        self._last_flush = datetime.now(timezone.utc)
        self._part_idx = self._next_part_idx()

    def _next_part_idx(self) -> int:
        existing = list(self.out_dir.glob("part-*.parquet"))
        return len(existing)

    def write(self, records: Iterable[dict]) -> None:
        for r in records:
            self._buf.append(r)
        self._maybe_flush()

    def _maybe_flush(self) -> None:
        now = datetime.now(timezone.utc)
        if (
            len(self._buf) >= self.flush_rows
            or (now - self._last_flush).total_seconds() >= self.flush_seconds
        ):
            self.flush()

    def flush(self) -> Optional[Path]:
        if not self._buf:
            return None
        table = pa.Table.from_pylist(self._buf, schema=self.schema)
        out = self.out_dir / f"part-{self._part_idx:05d}.parquet"
        pq.write_table(table, out, compression=self.compression)
        logger.debug(f"[parquet] flushed {len(self._buf)} rows -> {out}")
        self._buf.clear()
        self._part_idx += 1
        self._last_flush = datetime.now(timezone.utc)
        return out

    def close(self) -> None:
        self.flush()


# --------------------------------------------------------------------------
# Date-partitioned writer (for the live recorder)
# --------------------------------------------------------------------------

class DatePartitionedWriter:
    """
    A writer that maintains one ParquetWriter per UTC date.

    Records carry a timestamp field (`time_col`); when the UTC date in that
    field changes from the last seen one, we close the current writer and
    open a new one in the next day's partition.

    `dir_for_date(date) -> Path` builds the output directory for a given UTC
    date — this lets us share the rotation logic between raw/normalized stores.
    """

    def __init__(
        self,
        dir_for_date: Callable[[datetime], Path],
        schema: pa.Schema,
        time_col: str,
        flush_rows: int = 50_000,
        flush_seconds: int = 60,
        compression: str = "zstd",
    ) -> None:
        self.dir_for_date = dir_for_date
        self.schema = schema
        self.time_col = time_col
        self.flush_rows = flush_rows
        self.flush_seconds = flush_seconds
        self.compression = compression
        self._writers: Dict[str, ParquetWriter] = {}

    def _writer_for_date(self, date_iso: str, date: datetime) -> ParquetWriter:
        w = self._writers.get(date_iso)
        if w is None:
            out_dir = self.dir_for_date(date)
            w = ParquetWriter(
                out_dir, self.schema,
                flush_rows=self.flush_rows, flush_seconds=self.flush_seconds,
                compression=self.compression,
            )
            self._writers[date_iso] = w
        return w

    def write(self, records: Iterable[dict]) -> None:
        # Group records by UTC date so each batch goes to the right writer.
        # In the recorder this is typically a batch of 1 row per call so the
        # grouping is trivial — but doing it generically is cleaner than
        # forcing the caller to pre-split.
        for r in records:
            ts = r.get(self.time_col)
            if ts is None:
                # Fall back to "now" so we never lose a row.
                ts = datetime.now(timezone.utc)
            elif not isinstance(ts, datetime):
                # arrow timestamps deserialize as pandas.Timestamp / datetime
                # already, but defensive: try to coerce.
                try:
                    ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
                except Exception:
                    ts = datetime.now(timezone.utc)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)
            date_iso = ts.strftime("%Y-%m-%d")
            self._writer_for_date(date_iso, ts).write([r])

    def flush(self) -> None:
        for w in self._writers.values():
            w.flush()

    def close(self) -> None:
        for w in self._writers.values():
            w.close()
        self._writers.clear()


# --------------------------------------------------------------------------
# Convenience: path builders
# --------------------------------------------------------------------------

def raw_path(root: str | Path, exchange: str, symbol: str, dt: datetime) -> Path:
    return Path(root) / "raw" / exchange / symbol / f"date={dt.strftime('%Y-%m-%d')}"


def normalized_path(root: str | Path, exchange: str, symbol: str, dt: datetime) -> Path:
    return (Path(root) / "normalized" / f"exchange={exchange}"
            / f"symbol={symbol}" / f"date={dt.strftime('%Y-%m-%d')}")


def snapshots_path(root: str | Path, exchange: str, symbol: str, dt: datetime, interval_ms: int) -> Path:
    return (Path(root) / "snapshots" / f"exchange={exchange}"
            / f"symbol={symbol}" / f"date={dt.strftime('%Y-%m-%d')}"
            / f"interval_ms={interval_ms}")


def features_path(root: str | Path, exchange: str, symbol: str, dt: datetime, interval_ms: int) -> Path:
    return (Path(root) / "features"
            / f"interval_ms={interval_ms}"
            / f"exchange={exchange}" / f"symbol={symbol}"
            / f"date={dt.strftime('%Y-%m-%d')}")


def labels_path(root: str | Path, exchange: str, symbol: str, dt: datetime,
                interval_ms: int, horizon_s: int) -> Path:
    """
    Labels are a function of (feature interval, horizon, date, market). The
    interval matters because feature rows at 100ms vs 1000ms are different
    inputs to the same labelling rule; storing them in one directory would
    silently mix datasets.
    """
    return (Path(root) / "labels"
            / f"interval_ms={interval_ms}"
            / f"horizon_s={horizon_s}"
            / f"exchange={exchange}" / f"symbol={symbol}"
            / f"date={dt.strftime('%Y-%m-%d')}")


# --------------------------------------------------------------------------
# High-level writers used by the recorder
# --------------------------------------------------------------------------

class RawMessageStore:
    """
    Append raw wire payloads (as JSON strings) for later re-normalization.

    Rotates output directory by UTC date so a recorder running across midnight
    UTC writes to the correct partitions automatically.
    """

    def __init__(self, root: str | Path, exchange: str, symbol: str,
                 flush_rows: int = 50_000, flush_seconds: int = 60,
                 compression: str = "zstd") -> None:
        self.root = root
        self.exchange = exchange
        self.symbol = symbol
        self.writer = DatePartitionedWriter(
            dir_for_date=lambda dt: raw_path(self.root, self.exchange, self.symbol, dt),
            schema=RAW_MESSAGE_SCHEMA,
            time_col="receive_time",
            flush_rows=flush_rows, flush_seconds=flush_seconds,
            compression=compression,
        )

    def write_message(self, channel: str, payload: object,
                      receive_time: Optional[datetime] = None) -> None:
        """
        Write a single raw frame.

        IMPORTANT: `payload` should be the EXACT websocket text (str) when
        captured pre-normalization. If a dict is passed we json.dumps it as
        a fallback, but that loses the bytes-exact original.
        """
        if isinstance(payload, str):
            payload_str = payload
        else:
            payload_str = json.dumps(payload, default=str)
        self.writer.write([{
            "exchange": self.exchange,
            "symbol": self.symbol,
            "receive_time": receive_time or datetime.now(timezone.utc),
            "channel": channel,
            "payload": payload_str,
        }])

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


class NormalizedEventStore:
    """Append normalized BookEvents. Rotates by UTC date based on event_time."""

    def __init__(self, root: str | Path, exchange: str, symbol: str,
                 flush_rows: int = 50_000, flush_seconds: int = 60,
                 compression: str = "zstd") -> None:
        self.root = root
        self.exchange = exchange
        self.symbol = symbol
        self.writer = DatePartitionedWriter(
            dir_for_date=lambda dt: normalized_path(self.root, self.exchange, self.symbol, dt),
            schema=BOOK_EVENT_SCHEMA,
            time_col="event_time",
            flush_rows=flush_rows, flush_seconds=flush_seconds,
            compression=compression,
        )

    def write_events(self, events: Iterable[BookEvent]) -> None:
        self.writer.write([e.to_dict() for e in events])

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


def read_parquet_dir(path: str | Path) -> pa.Table:
    """
    Read all part files in a partition directory (or empty table).

    We use ParquetFile.read() rather than pq.read_table(path) here because
    the latter auto-detects Hive-style partition fields from the path
    (`interval_ms=...`, `exchange=...`, etc.) and tries to merge them into
    the file schema. When the file already has those columns with a
    different arrow type (e.g. dictionary vs plain string), pyarrow raises
    ArrowTypeError. Reading each file in isolation avoids that conflict.
    """
    p = Path(path)
    if not p.exists():
        return pa.table({})
    files = sorted(p.glob("part-*.parquet"))
    if not files:
        return pa.table({})
    tables = [pq.ParquetFile(str(f)).read() for f in files]
    # Promote conservatively so a "string" column in one file and a
    # dictionary<string> column in another both come out as plain string.
    return pa.concat_tables(tables, promote_options="default")
