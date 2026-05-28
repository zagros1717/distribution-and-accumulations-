"""
Replay engine.

Reads normalized BookEvents (from parquet) in order and feeds them into an
OrderBook instance. Every `interval_ms` (default 1000) we capture a
BookSnapshot and append it to the snapshots store.

Sort policy:
  - Coinbase emits monotonically-increasing `sequence` numbers; the live
    capture path enforces gap detection on those numbers. We sort by
    sequence here so the replay order matches the live order regardless
    of how parquet parts happened to be flushed.
  - Bitfinex has no sequence number on the public raw-book channel, so we
    sort by event_time.

Corruption handling (priority 7):
  - When the book becomes corrupt, we record the period, immediately reset
    the book, and stop applying mutations until we encounter the next
    'snapshot' event in the stream. That 'snapshot' acts as the resync —
    the order book accepts it even from a corrupt state and re-seeds.
  - During the corrupt window, snapshots written to disk carry is_valid=False
    so feature/label/backtest stages can drop them.

This is offline replay. Live data is recorded by `main.record` and replayed
here.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import pyarrow as pa

from src.book.order_book import OrderBook, BookSnapshot
from src.schema import BookEvent
from src.storage.parquet_store import normalized_path, snapshots_path, ParquetWriter, read_parquet_dir
from src.utils.logging import logger


SNAPSHOT_SCHEMA = pa.schema([
    pa.field("ts", pa.timestamp("us", tz="UTC")),
    pa.field("best_bid", pa.float64()),
    pa.field("best_ask", pa.float64()),
    pa.field("mid_price", pa.float64()),
    pa.field("spread", pa.float64()),
    pa.field("bid_size_1", pa.float64()),
    pa.field("ask_size_1", pa.float64()),
    pa.field("bid_size_5", pa.float64()),
    pa.field("ask_size_5", pa.float64()),
    pa.field("bid_size_10", pa.float64()),
    pa.field("ask_size_10", pa.float64()),
    pa.field("bid_orders_1", pa.int64()),
    pa.field("ask_orders_1", pa.int64()),
    pa.field("is_valid", pa.bool_()),
])


def _row_to_event(r: dict) -> BookEvent:
    """Reconstruct a BookEvent from a parquet row dict."""
    raw = r.get("raw_payload")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = {"_raw": raw}
    return BookEvent(
        exchange=r["exchange"],
        symbol=r["symbol"],
        event_time=r["event_time"],
        receive_time=r["receive_time"],
        sequence=r.get("sequence"),
        event_type=r["event_type"],
        order_id=r.get("order_id"),
        side=r.get("side"),
        price=r.get("price"),
        size=r.get("size"),
        trade_id=r.get("trade_id"),
        trade_price=r.get("trade_price"),
        trade_size=r.get("trade_size"),
        aggressor_side=r.get("aggressor_side"),
        raw_payload=raw if isinstance(raw, dict) else {},
    )


def _sort_events(table: pa.Table, exchange: str) -> pa.Table:
    """
    Coinbase: sort by sequence (with event_time as tie-break for snapshot rows
              that all share the snapshot's sequence).
    Bitfinex: sort by event_time (no per-message sequence).
    """
    if exchange == "coinbase":
        # If any sequence is null (e.g. heartbeats), arrow sort_by sends them
        # last; that's fine for our purposes because heartbeats don't mutate
        # the book.
        return table.sort_by([("sequence", "ascending"), ("event_time", "ascending")])
    return table.sort_by("event_time")


def replay_day(
    data_root: str | Path,
    exchange: str,
    symbol: str,
    date: datetime,
    interval_ms: int = 1000,
    max_depth: int = 10,
) -> int:
    """
    Replay one calendar day of normalized events for an (exchange, symbol)
    and write snapshots at `interval_ms` cadence.

    Returns the number of snapshots written.
    """
    in_dir = normalized_path(data_root, exchange, symbol, date)
    out_dir = snapshots_path(data_root, exchange, symbol, date, interval_ms)
    table = read_parquet_dir(in_dir)
    if table.num_rows == 0:
        logger.warning(f"replay: no events at {in_dir}")
        return 0

    table = _sort_events(table, exchange)
    book = OrderBook(max_depth_tracked=50)
    writer = ParquetWriter(out_dir, SNAPSHOT_SCHEMA, flush_rows=10_000, flush_seconds=10)

    next_snapshot_ts_ms: int | None = None
    corrupt_periods: List[Tuple[datetime, datetime]] = []
    corrupt_start: datetime | None = None
    n_snapshots = 0
    n_resyncs = 0
    awaiting_snapshot = False  # we just reset; only 'snapshot' will reseed

    rows = table.to_pylist()
    for r in rows:
        ev = _row_to_event(r)
        # Compute event time in ms since epoch (UTC)
        ts_ms = int(ev.event_time.timestamp() * 1000)
        if next_snapshot_ts_ms is None:
            next_snapshot_ts_ms = ((ts_ms // interval_ms) + 1) * interval_ms

        # Resync path: while waiting for a snapshot, drop every other event.
        if awaiting_snapshot:
            if ev.event_type == "snapshot":
                awaiting_snapshot = False
                book.apply_event(ev)  # this clears corruption
            # else: skip
        else:
            book.apply_event(ev)
            if book.is_corrupt:
                # Record the period, reset, and start waiting for a snapshot.
                if corrupt_start is None:
                    corrupt_start = ev.event_time
                book.reset()
                awaiting_snapshot = True
                n_resyncs += 1

        # If we just exited a corrupt window, close the period record.
        if (not book.is_corrupt) and (not awaiting_snapshot) and corrupt_start is not None:
            corrupt_periods.append((corrupt_start, ev.event_time))
            corrupt_start = None

        # Emit snapshots for every boundary we've now crossed.
        # During an awaiting-snapshot window, snapshots are marked invalid.
        while ts_ms >= next_snapshot_ts_ms:
            snap = book.snapshot(next_snapshot_ts_ms, levels=max_depth)
            is_valid = (not book.is_corrupt) and (not awaiting_snapshot)
            row = _snapshot_to_row(snap, next_snapshot_ts_ms, valid=is_valid)
            writer.write([row])
            n_snapshots += 1
            next_snapshot_ts_ms += interval_ms

    # If we ended the day in a corrupt/awaiting state, close the period.
    if corrupt_start is not None:
        corrupt_periods.append(
            (corrupt_start, rows[-1]["event_time"] if rows else datetime.now(timezone.utc))
        )

    writer.close()
    logger.info(
        f"replay: {exchange}/{symbol} {date.date()} -> {n_snapshots} snapshots, "
        f"{len(corrupt_periods)} corrupt periods, {n_resyncs} resyncs"
    )
    return n_snapshots


def _snapshot_to_row(snap: BookSnapshot, ts_ms: int, valid: bool) -> dict:
    bids = snap.bid_levels
    asks = snap.ask_levels
    bid_sz = lambda n: sum(s for _, s, _ in bids[:n]) if bids else 0.0
    ask_sz = lambda n: sum(s for _, s, _ in asks[:n]) if asks else 0.0
    return {
        "ts": datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc),
        "best_bid": snap.best_bid,
        "best_ask": snap.best_ask,
        "mid_price": snap.mid_price,
        "spread": snap.spread,
        "bid_size_1": bid_sz(1),
        "ask_size_1": ask_sz(1),
        "bid_size_5": bid_sz(5),
        "ask_size_5": ask_sz(5),
        "bid_size_10": bid_sz(10),
        "ask_size_10": ask_sz(10),
        "bid_orders_1": int(bids[0][2]) if bids else 0,
        "ask_orders_1": int(asks[0][2]) if asks else 0,
        "is_valid": valid and snap.is_valid,
    }
