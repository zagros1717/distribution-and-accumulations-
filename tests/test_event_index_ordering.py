"""
Deterministic replay-ordering tests.

BookEvent rows persisted to parquet carry a monotonic `event_index` that the
NormalizedEventStore stamps at write time. The reconstructor sorts:

  - Coinbase:  by (sequence ASC, event_index ASC)
  - Bitfinex:  by (event_time ASC, event_index ASC)

This file verifies:

  1. BookEvent.event_index defaults to None on construction (adapters do
     not populate it).
  2. NormalizedEventStore stamps event_index monotonically across the
     entire store lifetime — even across multiple write_events() calls
     and across UTC-date partition boundaries.
  3. _sort_events on a Coinbase table puts a 'reset' (which shares the
     snapshot's sequence) AHEAD of the snapshot rows with the same sequence,
     because the reset was written first and therefore has a lower
     event_index.
  4. _sort_events on a Bitfinex table preserves capture order when
     event_time is identical (the typical case for the trade-snapshot
     drain path where many rows are stamped at the same `now()`).
  5. A reset that arrives slightly LATER in event_time than some other
     events but has a HIGHER event_index still ends up sorted after them
     for Bitfinex — i.e. event_time wins as the primary key.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.book.reconstructor import _sort_events
from src.schema import BOOK_EVENT_SCHEMA, BookEvent
from src.storage.parquet_store import (
    NormalizedEventStore, normalized_path, read_parquet_dir,
)


NOW = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)


def _ev(event_type: str, **kw) -> BookEvent:
    defaults = dict(
        exchange="test", symbol="BTC",
        event_time=NOW, receive_time=NOW,
        sequence=None, event_type=event_type,
    )
    defaults.update(kw)
    return BookEvent(**defaults)


# ---------------------------------------------------------------------------
# 1. BookEvent.event_index defaults to None
# ---------------------------------------------------------------------------

def test_book_event_event_index_defaults_to_none():
    """Adapters construct BookEvents; they must not have to compute an index.
    The store stamps the index at write time."""
    ev = _ev("add", order_id="x", side="bid", price=50_000.0, size=1.0)
    assert ev.event_index is None


def test_book_event_to_dict_carries_event_index_key():
    ev = _ev("add", order_id="x", side="bid", price=50_000.0, size=1.0)
    d = ev.to_dict()
    assert "event_index" in d
    assert d["event_index"] is None


def test_arrow_schema_includes_event_index():
    """If you remove event_index from BOOK_EVENT_SCHEMA, replay sort fails
    silently. Lock it in."""
    names = [f.name for f in BOOK_EVENT_SCHEMA]
    assert "event_index" in names
    field = BOOK_EVENT_SCHEMA.field("event_index")
    assert field.type == pa.int64()


# ---------------------------------------------------------------------------
# 2. NormalizedEventStore stamps monotonic indices
# ---------------------------------------------------------------------------

def test_store_stamps_event_index_monotonically_in_one_batch(tmp_path: Path):
    store = NormalizedEventStore(tmp_path, "bitfinex", "BTCUSD")
    evs = [_ev("add", order_id=str(i), side="bid", price=50_000+i, size=1.0,
               exchange="bitfinex", symbol="BTCUSD")
           for i in range(5)]
    store.write_events(evs)
    store.close()

    t = read_parquet_dir(normalized_path(tmp_path, "bitfinex", "BTCUSD", NOW))
    assert t.column("event_index").to_pylist() == [0, 1, 2, 3, 4]


def test_store_event_index_continues_across_batches(tmp_path: Path):
    """Two write_events() calls -> 0,1,2 then 3,4 (the counter is per-store)."""
    store = NormalizedEventStore(tmp_path, "bitfinex", "BTCUSD")
    batch1 = [_ev("add", order_id=str(i), side="bid", price=50_000+i, size=1.0,
                  exchange="bitfinex", symbol="BTCUSD")
              for i in range(3)]
    batch2 = [_ev("add", order_id=str(100+i), side="ask", price=51_000+i, size=1.0,
                  exchange="bitfinex", symbol="BTCUSD")
              for i in range(2)]
    store.write_events(batch1)
    store.write_events(batch2)
    store.close()

    t = read_parquet_dir(normalized_path(tmp_path, "bitfinex", "BTCUSD", NOW))
    indices = t.column("event_index").to_pylist()
    assert indices == [0, 1, 2, 3, 4]


def test_store_event_index_continues_across_utc_date_boundary(tmp_path: Path):
    """The counter is global to the store, not per-partition. Events that
    fall in different UTC days still get strictly-increasing indices."""
    store = NormalizedEventStore(tmp_path, "bitfinex", "BTCUSD")
    # Today's events
    today = [_ev("add", order_id=str(i), side="bid", price=50_000, size=1.0,
                 exchange="bitfinex", symbol="BTCUSD",
                 event_time=datetime(2026, 5, 22, 23, 59, 59, tzinfo=timezone.utc))
             for i in range(2)]
    # Tomorrow's events (cross UTC midnight)
    tomorrow = [_ev("add", order_id=str(100+i), side="ask", price=51_000, size=1.0,
                    exchange="bitfinex", symbol="BTCUSD",
                    event_time=datetime(2026, 5, 23, 0, 0, 0, tzinfo=timezone.utc))
                for i in range(3)]
    store.write_events(today)
    store.write_events(tomorrow)
    store.close()

    # Read both partitions and merge by event_index.
    t_today = read_parquet_dir(normalized_path(
        tmp_path, "bitfinex", "BTCUSD",
        datetime(2026, 5, 22, tzinfo=timezone.utc)))
    t_tomorrow = read_parquet_dir(normalized_path(
        tmp_path, "bitfinex", "BTCUSD",
        datetime(2026, 5, 23, tzinfo=timezone.utc)))

    assert t_today.column("event_index").to_pylist() == [0, 1]
    assert t_tomorrow.column("event_index").to_pylist() == [2, 3, 4]


# ---------------------------------------------------------------------------
# 3. _sort_events: Coinbase reset replays before snapshot rows
# ---------------------------------------------------------------------------

def _build_coinbase_table(rows):
    """rows = list of dicts with at least sequence/event_index/event_time/event_type."""
    ts = pa.array([NOW] * len(rows), type=pa.timestamp("us", tz="UTC"))
    return pa.table({
        "exchange": [r.get("exchange", "coinbase") for r in rows],
        "sequence": [r["sequence"] for r in rows],
        "event_index": [r["event_index"] for r in rows],
        "event_time": ts,
        "event_type": [r["event_type"] for r in rows],
        "order_id": [r.get("order_id") for r in rows],
    })


def test_sort_events_coinbase_reset_replays_before_snapshot_rows():
    """The reset BookEvent and the snapshot rows that follow it all share the
    snapshot's sequence number. The reset is emitted first by the adapter so
    it gets a smaller event_index — _sort_events must put it first."""
    rows = [
        # In arrival order: reset (idx=10) first, then 3 snapshot rows
        # (idx=11,12,13), then a live add at the next sequence (idx=14).
        {"sequence": 500, "event_index": 11, "event_type": "snapshot", "order_id": "a"},
        {"sequence": 500, "event_index": 12, "event_type": "snapshot", "order_id": "b"},
        {"sequence": 500, "event_index": 10, "event_type": "reset"},
        {"sequence": 501, "event_index": 14, "event_type": "add", "order_id": "c"},
        {"sequence": 500, "event_index": 13, "event_type": "snapshot", "order_id": "d"},
    ]
    sorted_t = _sort_events(_build_coinbase_table(rows), "coinbase")
    types = sorted_t.column("event_type").to_pylist()
    assert types == ["reset", "snapshot", "snapshot", "snapshot", "add"]
    # Indices in their actual sort order:
    indices = sorted_t.column("event_index").to_pylist()
    assert indices == [10, 11, 12, 13, 14], (
        f"reset (idx 10) must appear before its snapshot rows (11,12,13); got {indices}"
    )


def test_sort_events_coinbase_unrelated_sequences_use_sequence_order():
    """Sequences across snapshot boundaries — event_index is the secondary
    key but sequence still drives the primary ordering."""
    rows = [
        {"sequence": 100, "event_index": 5, "event_type": "add", "order_id": "a"},
        {"sequence": 99,  "event_index": 6, "event_type": "add", "order_id": "b"},
        {"sequence": 101, "event_index": 4, "event_type": "add", "order_id": "c"},
    ]
    sorted_t = _sort_events(_build_coinbase_table(rows), "coinbase")
    seqs = sorted_t.column("sequence").to_pylist()
    assert seqs == [99, 100, 101]


# ---------------------------------------------------------------------------
# 4. _sort_events: Bitfinex preserves capture order at identical event_time
# ---------------------------------------------------------------------------

def _build_bitfinex_table(rows):
    return pa.table({
        "exchange": ["bitfinex"] * len(rows),
        "sequence": [None] * len(rows),
        "event_index": [r["event_index"] for r in rows],
        "event_time": pa.array(
            [r["event_time"] for r in rows], type=pa.timestamp("us", tz="UTC")
        ),
        "event_type": [r["event_type"] for r in rows],
    })


def test_sort_events_bitfinex_breaks_event_time_ties_with_event_index():
    """Many Bitfinex book/trade events share the same `utcnow()` stamp. The
    event_index tiebreaker is the only way to recover capture order."""
    rows = [
        {"event_time": NOW, "event_index": 3, "event_type": "add"},
        {"event_time": NOW, "event_index": 1, "event_type": "reset"},
        {"event_time": NOW, "event_index": 2, "event_type": "snapshot"},
    ]
    sorted_t = _sort_events(_build_bitfinex_table(rows), "bitfinex")
    types = sorted_t.column("event_type").to_pylist()
    indices = sorted_t.column("event_index").to_pylist()
    assert indices == [1, 2, 3]
    assert types == ["reset", "snapshot", "add"]


def test_sort_events_bitfinex_event_time_wins_over_event_index():
    """If event_time differs, event_time is the primary key — even if a
    later-captured event has a larger event_index, an earlier-stamped event
    sorts first."""
    earlier = NOW
    later = NOW + timedelta(seconds=1)
    rows = [
        # Earlier event captured later (larger index) — still sorts first.
        {"event_time": earlier, "event_index": 10, "event_type": "add"},
        {"event_time": later,   "event_index": 5,  "event_type": "add"},
    ]
    sorted_t = _sort_events(_build_bitfinex_table(rows), "bitfinex")
    indices = sorted_t.column("event_index").to_pylist()
    assert indices == [10, 5]


# ---------------------------------------------------------------------------
# 5. Backwards-compatible: legacy parquet without event_index still sorts
# ---------------------------------------------------------------------------

def test_sort_events_handles_table_without_event_index_column():
    """A parquet file written before event_index existed still has to be
    readable. _sort_events falls back to event_time as the only key."""
    t = pa.table({
        "exchange": ["bitfinex", "bitfinex"],
        "sequence": [None, None],
        "event_time": pa.array(
            [NOW + timedelta(seconds=1), NOW],
            type=pa.timestamp("us", tz="UTC"),
        ),
        "event_type": ["add", "snapshot"],
    })
    # No event_index column — must not raise.
    sorted_t = _sort_events(t, "bitfinex")
    assert sorted_t.column("event_type").to_pylist() == ["snapshot", "add"]
