"""
Bitfinex trade-snapshot handling tests.

When you subscribe to the `trades` channel, Bitfinex sends a SNAPSHOT first —
a list of recent trades, newest-first per the docs. The naive thing is to
emit them in arrival order; that would put newer trades before older ones
in the normalized stream, which breaks every downstream feature that uses
trade order (last_trade_price, trade_imbalance, vwap).

The adapter has two paths for trade snapshots:

  1. During fetch_snapshot(), trade frames arriving alongside the book
     snapshot are buffered. stream_events() drains the buffer in mts-ascending
     order via _drain_trade_buffer().
  2. After fetch_snapshot(), if Bitfinex re-sends a trade snapshot mid-stream
     (e.g. after an internal hiccup), normalize_message() detects the
     list-of-lists shape on the trade channel and emits the rows in mts order.

These tests exercise both paths.
"""
from __future__ import annotations

import pytest

from src.adapters.bitfinex import BitfinexRawBookAdapter


@pytest.fixture
def adapter():
    a = BitfinexRawBookAdapter()
    a._book_chan_id = 1
    a._trade_chan_id = 2
    return a


def test_trade_snapshot_in_stream_emitted_in_mts_order(adapter):
    """A list-of-lists trades frame arriving via normalize_message must be sorted."""
    # Snapshot rows newest-first per the docs (mts decreasing):
    snapshot = [
        [3000, 1_700_000_002_000, 0.10, 50_005.0],   # newest
        [2000, 1_700_000_001_500, -0.20, 50_004.0],
        [1000, 1_700_000_001_000, 0.05, 50_003.0],   # oldest
    ]
    events = list(adapter.normalize_message([2, snapshot]))
    assert len(events) == 3
    assert all(e.event_type == "trade" for e in events)
    # We expect oldest-first in the emitted stream.
    emitted_mts = [int(e.raw_payload["mts"]) for e in events]
    assert emitted_mts == sorted(emitted_mts), \
        f"trades not sorted ascending: {emitted_mts}"
    # Spot-check: first emitted is the oldest one (mts=1_700_000_001_000)
    assert events[0].trade_id == "1000"
    assert events[-1].trade_id == "3000"


def test_trade_snapshot_via_buffer_drain_is_sorted(adapter):
    """The _drain_trade_buffer path used in fetch_snapshot() must also sort."""
    # Simulate trade frames that arrived during the initial snapshot pump.
    adapter._trade_buffer = [
        [2, [[3001, 1_700_000_002_000, 0.10, 50_005.0],
             [2001, 1_700_000_001_500, -0.20, 50_004.0],
             [1001, 1_700_000_001_000, 0.05, 50_003.0]]],
    ]
    drained = list(adapter._drain_trade_buffer())
    assert [e.trade_id for e in drained] == ["1001", "2001", "3001"]
    # Buffer must be cleared after drain so we don't re-emit.
    assert adapter._trade_buffer == []


def test_trade_snapshot_with_individual_updates_in_buffer(adapter):
    """Buffer may contain a mix of snapshot frames and individual te/tu frames."""
    adapter._trade_buffer = [
        # A snapshot (list-of-lists)
        [2, [[5000, 1_700_000_005_000, 0.5, 50_010.0]]],
        # A solo 'te' that came in just after
        [2, "te", [6000, 1_700_000_006_000, -0.3, 50_011.0]],
        # An older solo 'te' that happened to be buffered too
        [2, "te", [4000, 1_700_000_004_000, 0.1, 50_009.0]],
    ]
    drained = list(adapter._drain_trade_buffer())
    # Sorted by mts (oldest first):
    assert [e.trade_id for e in drained] == ["4000", "5000", "6000"]


def test_trade_snapshot_tu_frames_are_filtered_out_of_buffer(adapter):
    """'tu' updates duplicate 'te' executions and must not double-count."""
    adapter._trade_buffer = [
        [2, "te", [7000, 1_700_000_007_000, 0.4, 50_020.0]],
        [2, "tu", [7000, 1_700_000_007_000, 0.4, 50_020.0]],  # ignored
    ]
    drained = list(adapter._drain_trade_buffer())
    assert len(drained) == 1
    assert drained[0].trade_id == "7000"


def test_trade_snapshot_malformed_rows_are_skipped(adapter):
    """A row with too few fields must be skipped, not crash."""
    snapshot = [
        [1, 1_700_000_000_000, 0.1, 50_000.0],
        [2, 1_700_000_001_000],                   # malformed
        [3, 1_700_000_002_000, 0.2, 50_001.0],
    ]
    events = list(adapter.normalize_message([2, snapshot]))
    assert len(events) == 2
    assert {e.trade_id for e in events} == {"1", "3"}


def test_trade_snapshot_negative_amount_marks_seller_aggressor(adapter):
    """Sign of AMOUNT tells us which side was the taker."""
    snapshot = [
        [10, 1_700_000_000_000, -1.0, 50_000.0],   # seller-aggressed (ask)
        [11, 1_700_000_001_000, 1.0, 50_001.0],    # buyer-aggressed (bid)
    ]
    events = list(adapter.normalize_message([2, snapshot]))
    assert events[0].aggressor_side == "ask"
    assert events[1].aggressor_side == "bid"
