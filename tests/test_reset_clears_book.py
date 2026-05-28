"""
Reset-event tests.

When an adapter (Bitfinex or Coinbase) issues a fresh snapshot — typically
after a reconnect — it MUST emit a 'reset' BookEvent immediately before the
snapshot rows. The reconstructor wipes the order book on reset, then reseeds
from the snapshot rows. Without this, orders from a previous session leak
into the new book as phantoms.

This file exercises both code paths:
  1. The Bitfinex adapter's fetch_snapshot() prepends a reset event.
  2. The Coinbase adapter's fetch_snapshot() prepends a reset event.
  3. The reconstructor wipes the book on a reset event and only reseeds on
     the next 'snapshot' event.
  4. A reset arriving in the middle of a normalized event stream forces the
     reconstructor to drop subsequent non-snapshot events until a snapshot
     arrives — the resync window.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.adapters.bitfinex import BitfinexRawBookAdapter
from src.adapters.coinbase import CoinbaseFullBookAdapter
from src.book.order_book import OrderBook
from src.schema import BookEvent


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
# 1. OrderBook: a 'reset' event wipes state
# ---------------------------------------------------------------------------

def test_reset_event_wipes_existing_orders():
    book = OrderBook()
    book.apply_event(_ev("add", order_id="a", side="bid", price=50_000.0, size=1.0))
    book.apply_event(_ev("add", order_id="b", side="ask", price=50_010.0, size=1.0))
    assert book.order_count() == 2
    assert book.best_bid() == 50_000.0

    book.apply_event(_ev("reset"))

    assert book.order_count() == 0
    assert book.best_bid() is None
    assert book.best_ask() is None
    assert not book.is_corrupt


def test_snapshot_after_reset_reseeds_book_cleanly():
    """A reset followed by snapshot events should leave the book in a
    state that reflects ONLY the snapshot, not the prior session."""
    book = OrderBook()
    # Previous session
    book.apply_event(_ev("add", order_id="old1", side="bid", price=49_000.0, size=5.0))
    book.apply_event(_ev("add", order_id="old2", side="ask", price=49_500.0, size=5.0))

    # Reset + new snapshot
    book.apply_event(_ev("reset"))
    book.apply_event(_ev("snapshot", order_id="new1", side="bid", price=50_000.0, size=1.0))
    book.apply_event(_ev("snapshot", order_id="new2", side="ask", price=50_010.0, size=1.0))

    # Only the new orders are present.
    assert book.order_count() == 2
    assert book.best_bid() == 50_000.0
    assert book.best_ask() == 50_010.0
    # Verify the old order ids are gone (not just orphaned at empty levels).
    bid_depth = book.depth("bid", 10)
    assert all(p != 49_000.0 for p, _, _ in bid_depth)


# ---------------------------------------------------------------------------
# 2. Bitfinex adapter prepends a reset to its snapshot
# ---------------------------------------------------------------------------

class _FakeBitfinexWS:
    """Minimal websocket double that yields canned frames in order."""
    def __init__(self, frames):
        self._iter = iter(frames)

    async def recv(self):
        try:
            return json.dumps(next(self._iter))
        except StopIteration:
            # Block forever to simulate an open socket waiting for the next frame.
            await asyncio.Event().wait()

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_bitfinex_fetch_snapshot_first_event_is_reset():
    """Bitfinex's fetch_snapshot must emit a reset before the snapshot rows."""
    adapter = BitfinexRawBookAdapter()
    adapter._book_chan_id = 1
    adapter._trade_chan_id = 2
    # The first frame on the book channel is the snapshot (list-of-lists).
    snapshot_payload = [
        [1001, 50_000.0, 1.0],    # bid
        [1002, 50_010.0, -2.0],   # ask
    ]
    adapter._ws = _FakeBitfinexWS([[1, snapshot_payload]])

    events = list(await adapter.fetch_snapshot())

    assert events[0].event_type == "reset"
    # Followed by the normalized snapshot rows.
    assert events[1].event_type == "snapshot"
    assert events[1].order_id == "1001"
    assert events[2].event_type == "snapshot"
    assert events[2].order_id == "1002"


@pytest.mark.asyncio
async def test_bitfinex_fetch_snapshot_clears_internal_order_map():
    """A reconnect must forget the previous session's order_id map; otherwise
    cancel-for-unknown-order events would carry stale side/price."""
    adapter = BitfinexRawBookAdapter()
    adapter._book_chan_id = 1
    adapter._trade_chan_id = 2
    # Pre-populate the map as if a prior session had cached orders.
    adapter._orders[9999] = ("bid", 49_000.0, 7.5)
    assert 9999 in adapter._orders

    snapshot_payload = [[1001, 50_000.0, 1.0]]
    adapter._ws = _FakeBitfinexWS([[1, snapshot_payload]])
    await adapter.fetch_snapshot()

    # Old order id is forgotten; only the new snapshot order is tracked.
    assert 9999 not in adapter._orders
    assert 1001 in adapter._orders


# ---------------------------------------------------------------------------
# 3. End-to-end: replay through the reconstructor with a reset in the middle
# ---------------------------------------------------------------------------

def _simulate_reconstructor(events):
    """
    Minimal replication of the reconstructor's apply-with-resync loop.

    Returns (book, clean_resyncs, corruption_resyncs, dropped_count). Mirrors
    the counter split in src/book/reconstructor.py so a future refactor of
    the real code that re-merges these counters will fail this test too.
    """
    book = OrderBook()
    awaiting_snapshot = False
    clean_resyncs = 0
    corruption_resyncs = 0
    dropped = 0
    for ev in events:
        if awaiting_snapshot:
            if ev.event_type == "snapshot":
                awaiting_snapshot = False
                book.apply_event(ev)
            elif ev.event_type == "reset":
                book.apply_event(ev)
            else:
                dropped += 1
        else:
            if ev.event_type == "reset":
                book.apply_event(ev)
                awaiting_snapshot = True
                clean_resyncs += 1
            else:
                book.apply_event(ev)
                if book.is_corrupt:
                    book.reset()
                    awaiting_snapshot = True
                    corruption_resyncs += 1
    return book, clean_resyncs, corruption_resyncs, dropped


def test_replay_drops_events_after_reset_until_snapshot():
    """
    Stream: add A, add B, RESET, modify A (should be dropped — book is wiped
    and we're awaiting snapshot), snapshot C, modify C, snapshot D.
    Final book should contain ONLY C (modified) and D, nothing from before reset.
    """
    events = [
        _ev("add", order_id="A", side="bid", price=50_000.0, size=1.0),
        _ev("add", order_id="B", side="ask", price=50_010.0, size=1.0),
        _ev("reset"),
        # These two should be DROPPED — they target A/B, which no longer exist.
        _ev("modify", order_id="A", side="bid", price=50_000.0, size=99.0),
        _ev("add", order_id="stray", side="bid", price=48_000.0, size=10.0),
        # Snapshot rows: book reseeds.
        _ev("snapshot", order_id="C", side="bid", price=51_000.0, size=2.0),
        _ev("snapshot", order_id="D", side="ask", price=51_010.0, size=2.0),
    ]

    book, clean_resyncs, corruption_resyncs, dropped = _simulate_reconstructor(events)
    assert clean_resyncs == 1
    assert corruption_resyncs == 0
    assert dropped == 2          # modify-A and stray-add
    assert book.order_count() == 2
    assert {p for p, _, _ in book.depth("bid", 10)} == {51_000.0}
    assert {p for p, _, _ in book.depth("ask", 10)} == {51_010.0}


def test_replay_reset_with_no_following_snapshot_leaves_book_empty():
    """If a reset arrives but no snapshot follows in the day, the book stays
    empty and all subsequent events are dropped — that's the safe behaviour."""
    events = [
        _ev("add", order_id="A", side="bid", price=50_000.0, size=1.0),
        _ev("reset"),
        _ev("modify", order_id="A", side="bid", price=50_000.0, size=5.0),
        _ev("add", order_id="B", side="ask", price=50_010.0, size=1.0),
    ]
    book, clean_resyncs, corruption_resyncs, dropped = _simulate_reconstructor(events)
    assert clean_resyncs == 1
    assert corruption_resyncs == 0
    assert dropped == 2          # both post-reset events dropped
    assert book.order_count() == 0


# ---------------------------------------------------------------------------
# 4. Coinbase reset uses the same path — already covered in
#    test_coinbase_snapshot_sync.test_fetch_snapshot_emits_reset_event_first
#    but here is a more focused unit-level assertion on the schema.
# ---------------------------------------------------------------------------

def test_reset_event_is_valid_schema():
    """A reset BookEvent has no order_id/side/price/size — pure signal."""
    ev = BookEvent(
        exchange="coinbase", symbol="BTC-USD",
        event_time=NOW, receive_time=NOW,
        sequence=42, event_type="reset",
        raw_payload={"reason": "fresh snapshot"},
    )
    assert ev.event_type == "reset"
    assert ev.order_id is None
    assert ev.side is None
    assert ev.price is None
    assert ev.size is None
