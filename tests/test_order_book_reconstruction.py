"""
Order-book engine tests.

Synthetic event sequences drive the L3 book and we assert on best/mid/depth
and on the corruption detector.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.book.order_book import OrderBook
from src.schema import BookEvent


NOW = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)


def _ev(event_type: str, **kw) -> BookEvent:
    """Build a BookEvent with sane defaults so each test focuses on what matters."""
    defaults = dict(
        exchange="test",
        symbol="BTCUSD",
        event_time=NOW,
        receive_time=NOW,
        sequence=None,
        event_type=event_type,
    )
    defaults.update(kw)
    return BookEvent(**defaults)


def test_empty_book_has_no_best_bid_or_ask():
    book = OrderBook()
    assert book.best_bid() is None
    assert book.best_ask() is None
    assert book.mid_price() is None
    assert book.spread() is None


def test_add_orders_produce_correct_best_and_mid():
    book = OrderBook()
    book.apply_event(_ev("add", order_id="b1", side="bid", price=50_000.0, size=1.0))
    book.apply_event(_ev("add", order_id="a1", side="ask", price=50_010.0, size=2.0))
    assert book.best_bid() == 50_000.0
    assert book.best_ask() == 50_010.0
    assert book.mid_price() == 50_005.0
    assert book.spread() == 10.0
    assert not book.is_corrupt


def test_better_price_becomes_new_best():
    book = OrderBook()
    book.apply_event(_ev("add", order_id="b1", side="bid", price=50_000.0, size=1.0))
    book.apply_event(_ev("add", order_id="b2", side="bid", price=50_005.0, size=1.0))  # better
    book.apply_event(_ev("add", order_id="a1", side="ask", price=50_020.0, size=1.0))
    book.apply_event(_ev("add", order_id="a2", side="ask", price=50_015.0, size=1.0))  # better
    assert book.best_bid() == 50_005.0
    assert book.best_ask() == 50_015.0


def test_cancel_removes_order_and_promotes_next_level():
    book = OrderBook()
    book.apply_event(_ev("add", order_id="b1", side="bid", price=50_000.0, size=1.0))
    book.apply_event(_ev("add", order_id="b2", side="bid", price=50_005.0, size=1.0))
    book.apply_event(_ev("add", order_id="a1", side="ask", price=50_020.0, size=1.0))
    assert book.best_bid() == 50_005.0
    book.apply_event(_ev("cancel", order_id="b2"))
    assert book.best_bid() == 50_000.0


def test_modify_same_price_changes_only_size():
    book = OrderBook()
    book.apply_event(_ev("add", order_id="b1", side="bid", price=50_000.0, size=1.0))
    book.apply_event(_ev("add", order_id="a1", side="ask", price=50_010.0, size=1.0))
    book.apply_event(_ev("modify", order_id="b1", side="bid", price=50_000.0, size=3.0))
    bid_depth = book.depth("bid", levels=5)
    assert bid_depth[0] == (50_000.0, 3.0, 1)


def test_modify_with_new_price_moves_order():
    book = OrderBook()
    book.apply_event(_ev("add", order_id="b1", side="bid", price=50_000.0, size=1.0))
    book.apply_event(_ev("add", order_id="a1", side="ask", price=50_020.0, size=1.0))
    # Move the bid up to 50_010
    book.apply_event(_ev("modify", order_id="b1", side="bid", price=50_010.0, size=1.0))
    assert book.best_bid() == 50_010.0
    # The 50_000 level should be empty / gone
    bid_depth = book.depth("bid", levels=5)
    assert all(p != 50_000.0 for p, _, _ in bid_depth)


def test_depth_aggregates_sizes_and_order_counts_at_a_level():
    book = OrderBook()
    book.apply_event(_ev("add", order_id="b1", side="bid", price=50_000.0, size=1.0))
    book.apply_event(_ev("add", order_id="b2", side="bid", price=50_000.0, size=2.5))  # same level
    book.apply_event(_ev("add", order_id="b3", side="bid", price=49_995.0, size=0.5))
    depth = book.depth("bid", levels=5)
    # Best level: 50_000 with 1.0 + 2.5 = 3.5 across 2 orders
    assert depth[0] == (50_000.0, 3.5, 2)
    assert depth[1] == (49_995.0, 0.5, 1)


def test_crossed_book_is_detected_and_marks_corrupt():
    book = OrderBook()
    book.apply_event(_ev("add", order_id="b1", side="bid", price=50_010.0, size=1.0))
    # Ask placed BELOW the bid — should cross and trip the health check.
    book.apply_event(_ev("add", order_id="a1", side="ask", price=50_000.0, size=1.0))
    assert book.is_corrupt, "crossed book must mark the engine corrupt"


def test_corrupt_book_drops_subsequent_events():
    book = OrderBook()
    book.apply_event(_ev("add", order_id="b1", side="bid", price=50_010.0, size=1.0))
    book.apply_event(_ev("add", order_id="a1", side="ask", price=50_000.0, size=1.0))
    assert book.is_corrupt
    # Further events must be no-ops — operator has to resync.
    book.apply_event(_ev("add", order_id="b2", side="bid", price=49_900.0, size=10.0))
    # b2 should not be in the book
    depth = book.depth("bid", levels=10)
    assert all(p != 49_900.0 for p, _, _ in depth)


def test_reset_clears_corruption_and_state():
    book = OrderBook()
    book.apply_event(_ev("add", order_id="b1", side="bid", price=50_010.0, size=1.0))
    book.apply_event(_ev("add", order_id="a1", side="ask", price=50_000.0, size=1.0))
    assert book.is_corrupt
    book.reset()
    assert not book.is_corrupt
    assert book.best_bid() is None
    assert book.best_ask() is None


def test_trade_and_heartbeat_events_do_not_mutate_book():
    book = OrderBook()
    book.apply_event(_ev("add", order_id="b1", side="bid", price=50_000.0, size=1.0))
    book.apply_event(_ev("add", order_id="a1", side="ask", price=50_010.0, size=1.0))
    before_orders = book.order_count()
    book.apply_event(_ev("trade", trade_id="t1", trade_price=50_005.0, trade_size=0.1, aggressor_side="bid"))
    book.apply_event(_ev("heartbeat"))
    assert book.order_count() == before_orders
    assert book.best_bid() == 50_000.0
    assert book.best_ask() == 50_010.0


def test_snapshot_returns_book_state_at_timestamp():
    book = OrderBook()
    book.apply_event(_ev("add", order_id="b1", side="bid", price=50_000.0, size=1.0))
    book.apply_event(_ev("add", order_id="a1", side="ask", price=50_010.0, size=1.0))
    snap = book.snapshot(ts_unix_ms=1_700_000_000_000, levels=5)
    assert snap.ts_unix_ms == 1_700_000_000_000
    assert snap.best_bid == 50_000.0
    assert snap.best_ask == 50_010.0
    assert snap.mid_price == 50_005.0
    assert snap.is_valid


def test_zero_or_negative_size_is_rejected_silently():
    """The book shouldn't accept add events with size <= 0; book stays empty."""
    book = OrderBook()
    book.apply_event(_ev("add", order_id="b1", side="bid", price=50_000.0, size=0.0))
    book.apply_event(_ev("add", order_id="b2", side="bid", price=50_000.0, size=-1.0))
    assert book.best_bid() is None
    assert not book.is_corrupt  # silent reject, not a corruption signal


def test_full_sequence_round_trip():
    """End-to-end: add 4 orders, modify 2, cancel 1, check final state."""
    book = OrderBook()
    book.apply_event(_ev("add", order_id="b1", side="bid", price=50_000.0, size=1.0))
    book.apply_event(_ev("add", order_id="b2", side="bid", price=50_005.0, size=2.0))
    book.apply_event(_ev("add", order_id="a1", side="ask", price=50_020.0, size=1.5))
    book.apply_event(_ev("add", order_id="a2", side="ask", price=50_015.0, size=0.5))
    # bump b1 size, drop a1
    book.apply_event(_ev("modify", order_id="b1", side="bid", price=50_000.0, size=5.0))
    book.apply_event(_ev("cancel", order_id="a1"))
    assert book.best_bid() == 50_005.0
    assert book.best_ask() == 50_015.0
    bid_depth = book.depth("bid", levels=10)
    # Two levels: best is 50_005 (2.0, 1 order); next 50_000 (5.0, 1 order)
    assert bid_depth[0] == (50_005.0, 2.0, 1)
    assert bid_depth[1] == (50_000.0, 5.0, 1)
    ask_depth = book.depth("ask", levels=10)
    assert ask_depth == [(50_015.0, 0.5, 1)]  # only one level left
    assert not book.is_corrupt
