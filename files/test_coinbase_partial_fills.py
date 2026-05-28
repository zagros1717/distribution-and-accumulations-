"""
Coinbase partial-fill tests.

The full-channel `match` message means a public execution against a resting
MAKER order. Two effects:

  1. The trade-flow features see a `trade` event (aggressor_side = taker side).
  2. The order book reduces the maker order's remaining size by `size`. We
     model that as a `match_fill` event so it's distinguishable from a regular
     `modify` (which uses `new_size`).

A `done` message normally follows when the maker is fully consumed, but the
book should not depend on its arrival: if `match_fill` brings the maker to
size <= 0, the order leaves the book immediately. Otherwise it stays at the
reduced size — partial fill.

This file exercises:
  - normalize_message emits both events with correct sides/sizes
  - OrderBook reduces maker size in place on a partial fill
  - OrderBook removes the maker entirely when fully consumed by a match
  - A 'done' arriving after a partial-fill removes the remaining size
  - The taker's `received`/`open` messages do NOT add the taker to the book
    on a marketable order that fully executes
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.adapters.coinbase import CoinbaseFullBookAdapter
from src.book.order_book import OrderBook
from src.schema import BookEvent


@pytest.fixture
def adapter():
    return CoinbaseFullBookAdapter()


def _open_msg(order_id: str, side: str, price: float, size: float, seq: int) -> dict:
    return {
        "type": "open",
        "time": "2026-05-22T12:00:00.000000Z",
        "product_id": "BTC-USD",
        "sequence": seq,
        "order_id": order_id,
        "price": str(price),
        "remaining_size": str(size),
        "side": {"bid": "buy", "ask": "sell"}[side],
    }


def _match_msg(maker_order_id: str, maker_side: str, price: float, size: float,
               trade_id: int, seq: int) -> dict:
    """
    Per Coinbase docs, `side` in a match message is the MAKER's side. The taker
    is on the opposite side. The adapter inverts to populate `aggressor_side`.
    """
    return {
        "type": "match",
        "time": "2026-05-22T12:00:01.000000Z",
        "product_id": "BTC-USD",
        "sequence": seq,
        "trade_id": trade_id,
        "maker_order_id": maker_order_id,
        "taker_order_id": "taker_xyz",
        "side": {"bid": "buy", "ask": "sell"}[maker_side],
        "price": str(price),
        "size": str(size),
    }


def _done_msg(order_id: str, side: str, price: float, remaining: float, seq: int) -> dict:
    return {
        "type": "done",
        "time": "2026-05-22T12:00:02.000000Z",
        "product_id": "BTC-USD",
        "sequence": seq,
        "order_id": order_id,
        "price": str(price),
        "remaining_size": str(remaining),
        "side": {"bid": "buy", "ask": "sell"}[side],
        "reason": "filled",
    }


# ---------------------------------------------------------------------------
# 1. Normalization: match emits both `trade` and `match_fill`
# ---------------------------------------------------------------------------

def test_match_emits_trade_and_match_fill(adapter):
    msg = _match_msg("maker_1", maker_side="ask", price=50_010.0, size=0.5,
                     trade_id=42, seq=200)
    events = list(adapter.normalize_message(msg))
    assert len(events) == 2
    trade, fill = events
    # 1) The public-trade event:
    assert trade.event_type == "trade"
    assert trade.trade_size == 0.5
    assert trade.trade_price == 50_010.0
    # Maker was on the ask side, so the taker (the aggressor) was on the bid.
    assert trade.aggressor_side == "bid"
    # 2) The book-mutation event:
    assert fill.event_type == "match_fill"
    assert fill.order_id == "maker_1"
    assert fill.side == "ask"
    assert fill.size == 0.5
    assert fill.price == 50_010.0


def test_match_with_seller_maker_marks_buyer_aggressor(adapter):
    """Maker side 'bid' (buy) means the taker sold; aggressor is 'ask'."""
    msg = _match_msg("maker_2", maker_side="bid", price=50_000.0, size=0.25,
                     trade_id=43, seq=201)
    trade, fill = list(adapter.normalize_message(msg))
    assert trade.aggressor_side == "ask"
    assert fill.side == "bid"


def test_received_does_NOT_emit_add(adapter):
    """A 'received' message means the matching engine got the order but it
    may immediately fill; we wait for 'open' before treating it as resting."""
    received = {
        "type": "received",
        "time": "2026-05-22T12:00:00.000000Z",
        "product_id": "BTC-USD",
        "sequence": 100,
        "order_id": "ord_marketable",
        "side": "buy",
        "order_type": "market",
        "funds": "10000.00",
    }
    events = list(adapter.normalize_message(received))
    assert events == [], "received must NOT produce a book event"


def test_open_produces_add(adapter):
    events = list(adapter.normalize_message(
        _open_msg("ord_resting", "bid", 50_000.0, 1.0, seq=101)
    ))
    assert len(events) == 1
    assert events[0].event_type == "add"
    assert events[0].order_id == "ord_resting"
    assert events[0].side == "bid"
    assert events[0].size == 1.0
    assert events[0].price == 50_000.0


def test_done_produces_cancel(adapter):
    events = list(adapter.normalize_message(
        _done_msg("ord_resting", "bid", 50_000.0, 0.5, seq=102)
    ))
    assert len(events) == 1
    assert events[0].event_type == "cancel"
    assert events[0].order_id == "ord_resting"


# ---------------------------------------------------------------------------
# 2. OrderBook end-to-end: maker partially filled, then done
# ---------------------------------------------------------------------------

def _apply(book: OrderBook, msgs: list[dict], adapter: CoinbaseFullBookAdapter) -> None:
    """Normalize each wire message and apply every emitted BookEvent."""
    for m in msgs:
        for ev in adapter.normalize_message(m):
            book.apply_event(ev)


def test_partial_fill_reduces_maker_size_in_book(adapter):
    book = OrderBook()
    # Resting ask of size 2.0 at 50_010
    _apply(book, [_open_msg("m1", "ask", 50_010.0, 2.0, seq=10)], adapter)
    assert book.depth("ask", 5) == [(50_010.0, 2.0, 1)]

    # Match consumes 0.5
    _apply(book, [_match_msg("m1", "ask", 50_010.0, 0.5, trade_id=1, seq=11)], adapter)
    assert book.depth("ask", 5) == [(50_010.0, 1.5, 1)], \
        "partial match must reduce maker size by exactly the match size"


def test_match_that_fully_consumes_maker_removes_order(adapter):
    book = OrderBook()
    _apply(book, [_open_msg("m2", "ask", 50_010.0, 1.0, seq=20)], adapter)
    # Match consumes the entire 1.0
    _apply(book, [_match_msg("m2", "ask", 50_010.0, 1.0, trade_id=2, seq=21)], adapter)
    assert book.depth("ask", 5) == []
    assert book.best_ask() is None


def test_done_after_partial_fill_removes_remainder(adapter):
    book = OrderBook()
    _apply(book, [_open_msg("m3", "bid", 50_000.0, 3.0, seq=30)], adapter)
    _apply(book, [_match_msg("m3", "bid", 50_000.0, 1.0, trade_id=3, seq=31)], adapter)
    assert book.depth("bid", 5) == [(50_000.0, 2.0, 1)]

    # Now a 'done' arrives for the remainder (e.g. canceled by the maker)
    _apply(book, [_done_msg("m3", "bid", 50_000.0, 2.0, seq=32)], adapter)
    assert book.depth("bid", 5) == []


def test_match_against_unknown_maker_is_safe_noop(adapter):
    """A match referencing an order we never saw must not corrupt the book."""
    book = OrderBook()
    _apply(book, [_open_msg("known", "bid", 50_000.0, 1.0, seq=40)], adapter)
    _apply(book, [_match_msg("never_saw_this", "ask", 50_010.0, 0.5,
                              trade_id=4, seq=41)], adapter)
    # The book is unchanged and not corrupt.
    assert not book.is_corrupt
    assert book.depth("bid", 5) == [(50_000.0, 1.0, 1)]


def test_marketable_taker_received_does_not_appear_in_book(adapter):
    """A market order that 'received' but never 'open'-ed must NOT show up."""
    book = OrderBook()
    msgs = [
        # Resting maker
        _open_msg("m4", "ask", 50_010.0, 1.0, seq=50),
        # Taker comes in, marketable
        {"type": "received", "time": "2026-05-22T12:00:00.000Z",
         "product_id": "BTC-USD", "sequence": 51, "order_id": "t1",
         "side": "buy", "order_type": "market", "funds": "1000"},
        # Match against the maker
        _match_msg("m4", "ask", 50_010.0, 1.0, trade_id=5, seq=52),
        # done for the maker (filled)
        _done_msg("m4", "ask", 50_010.0, 0.0, seq=53),
        # The taker's done comes too, but we never opened it so it's a no-op:
        _done_msg("t1", "bid", 50_010.0, 0.0, seq=54),
    ]
    _apply(book, msgs, adapter)
    # The book is empty; the taker never rested.
    assert book.depth("ask", 5) == []
    assert book.depth("bid", 5) == []
    assert book.best_bid() is None and book.best_ask() is None
    assert not book.is_corrupt
