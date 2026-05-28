"""
Bitfinex normalize_message() tests.

Bitfinex's raw-book (R0) wire format is positional:
    [CHAN_ID, [ORDER_ID, PRICE, AMOUNT]]                 (single update)
    [CHAN_ID, [[ORDER_ID, PRICE, AMOUNT], ...]]          (initial snapshot)
    [CHAN_ID, "hb"]                                       (heartbeat)
    [CHAN_ID, "te", [ID, MTS, AMOUNT, PRICE]]            (trade — execute)
    [CHAN_ID, "tu", [ID, MTS, AMOUNT, PRICE]]            (trade — update; ignore)

Semantics on R0:
  PRICE  == 0  -> cancel that order id
  AMOUNT >  0  -> bid (buy order)
  AMOUNT <  0  -> ask (sell order); size = |AMOUNT|

The first frame on the book channel is the snapshot (list-of-lists). Subsequent
frames are flat per-order updates.
"""
from __future__ import annotations

import pytest

from src.adapters.bitfinex import BitfinexRawBookAdapter


@pytest.fixture
def adapter():
    """A Bitfinex adapter with channel IDs pre-wired so we skip the WS handshake."""
    a = BitfinexRawBookAdapter()
    a._book_chan_id = 1
    a._trade_chan_id = 2
    return a


# ---------- snapshot --------------------------------------------------------

def test_snapshot_list_of_lists_yields_snapshot_events(adapter):
    snapshot = [
        [1001, 50_000.0, 1.5],     # bid (amount > 0)
        [1002, 50_010.0, -2.0],    # ask (amount < 0) — size 2.0
        [1003, 49_995.0, 0.5],     # bid
    ]
    events = list(adapter.normalize_message([1, snapshot]))
    assert len(events) == 3
    assert all(e.event_type == "snapshot" for e in events)
    # First: bid, price 50_000, size 1.5
    assert events[0].side == "bid"
    assert events[0].price == 50_000.0
    assert events[0].size == 1.5
    assert events[0].order_id == "1001"
    # Second: ask, size taken as |amount|
    assert events[1].side == "ask"
    assert events[1].price == 50_010.0
    assert events[1].size == 2.0
    # Third: bid
    assert events[2].side == "bid"


def test_snapshot_seeds_known_order_ids(adapter):
    snapshot = [[1001, 50_000.0, 1.5], [1002, 50_010.0, -2.0]]
    list(adapter.normalize_message([1, snapshot]))
    # After snapshot, an update with the same id should be a 'modify', not 'add'.
    follow_up = list(adapter.normalize_message([1, [1001, 50_000.0, 3.0]]))
    assert len(follow_up) == 1
    assert follow_up[0].event_type == "modify"


# ---------- flat updates ----------------------------------------------------

def test_new_order_id_emits_add(adapter):
    events = list(adapter.normalize_message([1, [9001, 50_005.0, 1.0]]))
    assert len(events) == 1
    e = events[0]
    assert e.event_type == "add"
    assert e.side == "bid"
    assert e.price == 50_005.0
    assert e.size == 1.0
    assert e.order_id == "9001"


def test_negative_amount_is_ask(adapter):
    events = list(adapter.normalize_message([1, [9002, 50_020.0, -0.75]]))
    assert len(events) == 1
    e = events[0]
    assert e.event_type == "add"
    assert e.side == "ask"
    assert e.size == 0.75  # |amount|


def test_seen_id_emits_modify(adapter):
    # First sighting -> add
    list(adapter.normalize_message([1, [4242, 50_000.0, 1.0]]))
    # Second sighting -> modify
    events = list(adapter.normalize_message([1, [4242, 50_000.0, 2.5]]))
    assert len(events) == 1
    assert events[0].event_type == "modify"
    assert events[0].size == 2.5


def test_price_zero_emits_cancel(adapter):
    """
    On Bitfinex R0 a price=0 update means 'delete order_id'. The wire format
    does NOT carry side or remaining size, so the adapter looks the order up
    in its local map and attaches them. Cancel events therefore carry the
    side and the cancelled size — that's what downstream large-order-cancelled
    features rely on.
    """
    # Add then cancel (we placed a 1.0-BTC bid at 50_000)
    list(adapter.normalize_message([1, [777, 50_000.0, 1.0]]))
    events = list(adapter.normalize_message([1, [777, 0.0, 1.0]]))
    assert len(events) == 1
    e = events[0]
    assert e.event_type == "cancel"
    assert e.side == "bid"
    assert e.price == 50_000.0       # cached price of the cancelled order
    assert e.size == 1.0             # cached size — the amount removed from the book


def test_cancel_for_unknown_order_id_has_no_side_or_size(adapter):
    """A cancel for an order we never saw cannot carry side/size — fields are None."""
    events = list(adapter.normalize_message([1, [9999, 0.0, 1.0]]))
    assert len(events) == 1
    e = events[0]
    assert e.event_type == "cancel"
    assert e.side is None
    assert e.price is None
    assert e.size is None


def test_cancel_carries_cancelled_size_after_modify(adapter):
    """If the order was modified, the cached size is the latest size."""
    list(adapter.normalize_message([1, [1234, 50_000.0, 1.0]]))   # add 1.0
    list(adapter.normalize_message([1, [1234, 50_000.0, 7.5]]))   # modify to 7.5
    events = list(adapter.normalize_message([1, [1234, 0.0, 1.0]]))  # cancel
    assert len(events) == 1
    e = events[0]
    assert e.event_type == "cancel"
    assert e.side == "bid"
    assert e.price == 50_000.0
    assert e.size == 7.5             # final size before cancel, not the original 1.0


def test_cancel_for_ask_carries_ask_side(adapter):
    """The cached side is what the order WAS, not anything from the cancel wire."""
    list(adapter.normalize_message([1, [4242, 50_010.0, -2.5]]))  # ask, size 2.5
    events = list(adapter.normalize_message([1, [4242, 0.0, 1.0]]))
    assert len(events) == 1
    e = events[0]
    assert e.event_type == "cancel"
    assert e.side == "ask"
    assert e.size == 2.5


def test_cancel_removes_id_from_known_set(adapter):
    list(adapter.normalize_message([1, [5555, 50_000.0, 1.0]]))   # add
    list(adapter.normalize_message([1, [5555, 0.0, 1.0]]))        # cancel
    # Re-seeing 5555 after cancel should be a new add, not a modify.
    events = list(adapter.normalize_message([1, [5555, 50_000.0, 2.0]]))
    assert events[0].event_type == "add"


# ---------- heartbeat -------------------------------------------------------

def test_heartbeat_emits_heartbeat_event(adapter):
    events = list(adapter.normalize_message([1, "hb"]))
    assert len(events) == 1
    assert events[0].event_type == "heartbeat"


# ---------- trades ----------------------------------------------------------

def test_trade_te_emits_trade_event(adapter):
    # [CHAN, "te", [ID, MTS, AMOUNT, PRICE]]
    frame = [2, "te", [10001, 1_700_000_000_000, 0.5, 50_007.0]]
    events = list(adapter.normalize_message(frame))
    assert len(events) == 1
    e = events[0]
    assert e.event_type == "trade"
    assert e.trade_id == "10001"
    assert e.trade_price == 50_007.0
    assert e.trade_size == 0.5
    assert e.aggressor_side == "bid"  # amount > 0 -> buyer aggressor


def test_trade_negative_amount_is_seller_aggressor(adapter):
    frame = [2, "te", [10002, 1_700_000_000_000, -0.25, 50_003.0]]
    events = list(adapter.normalize_message(frame))
    assert len(events) == 1
    assert events[0].aggressor_side == "ask"
    assert events[0].trade_size == 0.25


def test_trade_tu_is_suppressed_to_avoid_double_count(adapter):
    """`tu` is the post-trade update; `te` already counted the fill."""
    frame = [2, "tu", [10003, 1_700_000_000_000, 0.25, 50_003.0]]
    events = list(adapter.normalize_message(frame))
    assert events == []


# ---------- robustness ------------------------------------------------------

def test_unknown_channel_id_is_ignored(adapter):
    events = list(adapter.normalize_message([999, [1, 50_000.0, 1.0]]))
    assert events == []


def test_non_list_message_is_ignored(adapter):
    assert list(adapter.normalize_message(None)) == []
    assert list(adapter.normalize_message({"event": "info"})) == []
    assert list(adapter.normalize_message([1])) == []  # too short


def test_malformed_update_row_skipped(adapter):
    """A short row in a snapshot should be skipped, not crash the batch."""
    snapshot = [
        [1001, 50_000.0, 1.0],
        [1002],                 # malformed — skip
        [1003, 50_010.0, -1.0],
    ]
    events = list(adapter.normalize_message([1, snapshot]))
    assert len(events) == 2
    assert events[0].order_id == "1001"
    assert events[1].order_id == "1003"
