"""
Counter-discipline tests.

The OrderBook tracks two counters and they MUST NOT be confused:

  corruption_count : times the engine detected damage (crossed book,
                     'unknown' event, exception while applying). High
                     values indicate venue or parser problems and should
                     be investigated.
  resync_count     : times the book was cleanly wiped by an adapter-driven
                     reset (reconnect, fresh REST snapshot). Normal
                     operational behaviour — venues rotate sessions all
                     the time. Not a data-quality signal on its own.

A previous version of reset() incremented corruption_count for clean resets,
which made the daily report scream about "50 corruption periods" when in
fact the recorder had simply reconnected 50 times. These tests pin the
correct discipline.
"""
from __future__ import annotations

from datetime import datetime, timezone

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
# Initial state
# ---------------------------------------------------------------------------

def test_fresh_book_has_zero_counters():
    book = OrderBook()
    assert book.corruption_count == 0
    assert book.resync_count == 0
    assert not book.is_corrupt


# ---------------------------------------------------------------------------
# Clean reset path: increments resync_count only
# ---------------------------------------------------------------------------

def test_clean_reset_increments_resync_count_not_corruption_count():
    """The most common case: adapter issues a fresh snapshot after a
    reconnect. The book wasn't corrupt — it was just being told to wipe."""
    book = OrderBook()
    book.apply_event(_ev("add", order_id="a", side="bid", price=50_000.0, size=1.0))
    book.apply_event(_ev("add", order_id="b", side="ask", price=50_010.0, size=1.0))
    assert book.corruption_count == 0
    assert book.resync_count == 0

    book.apply_event(_ev("reset"))

    assert book.resync_count == 1, "clean reset must increment resync_count"
    assert book.corruption_count == 0, (
        "clean reset must NOT increment corruption_count — the book was healthy"
    )
    assert book.order_count() == 0


def test_repeated_clean_resets_accumulate_resync_count():
    """Each reconnect is one resync."""
    book = OrderBook()
    for _ in range(5):
        book.apply_event(_ev("reset"))
        # Reseed so the next reset is on a clean book again.
        book.apply_event(_ev("snapshot", order_id="x", side="bid",
                              price=50_000.0, size=1.0))
    assert book.resync_count == 5
    assert book.corruption_count == 0


def test_direct_reset_call_also_increments_resync_count():
    """The public reset() method (not just the event) follows the same rule."""
    book = OrderBook()
    book.apply_event(_ev("add", order_id="a", side="bid", price=50_000.0, size=1.0))
    book.reset()
    assert book.resync_count == 1
    assert book.corruption_count == 0


# ---------------------------------------------------------------------------
# Corruption path: increments corruption_count only, even if reset follows
# ---------------------------------------------------------------------------

def test_crossed_book_increments_corruption_count_not_resync_count():
    book = OrderBook()
    book.apply_event(_ev("add", order_id="b", side="bid", price=50_010.0, size=1.0))
    # Ask placed BELOW the bid — crossed book is the canonical corruption signal.
    book.apply_event(_ev("add", order_id="a", side="ask", price=50_000.0, size=1.0))
    assert book.is_corrupt
    assert book.corruption_count == 1
    assert book.resync_count == 0


def test_reset_during_corruption_does_not_double_count():
    """When the book detects damage we mark corrupt (+1 corruption_count).
    The recovery reset that follows must not also bump anything."""
    book = OrderBook()
    book.apply_event(_ev("add", order_id="b", side="bid", price=50_010.0, size=1.0))
    book.apply_event(_ev("add", order_id="a", side="ask", price=50_000.0, size=1.0))
    assert book.corruption_count == 1
    assert book.resync_count == 0
    # Now the reconstructor would call reset() as part of recovery.
    book.reset()
    # corruption_count stays at 1 (already counted); resync_count stays at 0
    # because this reset is recovery, not a clean operational resync.
    assert book.corruption_count == 1
    assert book.resync_count == 0
    assert not book.is_corrupt


def test_unknown_event_increments_corruption_count():
    book = OrderBook()
    book.apply_event(_ev("add", order_id="a", side="bid", price=50_000.0, size=1.0))
    book.apply_event(_ev("unknown", raw_payload={"reason": "test"}))
    assert book.is_corrupt
    assert book.corruption_count == 1
    assert book.resync_count == 0


# ---------------------------------------------------------------------------
# Mixed history: counters track separate concerns
# ---------------------------------------------------------------------------

def test_mixed_history_keeps_counters_independent():
    """Realistic sequence: a few clean reconnects + one corruption + recovery.
    Each counter ends up at the right value."""
    book = OrderBook()

    # Clean reconnect 1
    book.apply_event(_ev("reset"))
    book.apply_event(_ev("snapshot", order_id="x1", side="bid",
                          price=50_000.0, size=1.0))
    # Clean reconnect 2
    book.apply_event(_ev("reset"))
    book.apply_event(_ev("snapshot", order_id="x2", side="bid",
                          price=50_001.0, size=1.0))
    book.apply_event(_ev("add", order_id="y1", side="ask",
                          price=50_011.0, size=1.0))

    # Now venue sends a crossed book — corruption.
    book.apply_event(_ev("add", order_id="bad", side="ask",
                          price=49_500.0, size=1.0))  # below current best bid
    assert book.is_corrupt

    # Recover via reset + snapshot
    book.reset()
    book.apply_event(_ev("snapshot", order_id="x3", side="bid",
                          price=50_005.0, size=1.0))

    # 2 clean resyncs, 1 corruption.
    assert book.resync_count == 2
    assert book.corruption_count == 1


# ---------------------------------------------------------------------------
# Semantic guarantee
# ---------------------------------------------------------------------------

def test_counters_never_overlap_for_a_single_event():
    """For any given reset() call, AT MOST one counter changes (never both,
    never neither when the book was clean)."""
    book = OrderBook()
    book.apply_event(_ev("add", order_id="a", side="bid", price=50_000.0, size=1.0))

    before_corr, before_resync = book.corruption_count, book.resync_count
    book.reset()
    after_corr, after_resync = book.corruption_count, book.resync_count

    changed = [
        ("corruption_count", after_corr - before_corr),
        ("resync_count", after_resync - before_resync),
    ]
    nonzero = [c for c in changed if c[1] != 0]
    assert len(nonzero) == 1, (
        f"exactly one counter should change on a clean reset; got {nonzero}"
    )
    name, delta = nonzero[0]
    assert name == "resync_count"
    assert delta == 1
