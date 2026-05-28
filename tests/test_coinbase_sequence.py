"""
Coinbase sequence-handling tests.

We can't easily test the live snapshot-then-replay path without mocking aiohttp
+ websockets, so we focus on the deterministic piece: validate_sequence().
This is the function that flags gaps and tells the recorder to resync.

Contract (new, three-valued result):

  validate_sequence(msg) returns a SequenceStatus:
    OK    -> seq == last + 1; advance last_sequence and let the caller use msg
    SKIP  -> seq <= last_sequence; caller MUST drop the msg entirely
             (do not normalize, do not yield, do not apply)
    GAP   -> seq > last_sequence + 1; caller MUST raise SequenceGapError
             so the recorder reconnects (Coinbase sequences are contiguous
             within a session; a gap means data has been lost).
"""
from __future__ import annotations

from src.adapters.coinbase import (
    CoinbaseFullBookAdapter, SequenceStatus, SequenceGapError,
)


def _msg(seq, **kw):
    """Wire-shaped Coinbase message stub."""
    base = {"type": "open", "sequence": seq, "product_id": "BTC-USD"}
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# Single-message validation
# ---------------------------------------------------------------------------

def test_first_message_initializes_last_sequence_status_ok():
    a = CoinbaseFullBookAdapter()
    assert a.validate_sequence(_msg(100)) is SequenceStatus.OK
    assert a._last_sequence == 100


def test_consecutive_sequences_return_ok():
    a = CoinbaseFullBookAdapter()
    a.validate_sequence(_msg(100))
    for seq in range(101, 110):
        assert a.validate_sequence(_msg(seq)) is SequenceStatus.OK
    assert a._last_sequence == 109


def test_gap_returns_gap_status_and_does_not_advance_last_sequence():
    """A gap means the stream is broken. Caller raises; we do NOT advance
    last_sequence so subsequent diagnostics still know what we were expecting."""
    a = CoinbaseFullBookAdapter()
    a.validate_sequence(_msg(100))
    a.validate_sequence(_msg(101))
    # Skip 102, jump to 105 — a three-message gap.
    status = a.validate_sequence(_msg(105))
    assert status is SequenceStatus.GAP
    # last_sequence must NOT advance on a gap; advancing on a gap would mean
    # subsequent messages appear contiguous and the resync never triggers.
    assert a._last_sequence == 101


def test_duplicate_sequence_returns_skip_not_ok():
    """seq <= last_sequence is a duplicate / pre-snapshot leftover.
    The caller MUST drop these messages, not advance state, not yield."""
    a = CoinbaseFullBookAdapter()
    a.validate_sequence(_msg(100))
    a.validate_sequence(_msg(101))
    # Re-feed earlier sequences (e.g. queued from before the snapshot)
    assert a.validate_sequence(_msg(100)) is SequenceStatus.SKIP
    assert a.validate_sequence(_msg(101)) is SequenceStatus.SKIP
    # last_sequence must not regress.
    assert a._last_sequence == 101


def test_heartbeat_without_sequence_returns_ok_and_does_not_advance_state():
    a = CoinbaseFullBookAdapter()
    a.validate_sequence(_msg(100))
    # Some heartbeat shapes have no sequence field.
    assert a.validate_sequence({"type": "heartbeat"}) is SequenceStatus.OK
    assert a._last_sequence == 100


def test_non_dict_messages_are_treated_as_ok_no_op():
    a = CoinbaseFullBookAdapter()
    assert a.validate_sequence("not a dict") is SequenceStatus.OK
    assert a.validate_sequence(None) is SequenceStatus.OK
    assert a.validate_sequence([1, 2, 3]) is SequenceStatus.OK


# ---------------------------------------------------------------------------
# Snapshot/replay logic, modelled as a pure function on a queue.
# Mirrors the rule in stream_events Phase 1.
# ---------------------------------------------------------------------------

def _filter_queue_after_snapshot(queue, snapshot_seq):
    """Mirror the rule in CoinbaseFullBookAdapter.stream_events Phase 1."""
    return [m for m in queue if isinstance(m.get("sequence"), int)
                              and m["sequence"] > snapshot_seq]


def test_snapshot_replay_drops_pre_snapshot_messages():
    # Snapshot taken at seq=100. Queue contains 95, 98, 101, 102 — the first
    # two predate the snapshot and must be dropped; the last two are replayed.
    queue = [_msg(95), _msg(98), _msg(101), _msg(102)]
    kept = _filter_queue_after_snapshot(queue, snapshot_seq=100)
    assert [m["sequence"] for m in kept] == [101, 102]


def test_snapshot_replay_keeps_everything_above_snapshot():
    queue = [_msg(200), _msg(201), _msg(202)]
    kept = _filter_queue_after_snapshot(queue, snapshot_seq=199)
    assert [m["sequence"] for m in kept] == [200, 201, 202]


def test_snapshot_replay_drops_everything_when_snapshot_is_newest():
    queue = [_msg(50), _msg(60), _msg(70)]
    kept = _filter_queue_after_snapshot(queue, snapshot_seq=100)
    assert kept == []
