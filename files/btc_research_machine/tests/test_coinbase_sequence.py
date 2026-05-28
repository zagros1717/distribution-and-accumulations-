"""
Coinbase sequence-handling tests.

We can't easily test the live snapshot-then-replay path without mocking aiohttp
+ websockets, so we focus on the deterministic piece: validate_sequence().
This is the function that flags gaps in the live stream and tells the
recorder to resync.

Coinbase's contract:
  - every message in `full` has a monotonically increasing `sequence`
  - if we see seq != last + 1 (and seq > last), that's a gap -> caller resyncs
  - duplicates or pre-snapshot leftovers (seq <= last) are dropped silently
"""
from __future__ import annotations

from src.adapters.coinbase import CoinbaseFullBookAdapter


def _msg(seq, **kw):
    """Wire-shaped Coinbase message stub."""
    base = {"type": "open", "sequence": seq, "product_id": "BTC-USD"}
    base.update(kw)
    return base


def test_first_message_initializes_last_sequence_no_gap():
    a = CoinbaseFullBookAdapter()
    assert a.validate_sequence(_msg(100)) is None
    assert a._last_sequence == 100


def test_consecutive_sequences_pass_validation():
    a = CoinbaseFullBookAdapter()
    a.validate_sequence(_msg(100))
    for seq in range(101, 110):
        assert a.validate_sequence(_msg(seq)) is None
    assert a._last_sequence == 109


def test_gap_returns_reason_string():
    a = CoinbaseFullBookAdapter()
    a.validate_sequence(_msg(100))
    a.validate_sequence(_msg(101))
    # Now skip 102, jump to 105 — that's a 3-message gap.
    reason = a.validate_sequence(_msg(105))
    assert reason is not None
    assert "gap" in reason.lower()
    assert "102" in reason  # expected the next one
    assert "105" in reason  # but got this
    # After a gap, last_sequence should advance so we don't keep firing on every msg.
    assert a._last_sequence == 105


def test_duplicate_sequence_silently_ignored():
    """seq <= last is a duplicate / pre-snapshot leftover. Return None."""
    a = CoinbaseFullBookAdapter()
    a.validate_sequence(_msg(100))
    a.validate_sequence(_msg(101))
    # Re-feed an earlier one — common during snapshot replay
    assert a.validate_sequence(_msg(100)) is None
    assert a.validate_sequence(_msg(101)) is None
    # last_sequence must not regress
    assert a._last_sequence == 101


def test_heartbeat_without_sequence_does_not_advance_state():
    a = CoinbaseFullBookAdapter()
    a.validate_sequence(_msg(100))
    # Heartbeats lack a sequence in some Coinbase channels.
    assert a.validate_sequence({"type": "heartbeat"}) is None
    assert a._last_sequence == 100


def test_non_dict_messages_are_ignored():
    a = CoinbaseFullBookAdapter()
    assert a.validate_sequence("not a dict") is None
    assert a.validate_sequence(None) is None
    assert a.validate_sequence([1, 2, 3]) is None


# ---------------------------------------------------------------------------
# Snapshot/replay logic, modeled as a pure function on a queue.
#
# The real fetch_snapshot does aiohttp + websocket. We replicate its filter
# rule here — "drop any queued msg with seq <= snapshot_seq, apply the rest" —
# so the contract is locked down with a unit test rather than only living in
# the production code path.
# ---------------------------------------------------------------------------

def _filter_queue_after_snapshot(queue, snapshot_seq):
    """Mirror the rule in CoinbaseFullBookAdapter.stream_events."""
    return [m for m in queue if (m.get("sequence") or 0) > snapshot_seq]


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
