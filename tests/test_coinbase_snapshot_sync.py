"""
Coinbase snapshot-sync tests.

End-to-end integration tests for fetch_snapshot() + stream_events() that
exercise the queueing-during-REST behaviour, the post-snapshot filtering,
and the reset emission. We mock the websocket and aiohttp.

What the test fakes verify:

  - While the REST snapshot is in flight, websocket messages keep arriving;
    they must be queued (not dropped, not held in OS buffers).
  - The first event yielded by fetch_snapshot() is a 'reset' event so the
    reconstructor wipes any stale state from a previous session.
  - After the REST returns, queued msgs with sequence <= snapshot_sequence
    are dropped entirely.
  - The first message kept must be snapshot_sequence + 1 — otherwise the
    snapshot was stale and we must resync.
  - Live-phase duplicates (seq <= last_sequence) are dropped silently
    without being yielded.
  - A live-phase gap raises SequenceGapError, not a silent return.

Note: we never call adapter.connect() because that opens a real websocket.
Tests inject a FakeWebSocket directly into adapter._ws.
"""
from __future__ import annotations

import asyncio
import json
from typing import List
from unittest.mock import patch

import pytest

from src.adapters.coinbase import (
    CoinbaseFullBookAdapter, SequenceGapError,
)
from src.schema import BookEvent


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeWebSocket:
    """
    Async-iterable websocket double.

    `feed(payload)` pushes a frame the next recv()/async-iter call will return.
    The async-for protocol used in stream_events() is implemented via
    __aiter__/__anext__ delegating to a queue.
    """
    def __init__(self) -> None:
        self._pending: asyncio.Queue[str] = asyncio.Queue()
        self._closed = False

    async def feed(self, payload) -> None:
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload)
        await self._pending.put(payload)

    async def recv(self) -> str:
        if self._closed and self._pending.empty():
            raise StopAsyncIteration
        return await self._pending.get()

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if self._closed and self._pending.empty():
            raise StopAsyncIteration
        return await self._pending.get()

    async def close(self) -> None:
        self._closed = True


class FakeAiohttpResponse:
    """aiohttp response double that returns canned JSON snapshot data."""
    def __init__(self, data: dict, delay: float = 0.0) -> None:
        self._data = data
        self._delay = delay

    async def __aenter__(self):
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        return self

    async def __aexit__(self, *_):
        return False

    def raise_for_status(self) -> None:
        return

    async def text(self) -> str:
        return json.dumps(self._data)


class FakeAiohttpSession:
    """aiohttp.ClientSession double."""
    def __init__(self, snapshot_data: dict, delay: float = 0.0) -> None:
        self._data = snapshot_data
        self._delay = delay

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    def get(self, url, **_kw):
        return FakeAiohttpResponse(self._data, self._delay)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_frame(seq: int, order_id: str, side: str, price: float, size: float) -> dict:
    return {
        "type": "open",
        "sequence": seq,
        "order_id": order_id,
        "side": {"bid": "buy", "ask": "sell"}[side],
        "price": str(price),
        "remaining_size": str(size),
        "time": "2026-05-22T12:00:00.000Z",
        "product_id": "BTC-USD",
    }


def _snapshot_data(seq: int) -> dict:
    return {
        "sequence": seq,
        "bids": [["50000.00", "1.0", "snap_bid_1"], ["49999.00", "0.5", "snap_bid_2"]],
        "asks": [["50010.00", "1.0", "snap_ask_1"]],
    }


def _make_adapter_with_fake_ws() -> tuple[CoinbaseFullBookAdapter, FakeWebSocket]:
    """Build an adapter and wire a FakeWebSocket into it directly.
    Bypasses connect() so no real network is touched."""
    adapter = CoinbaseFullBookAdapter()
    ws = FakeWebSocket()
    adapter._ws = ws
    return adapter, ws


# ---------------------------------------------------------------------------
# 1. WS messages buffered DURING the REST call are not lost
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_websocket_frames_during_rest_snapshot_are_queued():
    """
    While fetch_snapshot() is in the middle of its REST GET (we give it a
    delay), the background drainer must continue reading websocket frames
    into the queue. Otherwise frames that arrive in that window are lost.
    """
    adapter, ws = _make_adapter_with_fake_ws()

    # Pre-load frames that will be delivered to the drainer while REST is
    # in flight. They MUST end up in adapter._queue.
    during_rest_frames = [
        _open_frame(seq=100, order_id="x1", side="bid", price=50_001.0, size=0.1),
        _open_frame(seq=101, order_id="x2", side="ask", price=50_011.0, size=0.2),
        _open_frame(seq=102, order_id="x3", side="bid", price=50_002.0, size=0.3),
    ]
    for f in during_rest_frames:
        await ws.feed(f)

    # Snapshot at seq=99 (so 100,101,102 are post-snapshot).
    # 0.3s REST delay -> drainer has plenty of time to pull frames.
    session = FakeAiohttpSession(_snapshot_data(seq=99), delay=0.3)
    with patch("src.adapters.coinbase.aiohttp.ClientSession", return_value=session):
        events = list(await adapter.fetch_snapshot())

    # First event must be a reset; remaining are snapshot rows.
    assert events[0].event_type == "reset"
    assert all(e.event_type == "snapshot" for e in events[1:])

    # The three frames must now be in the queue (drainer captured them
    # during the REST call). Give the drainer a brief moment to finish.
    await asyncio.sleep(0.05)
    queued_seqs = sorted(m["sequence"] for m in adapter._queue if isinstance(m, dict))
    assert queued_seqs == [100, 101, 102], (
        f"queued frames lost or extra; got {queued_seqs}"
    )

    await adapter.close()


@pytest.mark.asyncio
async def test_fetch_snapshot_emits_reset_event_first():
    """The very first event from fetch_snapshot() must be a 'reset' event,
    so the reconstructor wipes prior state before applying the new snapshot."""
    adapter, _ws = _make_adapter_with_fake_ws()

    session = FakeAiohttpSession(_snapshot_data(seq=500))
    with patch("src.adapters.coinbase.aiohttp.ClientSession", return_value=session):
        events = list(await adapter.fetch_snapshot())

    assert events, "fetch_snapshot returned no events"
    assert events[0].event_type == "reset"
    # The reset event carries the snapshot sequence so the reconstructor
    # has a hint about where this reset came from.
    assert events[0].sequence == 500
    # Snapshot rows follow.
    snap_rows = [e for e in events[1:] if e.event_type == "snapshot"]
    assert len(snap_rows) == 3   # 2 bids + 1 ask in our fixture
    await adapter.close()


# ---------------------------------------------------------------------------
# 2. Post-snapshot filtering: pre-snapshot leftovers dropped, gap raises
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pre_snapshot_messages_in_queue_are_filtered_in_phase_one():
    """
    Suppose snapshot_seq=100. Queued frames with seq 97, 99, 100 (pre/at)
    must be dropped; only seq 101+ are replayed.
    """
    adapter, ws = _make_adapter_with_fake_ws()

    # Pre-load a mix of pre-, at-, and post-snapshot frames.
    for f in [
        _open_frame(seq=97,  order_id="old1", side="bid", price=49_990.0, size=1.0),
        _open_frame(seq=99,  order_id="old2", side="ask", price=50_020.0, size=1.0),
        _open_frame(seq=100, order_id="at",   side="bid", price=49_995.0, size=1.0),
        _open_frame(seq=101, order_id="new1", side="bid", price=50_001.0, size=1.0),
        _open_frame(seq=102, order_id="new2", side="ask", price=50_009.0, size=1.0),
    ]:
        await ws.feed(f)

    session = FakeAiohttpSession(_snapshot_data(seq=100), delay=0.05)
    with patch("src.adapters.coinbase.aiohttp.ClientSession", return_value=session):
        await adapter.fetch_snapshot()

    # Let the drainer enqueue them.
    await asyncio.sleep(0.05)

    # Phase 1 replay yields only the post-snapshot frames as normalized adds.
    # We bound the iteration since the underlying ws still has nothing else.
    yielded: List[BookEvent] = []

    async def _drain():
        async for ev in adapter.stream_events():
            yielded.append(ev)
            if len(yielded) >= 2:
                return  # we expect exactly two normalized adds

    await asyncio.wait_for(_drain(), timeout=2.0)

    assert len(yielded) == 2
    assert yielded[0].event_type == "add"
    assert yielded[0].order_id == "new1"
    assert yielded[1].order_id == "new2"
    await adapter.close()


@pytest.mark.asyncio
async def test_stale_snapshot_raises_sequence_gap_error_in_phase_one():
    """
    If the first queued msg post-snapshot is NOT snapshot_seq + 1, the
    snapshot itself is stale. Phase 1 must raise SequenceGapError; the
    recorder will reconnect and try again with a fresh snapshot.
    """
    adapter, ws = _make_adapter_with_fake_ws()

    # snapshot_seq=100 but the first queued is 105 — a 4-msg gap.
    for f in [
        _open_frame(seq=105, order_id="g1", side="bid", price=50_000.0, size=1.0),
        _open_frame(seq=106, order_id="g2", side="ask", price=50_010.0, size=1.0),
    ]:
        await ws.feed(f)

    session = FakeAiohttpSession(_snapshot_data(seq=100), delay=0.05)
    with patch("src.adapters.coinbase.aiohttp.ClientSession", return_value=session):
        await adapter.fetch_snapshot()
    await asyncio.sleep(0.05)

    with pytest.raises(SequenceGapError):
        async def _drain():
            async for _ev in adapter.stream_events():
                pass
        await asyncio.wait_for(_drain(), timeout=2.0)

    await adapter.close()


# ---------------------------------------------------------------------------
# 3. Live-phase: duplicates dropped silently; gaps raise
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_duplicates_are_skipped_not_yielded():
    """
    A duplicate (seq <= last_sequence) during live streaming must be silently
    dropped. Pre-fix, it was passed to validate_sequence which returned None
    (interpreted as 'fine') and then normalized + yielded, double-counting.
    """
    adapter, ws = _make_adapter_with_fake_ws()
    # Skip phase-1 entirely by pretending we just finished it.
    adapter._snapshot_sequence = 200
    adapter._last_sequence = 200
    adapter._stop_drain.set()  # drainer not running for this test

    # Feed: 201 (OK), then 199 (duplicate), then 201 again (dup), then 202 (OK).
    await ws.feed(_open_frame(201, "ord201", "bid", 50_000.0, 1.0))
    await ws.feed(_open_frame(199, "ord199", "ask", 50_020.0, 1.0))  # duplicate
    await ws.feed(_open_frame(201, "ord201_dup", "bid", 49_999.0, 1.0))  # dup
    await ws.feed(_open_frame(202, "ord202", "ask", 50_011.0, 1.0))

    yielded: List[BookEvent] = []

    async def _drain():
        async for ev in adapter.stream_events():
            yielded.append(ev)
            if len(yielded) >= 2:
                return

    await asyncio.wait_for(_drain(), timeout=2.0)

    assert len(yielded) == 2
    # Only 201 and 202 should be yielded as adds.
    assert {e.order_id for e in yielded} == {"ord201", "ord202"}
    # The state must reflect that we never accepted the duplicate.
    assert adapter._last_sequence == 202
    await adapter.close()


@pytest.mark.asyncio
async def test_live_sequence_gap_raises_SequenceGapError():
    """A jump in sequence (e.g. 201 -> 205) during live streaming must raise."""
    adapter, ws = _make_adapter_with_fake_ws()
    adapter._snapshot_sequence = 200
    adapter._last_sequence = 200
    adapter._stop_drain.set()

    await ws.feed(_open_frame(201, "a", "bid", 50_000.0, 1.0))
    await ws.feed(_open_frame(205, "b", "bid", 50_000.0, 1.0))  # gap!

    with pytest.raises(SequenceGapError):
        async def _drain():
            async for _ev in adapter.stream_events():
                pass
        await asyncio.wait_for(_drain(), timeout=2.0)

    await adapter.close()
