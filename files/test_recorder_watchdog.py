"""
Recorder watchdog tests.

The old watchdog raised asyncio.TimeoutError inside a fire-and-forget task —
Python silently discards exceptions in unawaited tasks, so a truly silent
stream would just hang the recorder forever instead of failing over to the
fallback exchange.

The new watchdog signals via an asyncio.Event that the supervisor's
`asyncio.wait(FIRST_COMPLETED)` is watching. We verify:

  1. With a stream that emits nothing, _run_one() raises StreamSilentError
     within a few wall-clock seconds (not "never").
  2. With a stream that emits events steadily, _run_one() does NOT raise
     for many heartbeat windows.
  3. The shutdown event cleanly terminates _run_one() without an error.

We use a fake MarketDataAdapter so the tests run offline. They are slightly
slower (we exercise real sleeps) but still complete in seconds.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Iterable, Optional

import pytest

from src.adapters.base import MarketDataAdapter
from src.schema import BookEvent
from src.recorder import Recorder, StreamSilentError


class FakeAdapter(MarketDataAdapter):
    """
    Adapter that emits a configurable schedule of events. `interval` is the
    delay (seconds) between yielded events; if None, the stream is silent
    forever after connect+snapshot.
    """
    name = "fake"

    def __init__(self, *, interval: Optional[float] = None,
                 burst: int = 0, canonical_symbol: str = "FAKEUSD"):
        self.canonical_symbol = canonical_symbol
        self.symbol = canonical_symbol
        self.interval = interval
        self.burst = burst
        self.on_raw_frame = None
        self._closed = False

    async def connect(self) -> None:
        return

    async def fetch_snapshot(self) -> Iterable[BookEvent]:
        return []

    async def stream_events(self) -> AsyncIterator[BookEvent]:
        # Emit `burst` events back-to-back, then go silent (or steady).
        for _ in range(self.burst):
            yield self._mk()
        if self.interval is None:
            # Silent forever — sleep on an Event so the test can cancel us cleanly.
            stop = asyncio.Event()
            await stop.wait()
            return
        while not self._closed:
            await asyncio.sleep(self.interval)
            yield self._mk()

    def _mk(self) -> BookEvent:
        now = datetime.now(timezone.utc)
        return BookEvent(
            exchange=self.name, symbol=self.canonical_symbol,
            event_time=now, receive_time=now, sequence=None,
            event_type="heartbeat", raw_payload={},
        )

    def normalize_message(self, raw_message: object):
        return []

    def validate_sequence(self, message: object):
        return None

    async def close(self) -> None:
        self._closed = True


def _make_recorder(tmp_path: Path, heartbeat_timeout: float = 1.0) -> Recorder:
    """A recorder that points storage at a tmp dir and uses a short timeout."""
    cfg = {
        "storage": {
            "root": str(tmp_path / "data"),
            "rotate_files_every_minutes": 60,
        },
        "sources": {
            "primary": {},  # unused (we inject the adapter directly)
            "fallback": {},
            "failover": {
                "max_reconnects_before_switch": 3,
                "reconnect_backoff_seconds": [0.01],
                "heartbeat_timeout_seconds": heartbeat_timeout,
            },
        },
    }
    return Recorder(cfg)


@pytest.mark.asyncio
async def test_watchdog_triggers_on_silent_stream(tmp_path):
    """A stream that emits 0 events after snapshot must raise StreamSilentError."""
    rec = _make_recorder(tmp_path, heartbeat_timeout=0.5)
    adapter = FakeAdapter(interval=None, burst=0)

    # _run_one should raise within ~ heartbeat_timeout * 2 + poll. Bound it to 5s.
    with pytest.raises(StreamSilentError):
        await asyncio.wait_for(rec._run_one(adapter), timeout=5.0)


@pytest.mark.asyncio
async def test_watchdog_does_not_trip_on_active_stream(tmp_path):
    """A stream emitting steadily must NOT trigger the watchdog."""
    rec = _make_recorder(tmp_path, heartbeat_timeout=0.5)
    # One event every 100ms; threshold is 2 * 0.5 = 1s of silence to trip.
    adapter = FakeAdapter(interval=0.1, burst=0)

    async def stop_soon():
        await asyncio.sleep(2.5)
        rec.request_shutdown()

    stop_task = asyncio.create_task(stop_soon())
    try:
        # Should exit cleanly via shutdown, not raise StreamSilentError.
        await asyncio.wait_for(rec._run_one(adapter), timeout=6.0)
    finally:
        stop_task.cancel()
        try: await stop_task
        except asyncio.CancelledError: pass


@pytest.mark.asyncio
async def test_initial_burst_then_silence_still_trips_watchdog(tmp_path):
    """A stream that emits a burst then goes silent must trip the watchdog."""
    rec = _make_recorder(tmp_path, heartbeat_timeout=0.5)
    adapter = FakeAdapter(interval=None, burst=5)  # 5 quick events then silence
    with pytest.raises(StreamSilentError):
        await asyncio.wait_for(rec._run_one(adapter), timeout=5.0)


@pytest.mark.asyncio
async def test_failover_supervisor_switches_to_fallback_after_repeated_silences(tmp_path):
    """
    The supervisor counts primary failures and switches to fallback after
    max_reconnects_before_switch. We patch _make_primary/_make_fallback to
    return Fake adapters and watch which one is used.
    """
    rec = _make_recorder(tmp_path, heartbeat_timeout=0.3)
    rec.failover["max_reconnects_before_switch"] = 2  # switch fast for the test

    used_names: list[str] = []

    def primary_factory():
        a = FakeAdapter(interval=None, burst=0, canonical_symbol="P-FAKE")
        a.name = "primary_fake"
        used_names.append("primary_fake")
        return a

    def fallback_factory():
        # Fallback emits steadily; once we're on it, shutdown will terminate.
        a = FakeAdapter(interval=0.05, burst=0, canonical_symbol="F-FAKE")
        a.name = "fallback_fake"
        used_names.append("fallback_fake")
        return a

    rec._make_primary = primary_factory
    rec._make_fallback = fallback_factory

    async def stop_soon():
        # Wait long enough to see >= 2 primary failures, then ask shutdown.
        await asyncio.sleep(3.5)
        rec.request_shutdown()

    stop_task = asyncio.create_task(stop_soon())
    try:
        await asyncio.wait_for(rec.run(), timeout=10.0)
    finally:
        stop_task.cancel()
        try: await stop_task
        except asyncio.CancelledError: pass

    # We expect: primary_fake at least twice (failures), then fallback_fake.
    assert used_names.count("primary_fake") >= 2, used_names
    assert "fallback_fake" in used_names, used_names
    # And fallback must come AFTER the primary failures, not before.
    first_fallback = used_names.index("fallback_fake")
    primary_before = used_names[:first_fallback].count("primary_fake")
    assert primary_before >= 2, used_names
