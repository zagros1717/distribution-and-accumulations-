"""
Recorder + SequenceGapError integration tests.

A sequence gap during the Coinbase live stream MUST raise SequenceGapError
(not yield-an-unknown-then-return-silently like the old design did). The
recorder supervisor catches that exception, counts a session failure, and
either reconnects to the same source or fails over to the fallback after
max_reconnects_before_switch consecutive failures.

These tests use the same FakeAdapter infrastructure as the watchdog tests:
we inject adapters into the recorder via _make_primary/_make_fallback so no
real network is touched.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Iterable, List, Optional

import pytest

from src.adapters.base import MarketDataAdapter
from src.adapters.coinbase import SequenceGapError
from src.recorder import Recorder
from src.schema import BookEvent


# ---------------------------------------------------------------------------
# Fake adapter that raises SequenceGapError on demand
# ---------------------------------------------------------------------------

class GappyAdapter(MarketDataAdapter):
    """
    Adapter that emits N events then raises SequenceGapError. The test uses
    this to simulate Coinbase detecting a missing sequence number mid-stream.
    """
    name = "gappy"

    def __init__(self, *, events_before_gap: int = 2, canonical_symbol: str = "G-FAKE"):
        self.canonical_symbol = canonical_symbol
        self.symbol = canonical_symbol
        self.events_before_gap = events_before_gap
        self.on_raw_frame = None
        self.run_count = 0       # number of times this instance has been streamed
        self._gap_raised = False

    async def connect(self) -> None:
        return

    async def fetch_snapshot(self) -> Iterable[BookEvent]:
        return []

    async def stream_events(self) -> AsyncIterator[BookEvent]:
        self.run_count += 1
        for _ in range(self.events_before_gap):
            now = datetime.now(timezone.utc)
            yield BookEvent(
                exchange=self.name, symbol=self.canonical_symbol,
                event_time=now, receive_time=now, sequence=None,
                event_type="heartbeat", raw_payload={},
            )
        # Simulate Coinbase's sequence-gap path.
        self._gap_raised = True
        raise SequenceGapError("simulated: expected 102 got 105")

    def normalize_message(self, raw_message: object):
        return []

    def validate_sequence(self, message: object):
        return None

    async def close(self) -> None:
        return


class HealthyAdapter(MarketDataAdapter):
    """Adapter that streams heartbeats steadily; used as the fallback to
    verify the supervisor switches to it after primary's gaps."""
    name = "healthy"

    def __init__(self, canonical_symbol: str = "H-FAKE"):
        self.canonical_symbol = canonical_symbol
        self.symbol = canonical_symbol
        self.on_raw_frame = None
        self.run_count = 0

    async def connect(self) -> None:
        return

    async def fetch_snapshot(self) -> Iterable[BookEvent]:
        return []

    async def stream_events(self) -> AsyncIterator[BookEvent]:
        self.run_count += 1
        while True:
            await asyncio.sleep(0.05)
            now = datetime.now(timezone.utc)
            yield BookEvent(
                exchange=self.name, symbol=self.canonical_symbol,
                event_time=now, receive_time=now, sequence=None,
                event_type="heartbeat", raw_payload={},
            )

    def normalize_message(self, raw_message: object):
        return []

    def validate_sequence(self, message: object):
        return None

    async def close(self) -> None:
        return


# ---------------------------------------------------------------------------
# Recorder builder
# ---------------------------------------------------------------------------

def _make_recorder(tmp_path: Path, heartbeat_timeout: float = 5.0) -> Recorder:
    cfg = {
        "storage": {
            "root": str(tmp_path / "data"),
            "rotate_files_every_minutes": 60,
        },
        "sources": {
            "primary": {},
            "fallback": {},
            "failover": {
                "max_reconnects_before_switch": 2,
                "reconnect_backoff_seconds": [0.01],
                "heartbeat_timeout_seconds": heartbeat_timeout,
            },
        },
    }
    return Recorder(cfg)


# ---------------------------------------------------------------------------
# 1. _run_one propagates SequenceGapError so the supervisor sees it
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_one_propagates_sequence_gap_error(tmp_path):
    """If stream_events raises, _run_one must re-raise (not swallow), so the
    supervisor's failure counter increments."""
    rec = _make_recorder(tmp_path)
    adapter = GappyAdapter(events_before_gap=1)

    with pytest.raises(SequenceGapError):
        await asyncio.wait_for(rec._run_one(adapter), timeout=5.0)

    # The adapter did actually run and raise.
    assert adapter._gap_raised


# ---------------------------------------------------------------------------
# 2. Supervisor: a gap counts as a session failure and triggers reconnect.
#    After max_reconnects_before_switch gaps, switch to fallback.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_supervisor_reconnects_and_fails_over_on_repeated_gaps(tmp_path):
    """Two SequenceGapErrors from primary -> supervisor switches to fallback."""
    rec = _make_recorder(tmp_path)
    rec.failover["max_reconnects_before_switch"] = 2

    primary_instances: List[GappyAdapter] = []
    fallback_instances: List[HealthyAdapter] = []

    def primary_factory():
        a = GappyAdapter(events_before_gap=1, canonical_symbol="P-GAP")
        primary_instances.append(a)
        return a

    def fallback_factory():
        a = HealthyAdapter(canonical_symbol="F-HEALTHY")
        fallback_instances.append(a)
        return a

    rec._make_primary = primary_factory
    rec._make_fallback = fallback_factory

    async def stop_when_fallback_active():
        # Wait until at least one HealthyAdapter has begun streaming.
        for _ in range(200):  # up to ~10s
            if fallback_instances and fallback_instances[-1].run_count >= 1:
                await asyncio.sleep(0.2)  # let it run for a moment
                rec.request_shutdown()
                return
            await asyncio.sleep(0.05)
        rec.request_shutdown()

    stopper = asyncio.create_task(stop_when_fallback_active())
    try:
        await asyncio.wait_for(rec.run(), timeout=10.0)
    finally:
        stopper.cancel()
        try: await stopper
        except asyncio.CancelledError: pass

    # Primary should have been retried at least twice (each time raising).
    assert len(primary_instances) >= 2, (
        f"expected >=2 primary attempts before failover; got {len(primary_instances)}"
    )
    for a in primary_instances:
        assert a._gap_raised, "every primary attempt must have raised a gap"
    # Fallback should have been engaged at least once.
    assert len(fallback_instances) >= 1, "fallback never engaged after gaps"
    assert fallback_instances[-1].run_count >= 1


# ---------------------------------------------------------------------------
# 3. SequenceGapError is a real exception that subclasses RuntimeError
#    (so generic 'except Exception' supervisor code catches it)
# ---------------------------------------------------------------------------

def test_sequence_gap_error_is_an_exception():
    """A test that fails if anyone changes SequenceGapError to be e.g. a
    BaseException subclass (which 'except Exception' would NOT catch)."""
    assert issubclass(SequenceGapError, Exception)
    assert issubclass(SequenceGapError, RuntimeError)
    # And it carries a message.
    err = SequenceGapError("gap reason here")
    assert "gap reason here" in str(err)
