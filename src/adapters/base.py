"""
Base class for market-data adapters.

Adapters are responsible for:
  1. Speaking the exchange's wire protocol.
  2. Capturing the original payload (so we can re-normalize on bugfix).
  3. Emitting BookEvent objects via normalize_message().
  4. Tracking sequence numbers to detect gaps.

Adapters MUST NOT:
  - Call any private/authenticated endpoint.
  - Place orders. (See safety.forbid_trading_call.)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterable, Optional

from src.schema import BookEvent


class MarketDataAdapter(ABC):
    """Public market-data adapter contract."""

    name: str = "abstract"
    canonical_symbol: str = ""

    @abstractmethod
    async def connect(self) -> None:
        """Open the websocket and subscribe to the order-book channel(s)."""

    @abstractmethod
    async def fetch_snapshot(self) -> Iterable[BookEvent]:
        """
        Return the initial book as a sequence of `snapshot` events.

        For Coinbase this is a REST level=3 call. For Bitfinex this is the
        first websocket message after subscribe (the raw snapshot).
        """

    @abstractmethod
    async def stream_events(self) -> AsyncIterator[BookEvent]:
        """
        Async-iterate normalized events from the connection. Must yield until
        the underlying connection drops; the caller decides whether to retry.
        """
        # Make this an async generator for type-checkers; subclasses override.
        if False:  # pragma: no cover
            yield

    @abstractmethod
    def normalize_message(self, raw_message: object) -> Iterable[BookEvent]:
        """
        Pure function: raw wire payload -> 0..N BookEvent objects.
        No side effects, no state mutation outside what's needed for sequence
        tracking. Easy to unit test from canned fixtures.
        """

    @abstractmethod
    def validate_sequence(self, message: object) -> object:
        """
        Validate the message's place in the per-exchange sequence stream.

        Return type is exchange-specific:

          - Bitfinex public book has no per-message sequence, so the Bitfinex
            adapter returns None unconditionally. Gaps are detected externally
            via heartbeats (handled by the recorder watchdog).
          - Coinbase returns a SequenceStatus (OK / SKIP / GAP). The Coinbase
            stream-events loop reads that to decide whether to use the message,
            silently drop it, or raise SequenceGapError to force a resync.

        Callers must consult the adapter's documented return type — never
        compare against a generic sentinel here.
        """

    # ---- common utilities exposed to subclasses --------------------------------

    async def close(self) -> None:
        """Default no-op; override if the adapter holds resources."""
        return None
