"""
Canonical schema for normalized order-book events.

Both Bitfinex (raw books) and Coinbase (full channel) messages are mapped into
this single representation. Every downstream stage — reconstruction, features,
labels — reads from this shape and nothing else. If you add a new exchange,
the only thing you have to write is an adapter that emits BookEvent objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Literal

import pyarrow as pa


# Allowed event types. Anything else from an adapter is a bug.
EVENT_TYPES = (
    "snapshot",    # initial book level (from REST or first WS message)
    "add",         # new resting order
    "modify",      # existing order changed size at the same price
    "cancel",      # order or level removed (full takedown)
    "match_fill",  # a public trade also consumed `size` from order_id (maker order)
    "trade",       # public execution (trade-flow feature signal)
    "heartbeat",   # exchange keepalive — useful for gap detection
    "reset",       # adapter is signaling the book must be wiped (resync coming)
    "unknown",     # unparsed but preserved in raw_payload
)

SIDES = ("bid", "ask")


@dataclass
class BookEvent:
    """A single normalized order-book or trade event."""

    # Provenance
    exchange: str
    symbol: str
    event_time: datetime           # exchange-reported time (UTC)
    receive_time: datetime         # local time we read it off the wire (UTC)
    sequence: Optional[int]        # monotonic sequence if exchange gives one

    # Classification
    event_type: str                # one of EVENT_TYPES

    # Order-book fields (optional — trades don't fill these the same way)
    order_id: Optional[str] = None
    side: Optional[str] = None     # "bid" or "ask"
    price: Optional[float] = None
    size: Optional[float] = None

    # Trade fields
    trade_id: Optional[str] = None
    trade_price: Optional[float] = None
    trade_size: Optional[float] = None
    aggressor_side: Optional[str] = None  # "bid" if buyer aggressed, "ask" if seller

    # Original message kept verbatim so we can re-normalize if a bug is found.
    # Stored as a JSON string in Parquet (Parquet doesn't love arbitrary dicts).
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    # -- helpers --------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"Invalid event_type: {self.event_type!r}")
        if self.side is not None and self.side not in SIDES:
            raise ValueError(f"Invalid side: {self.side!r}")
        if self.aggressor_side is not None and self.aggressor_side not in SIDES:
            raise ValueError(f"Invalid aggressor_side: {self.aggressor_side!r}")
        # Force UTC. Naive datetimes cause off-by-timezone bugs in features.
        if self.event_time.tzinfo is None:
            self.event_time = self.event_time.replace(tzinfo=timezone.utc)
        if self.receive_time.tzinfo is None:
            self.receive_time = self.receive_time.replace(tzinfo=timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Flat dict suitable for Parquet writing. raw_payload becomes JSON string."""
        import json
        d = asdict(self)
        d["raw_payload"] = json.dumps(self.raw_payload, default=str)
        return d


# Arrow schema for normalized events. Keep this in sync with BookEvent.
BOOK_EVENT_SCHEMA = pa.schema(
    [
        pa.field("exchange", pa.string()),
        pa.field("symbol", pa.string()),
        pa.field("event_time", pa.timestamp("us", tz="UTC")),
        pa.field("receive_time", pa.timestamp("us", tz="UTC")),
        pa.field("sequence", pa.int64()),
        pa.field("event_type", pa.string()),
        pa.field("order_id", pa.string()),
        pa.field("side", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("size", pa.float64()),
        pa.field("trade_id", pa.string()),
        pa.field("trade_price", pa.float64()),
        pa.field("trade_size", pa.float64()),
        pa.field("aggressor_side", pa.string()),
        pa.field("raw_payload", pa.string()),
    ]
)


# Schema for the raw WS message log (we keep this BEFORE normalization).
RAW_MESSAGE_SCHEMA = pa.schema(
    [
        pa.field("exchange", pa.string()),
        pa.field("symbol", pa.string()),
        pa.field("receive_time", pa.timestamp("us", tz="UTC")),
        pa.field("channel", pa.string()),
        pa.field("payload", pa.string()),   # raw JSON string
    ]
)
