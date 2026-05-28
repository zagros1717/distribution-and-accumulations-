"""
Coinbase Exchange L3 adapter (full channel).

The 'full' websocket channel emits per-order messages:
  - received: an order has been received by the matching engine. The order may
              be a marketable order that immediately fills; it is NOT necessarily
              resting in the book. We do NOT emit an add on `received`.
  - open:     a maker order is now resting in the book. THIS is the add.
  - match:    a public execution. The MAKER order on the book is being consumed.
              We emit a 'trade' (for trade-flow features) AND, separately, the
              order-book engine reduces the maker_order_id's size by `size`.
              We model that as a 'modify' event with negative size delta; the
              book engine treats size<=0 as cancel.
  - done:     a resting order has been filled or canceled. The remainder (if any)
              is removed from the book. We emit a 'cancel' carrying the order_id;
              the book engine looks up the side and remaining size itself.
  - change:   a maker order resized at the same price (rare).

To maintain a correct L3 book Coinbase's docs prescribe this procedure:

  1. Open the websocket and start QUEUEING all incoming messages.
  2. Fetch a REST level=3 snapshot. It has a `sequence` number.
  3. Drop queued messages with sequence <= snapshot_sequence.
  4. The remaining queued messages MUST form a contiguous sequence starting at
     snapshot_sequence + 1. If they don't, the snapshot is stale — resync.
  5. Apply the remaining queued messages in order.
  6. Continue applying live messages, asserting sequence is contiguous.

If a gap is detected, the only safe thing is to resync: re-fetch the snapshot.

Reference:
  - https://docs.cdp.coinbase.com/exchange/docs/websocket-channels
  - https://docs.cdp.coinbase.com/exchange/docs/rest-api  (GET /products/{id}/book?level=3)

Authenticated channels are NOT used. Per project rules we never send any API key.
"""
from __future__ import annotations

import asyncio
import json
from collections import deque
from typing import AsyncIterator, Deque, Iterable, Optional

import aiohttp
import websockets

from src.adapters.base import MarketDataAdapter
from src.schema import BookEvent
from src.utils.logging import logger
from src.utils.time import parse_iso_utc, utcnow


class CoinbaseFullBookAdapter(MarketDataAdapter):
    """Coinbase Exchange L3 (full channel) adapter."""

    name = "coinbase"

    def __init__(
        self,
        symbol: str = "BTC-USD",
        canonical_symbol: str = "BTC-USD",
        rest_snapshot_url: str = "https://api.exchange.coinbase.com/products/BTC-USD/book?level=3",
        ws_url: str = "wss://ws-feed.exchange.coinbase.com",
        channel: str = "full",
    ) -> None:
        self.symbol = symbol
        self.canonical_symbol = canonical_symbol
        self.rest_snapshot_url = rest_snapshot_url
        self.ws_url = ws_url
        self.channel = channel

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._queue: Deque[dict] = deque()
        self._snapshot_sequence: Optional[int] = None
        self._last_sequence: Optional[int] = None
        self._replay_done = False
        # Public hook for raw-frame capture (set by recorder).
        # Signature: (channel: str, raw_text: str) -> None
        self.on_raw_frame = None

    # ---- connect ----------------------------------------------------------

    async def connect(self) -> None:
        logger.info(f"[coinbase] connecting to {self.ws_url}")
        self._ws = await websockets.connect(self.ws_url, ping_interval=20, max_size=2**22)
        await self._ws.send(json.dumps({
            "type": "subscribe",
            "product_ids": [self.symbol],
            "channels": [self.channel],
        }))
        # Wait for the 'subscriptions' ack; meanwhile queue early frames.
        while True:
            raw_text = await asyncio.wait_for(self._ws.recv(), timeout=30)
            if self.on_raw_frame is not None:
                try:
                    self.on_raw_frame("ws", raw_text)
                except Exception:
                    pass
            msg = json.loads(raw_text)
            if msg.get("type") == "subscriptions":
                logger.info(f"[coinbase] subscribed to {self.channel}")
                break
            if msg.get("type") == "error":
                raise RuntimeError(f"Coinbase subscribe error: {msg}")
            self._queue.append(msg)

    # ---- snapshot ---------------------------------------------------------

    async def fetch_snapshot(self) -> Iterable[BookEvent]:
        """
        REST level=3 snapshot. Returns BookEvent(event_type='snapshot') for
        each resting order. Records the snapshot sequence so the WS replay
        can drop pre-snapshot duplicates and validate that the post-snapshot
        queue is contiguous.
        """
        logger.info(f"[coinbase] GET {self.rest_snapshot_url}")
        async with aiohttp.ClientSession() as session:
            async with session.get(self.rest_snapshot_url, timeout=30) as resp:
                resp.raise_for_status()
                snapshot_text = await resp.text()
                data = json.loads(snapshot_text)
        if self.on_raw_frame is not None:
            try:
                self.on_raw_frame("rest_snapshot", snapshot_text)
            except Exception:
                pass

        self._snapshot_sequence = int(data["sequence"])
        self._last_sequence = self._snapshot_sequence
        now = utcnow()
        events: list[BookEvent] = []
        for price, size, order_id in data.get("bids", []):
            events.append(BookEvent(
                exchange=self.name, symbol=self.canonical_symbol,
                event_time=now, receive_time=now,
                sequence=self._snapshot_sequence,
                event_type="snapshot",
                order_id=str(order_id), side="bid",
                price=float(price), size=float(size),
                raw_payload={"snapshot_row": [price, size, order_id]},
            ))
        for price, size, order_id in data.get("asks", []):
            events.append(BookEvent(
                exchange=self.name, symbol=self.canonical_symbol,
                event_time=now, receive_time=now,
                sequence=self._snapshot_sequence,
                event_type="snapshot",
                order_id=str(order_id), side="ask",
                price=float(price), size=float(size),
                raw_payload={"snapshot_row": [price, size, order_id]},
            ))
        logger.info(
            f"[coinbase] snapshot seq={self._snapshot_sequence} "
            f"bids={len(data.get('bids', []))} asks={len(data.get('asks', []))}"
        )
        return events

    # ---- stream -----------------------------------------------------------

    async def stream_events(self) -> AsyncIterator[BookEvent]:
        """
        Stream events with strict sequence validation.

        Phase 1: drain the pre-snapshot queue.
          - Drop queued msgs with seq <= snapshot_sequence.
          - Sort the remainder by sequence (Coinbase typically sends in order,
            but we don't trust it under reconnect).
          - The first kept seq MUST equal snapshot_sequence + 1. If not -> gap.
          - From there, every queued seq must be contiguous. If any gap -> resync.
        Phase 2: live messages. Same contiguity requirement; on gap, emit
                 an 'unknown' event and return so the supervisor can resync.
        """
        assert self._ws is not None

        # ---- Phase 1: replay queued frames that postdate the snapshot ----
        snap_seq = self._snapshot_sequence
        if snap_seq is None:
            raise RuntimeError("stream_events called before fetch_snapshot")

        queued = [m for m in self._queue if isinstance(m.get("sequence"), int) and m["sequence"] > snap_seq]
        # Sort by sequence for safety.
        queued.sort(key=lambda m: m["sequence"])

        # Reset last_sequence to snapshot baseline so we can validate the queue.
        self._last_sequence = snap_seq

        if queued:
            first_seq = queued[0]["sequence"]
            if first_seq != snap_seq + 1:
                reason = (
                    f"queued-gap-after-snapshot: snapshot_seq={snap_seq} "
                    f"first_queued_seq={first_seq} (expected {snap_seq+1})"
                )
                logger.warning(f"[coinbase] {reason}")
                yield BookEvent(
                    exchange=self.name, symbol=self.canonical_symbol,
                    event_time=utcnow(), receive_time=utcnow(),
                    sequence=first_seq, event_type="unknown",
                    raw_payload={"reason": reason},
                )
                self._queue.clear()
                return  # supervisor will resync

            for msg in queued:
                reason = self.validate_sequence(msg)
                if reason:
                    logger.warning(f"[coinbase] queued sequence issue: {reason}")
                    yield BookEvent(
                        exchange=self.name, symbol=self.canonical_symbol,
                        event_time=utcnow(), receive_time=utcnow(),
                        sequence=msg.get("sequence"), event_type="unknown",
                        raw_payload={"reason": reason, "original": msg},
                    )
                    self._queue.clear()
                    return
                for ev in self.normalize_message(msg):
                    yield ev

        self._queue.clear()
        self._replay_done = True

        # ---- Phase 2: live stream ----
        async for raw_text in self._ws:
            if self.on_raw_frame is not None:
                try:
                    self.on_raw_frame("ws", raw_text)
                except Exception:
                    pass
            msg = json.loads(raw_text)
            reason = self.validate_sequence(msg)
            if reason:
                logger.warning(f"[coinbase] sequence issue: {reason}")
                yield BookEvent(
                    exchange=self.name, symbol=self.canonical_symbol,
                    event_time=utcnow(), receive_time=utcnow(),
                    sequence=msg.get("sequence"), event_type="unknown",
                    raw_payload={"reason": reason, "original": msg},
                )
                return  # force resync
            for ev in self.normalize_message(msg):
                yield ev

    # ---- normalize --------------------------------------------------------

    def normalize_message(self, raw_message: object) -> Iterable[BookEvent]:
        if not isinstance(raw_message, dict):
            return []
        t = raw_message.get("type")
        if t == "heartbeat":
            return [BookEvent(
                exchange=self.name, symbol=self.canonical_symbol,
                event_time=parse_iso_utc(raw_message.get("time", utcnow().isoformat())),
                receive_time=utcnow(),
                sequence=raw_message.get("sequence"),
                event_type="heartbeat",
                raw_payload=raw_message,
            )]

        if t in ("subscriptions", "status", "error"):
            return []

        evt_time = parse_iso_utc(raw_message.get("time", utcnow().isoformat()))
        recv_time = utcnow()
        seq = raw_message.get("sequence")
        side_raw = raw_message.get("side")
        side = {"buy": "bid", "sell": "ask"}.get(side_raw)

        if t == "received":
            # An order has hit the matching engine, but it may immediately fill
            # without ever resting. The 'open' message (if any) is what tells us
            # the order is in the book. We deliberately DO NOT emit an add here.
            # We preserve it as a no-op normalized event for the raw log only.
            return []

        if t == "open":
            return [BookEvent(
                exchange=self.name, symbol=self.canonical_symbol,
                event_time=evt_time, receive_time=recv_time, sequence=seq,
                event_type="add",
                order_id=raw_message.get("order_id"),
                side=side,
                price=_safe_float(raw_message.get("price")),
                size=_safe_float(raw_message.get("remaining_size")),
                raw_payload=raw_message,
            )]

        if t == "done":
            # Remove the remaining portion (if any) of a resting order. The
            # book engine treats this as a cancel and looks the order up.
            return [BookEvent(
                exchange=self.name, symbol=self.canonical_symbol,
                event_time=evt_time, receive_time=recv_time, sequence=seq,
                event_type="cancel",
                order_id=raw_message.get("order_id"),
                # side/price/remaining_size are informational — the book engine
                # uses its own state. We still pass them through.
                side=side,
                price=_safe_float(raw_message.get("price")),
                size=_safe_float(raw_message.get("remaining_size")) or 0.0,
                raw_payload=raw_message,
            )]

        if t == "change":
            return [BookEvent(
                exchange=self.name, symbol=self.canonical_symbol,
                event_time=evt_time, receive_time=recv_time, sequence=seq,
                event_type="modify",
                order_id=raw_message.get("order_id"),
                side=side,
                price=_safe_float(raw_message.get("price")),
                size=_safe_float(raw_message.get("new_size")),
                raw_payload=raw_message,
            )]

        if t == "match":
            # Two effects emitted as two events:
            #  1. A 'trade' event (aggressor_side = taker's side, in `side`).
            #     This goes into trade-flow features.
            #  2. A book mutation against the MAKER order: its remaining size
            #     is reduced by `size`. We emit a 'match_fill' event carrying
            #     maker_order_id + delta; the OrderBook handles the reduction.
            match_size = _safe_float(raw_message.get("size")) or 0.0
            match_price = _safe_float(raw_message.get("price"))
            maker_order_id = raw_message.get("maker_order_id")
            # Aggressor side: Coinbase's `side` in a match is the MAKER side,
            # not the taker. The taker is on the OPPOSITE side. (Per Coinbase
            # docs: "side indicates the maker order side.")
            maker_side = side
            aggressor_side = {"bid": "ask", "ask": "bid"}.get(maker_side) if maker_side else None
            trade_ev = BookEvent(
                exchange=self.name, symbol=self.canonical_symbol,
                event_time=evt_time, receive_time=recv_time, sequence=seq,
                event_type="trade",
                trade_id=str(raw_message.get("trade_id")),
                trade_price=match_price,
                trade_size=match_size,
                aggressor_side=aggressor_side,
                raw_payload=raw_message,
            )
            out = [trade_ev]
            if maker_order_id is not None and match_size > 0:
                out.append(BookEvent(
                    exchange=self.name, symbol=self.canonical_symbol,
                    event_time=evt_time, receive_time=recv_time, sequence=seq,
                    event_type="match_fill",
                    order_id=str(maker_order_id),
                    side=maker_side,
                    price=match_price,
                    size=match_size,  # this is the AMOUNT TO SUBTRACT
                    raw_payload={"match_fill": True, "maker_order_id": maker_order_id,
                                 "size": match_size, "price": match_price},
                ))
            return out

        return [BookEvent(
            exchange=self.name, symbol=self.canonical_symbol,
            event_time=evt_time, receive_time=recv_time, sequence=seq,
            event_type="unknown",
            raw_payload=raw_message,
        )]

    # ---- sequence validation ---------------------------------------------

    def validate_sequence(self, message: object) -> Optional[str]:
        if not isinstance(message, dict):
            return None
        seq = message.get("sequence")
        if seq is None:
            return None  # heartbeat-like, fine
        if self._last_sequence is None:
            self._last_sequence = seq
            return None
        if seq <= self._last_sequence:
            # Duplicate / pre-snapshot leftover. Caller filters these out
            # during replay; once stream is live we just ignore.
            return None
        if seq != self._last_sequence + 1:
            reason = f"gap: expected {self._last_sequence + 1} got {seq}"
            self._last_sequence = seq
            return reason
        self._last_sequence = seq
        return None

    async def close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            finally:
                self._ws = None


def _safe_float(x) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
