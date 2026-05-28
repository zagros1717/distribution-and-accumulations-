"""
Coinbase Exchange L3 adapter (full channel).

The 'full' websocket channel emits per-order messages:
  - received: an order has been received by the matching engine. The order may
              be a marketable order that immediately fills; it is NOT necessarily
              resting in the book. We do NOT emit an add on `received`.
  - open:     a maker order is now resting in the book. THIS is the add.
  - match:    a public execution. The MAKER order on the book is being consumed.
              We emit a 'trade' (for trade-flow features) AND a 'match_fill'
              event so the order book reduces the maker_order_id's size by `size`.
  - done:     a resting order has been filled or canceled. The remainder (if any)
              is removed from the book.
  - change:   a maker order resized at the same price (rare).

To maintain a correct L3 book Coinbase's docs prescribe this procedure:

  1. Open the websocket and start QUEUEING all incoming messages.
  2. Fetch a REST level=3 snapshot. It has a `sequence` number.
  3. Drop queued messages with sequence <= snapshot_sequence.
  4. The remaining queued messages MUST form a contiguous sequence starting at
     snapshot_sequence + 1. If they don't, the snapshot is stale — resync.
  5. Apply the remaining queued messages in order.
  6. Continue applying live messages, asserting sequence is contiguous.

CRITICAL: while the REST snapshot is being fetched (an HTTP round-trip that
takes 100s of ms), the websocket is still emitting messages. If we do not
continuously read from the socket during the REST call, the OS buffers fill
and frames may be lost; even if they aren't lost, we won't see them until
after the snapshot returns, which can hide ordering issues. We therefore start
a background "drainer" task BEFORE issuing the REST request and let it run
until phase-1 replay is complete.

Sequence handling:
  - validate_sequence returns one of:
      SequenceStatus.OK    : seq == last + 1; advance last and use the msg
      SequenceStatus.SKIP  : seq <= last (duplicate / pre-snapshot leftover);
                              caller MUST skip the message entirely
      SequenceStatus.GAP   : seq > last + 1; caller MUST raise SequenceGapError
                              so the recorder supervisor counts a failure and
                              resyncs from a fresh snapshot
  - The previous design returned strings + yielded an 'unknown' event then
    `return`. That made the iterator end cleanly, which the recorder cannot
    distinguish from a normal end-of-stream. We raise instead so failure
    propagates.

Reset semantics:
  - Before yielding the snapshot rows, fetch_snapshot() emits ONE event with
    event_type='reset'. The reconstructor wipes the book on that event and
    enters an "awaiting snapshot" state. The next rows (the snapshot itself)
    then reseed the book cleanly. This prevents phantom orders carrying over
    from a prior session after a reconnect.

Reference:
  - https://docs.cdp.coinbase.com/exchange/docs/websocket-channels
  - https://docs.cdp.coinbase.com/exchange/docs/rest-api  (GET /products/{id}/book?level=3)

Authenticated channels are NOT used. Per project rules we never send any API key.
"""
from __future__ import annotations

import asyncio
import enum
import json
from collections import deque
from typing import AsyncIterator, Deque, Iterable, Optional

import aiohttp
import websockets

from src.adapters.base import MarketDataAdapter
from src.schema import BookEvent
from src.utils.logging import logger
from src.utils.time import parse_iso_utc, utcnow


class SequenceGapError(RuntimeError):
    """
    Raised when the Coinbase stream skips a sequence number.

    Coinbase guarantees monotonic, contiguous sequences within a single
    websocket session. A gap therefore means either:
      - we lost a message (process backed up, OS buffer dropped),
      - we are still seeing pre-snapshot messages we already processed, or
      - the venue had an internal hiccup (rare).

    None of these are recoverable in place. The recorder catches this
    exception in its supervisor loop, increments its failure counter, and
    reconnects (which fetches a fresh snapshot).
    """


class SequenceStatus(enum.Enum):
    """Three-valued result of validate_sequence(). See module docstring."""
    OK = "ok"
    SKIP = "skip"   # duplicate / pre-snapshot leftover; do nothing
    GAP = "gap"     # missing message(s); caller must raise


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
        # Background coroutine that reads the websocket and appends to _queue
        # while we are mid-snapshot. Must be cancelled before phase-1 replay.
        self._drain_task: Optional[asyncio.Task] = None
        self._stop_drain = asyncio.Event()
        # Public hook for raw-frame capture (set by recorder).
        # Signature: (channel: str, raw_text: str) -> None
        self.on_raw_frame = None

    # ---- connect ----------------------------------------------------------

    async def connect(self) -> None:
        """Open the websocket and send the subscribe request. The 'subscriptions'
        ack is consumed here; everything after that goes into the queue via
        _drain_ws_to_queue() once fetch_snapshot() starts the drainer."""
        logger.info(f"[coinbase] connecting to {self.ws_url}")
        self._ws = await websockets.connect(self.ws_url, ping_interval=20, max_size=2**22)
        await self._ws.send(json.dumps({
            "type": "subscribe",
            "product_ids": [self.symbol],
            "channels": [self.channel],
        }))
        # Read until we see the subscriptions ack. We do NOT queue here —
        # _drain_ws_to_queue (started by fetch_snapshot) takes over for
        # subsequent frames.
        while True:
            raw_text = await asyncio.wait_for(self._ws.recv(), timeout=30)
            if self.on_raw_frame is not None:
                try:
                    self.on_raw_frame("ws", raw_text)
                except Exception:
                    pass
            msg = json.loads(raw_text)
            t = msg.get("type")
            if t == "subscriptions":
                logger.info(f"[coinbase] subscribed to {self.channel}")
                return
            if t == "error":
                raise RuntimeError(f"Coinbase subscribe error: {msg}")
            # Pre-ack data frames are extremely rare but technically possible.
            # Buffer them so we don't lose any pre-snapshot data.
            if isinstance(msg, dict):
                self._queue.append(msg)

    # ---- background drainer ------------------------------------------------

    async def _drain_ws_to_queue(self) -> None:
        """
        Continuously read websocket frames and append them to self._queue.

        This runs as a background task during fetch_snapshot() so that frames
        arriving DURING the REST snapshot HTTP call are not dropped or held in
        OS buffers. The drainer stops when stream_events() sets _stop_drain
        before phase-1 replay begins.
        """
        assert self._ws is not None
        try:
            while not self._stop_drain.is_set():
                # Use wait_for so we can periodically check _stop_drain rather
                # than blocking indefinitely on recv().
                try:
                    raw_text = await asyncio.wait_for(self._ws.recv(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if self.on_raw_frame is not None:
                    try:
                        self.on_raw_frame("ws", raw_text)
                    except Exception:
                        pass
                msg = json.loads(raw_text)
                if not isinstance(msg, dict):
                    continue
                if msg.get("type") in ("subscriptions", "error"):
                    continue
                self._queue.append(msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Don't let drainer crashes silently kill the queue; surface them
            # to the supervisor by re-raising on the next read attempt.
            logger.exception(f"[coinbase] drainer error: {e}")
            raise

    # ---- snapshot ---------------------------------------------------------

    async def fetch_snapshot(self) -> Iterable[BookEvent]:
        """
        REST level=3 snapshot.

        Starts the websocket drainer BEFORE making the REST call so that
        frames arriving during the HTTP round-trip are queued without loss.
        Returns the snapshot as a 'reset' event followed by one BookEvent
        (event_type='snapshot') per resting order. The reconstructor wipes
        the book on the reset, then reseeds from the snapshot rows.
        """
        # Start the drainer. It will run until stream_events() stops it.
        self._stop_drain = asyncio.Event()
        self._drain_task = asyncio.create_task(
            self._drain_ws_to_queue(), name="coinbase-drainer"
        )

        logger.info(f"[coinbase] GET {self.rest_snapshot_url}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.rest_snapshot_url, timeout=30) as resp:
                    resp.raise_for_status()
                    snapshot_text = await resp.text()
                    data = json.loads(snapshot_text)
        except Exception:
            # If the REST call fails the drainer is still running; cancel it
            # so we don't leak the task. The caller will surface the error.
            self._stop_drain.set()
            if self._drain_task is not None:
                self._drain_task.cancel()
                try:
                    await self._drain_task
                except (asyncio.CancelledError, Exception):
                    pass
                self._drain_task = None
            raise

        if self.on_raw_frame is not None:
            try:
                self.on_raw_frame("rest_snapshot", snapshot_text)
            except Exception:
                pass

        self._snapshot_sequence = int(data["sequence"])
        self._last_sequence = self._snapshot_sequence
        now = utcnow()
        events: list[BookEvent] = []

        # The reset MUST come first. The reconstructor wipes the book on this
        # event, so any orders from a prior session/snapshot are cleared
        # before the new snapshot rows are applied. Without this, a reconnect
        # would leave phantom orders in the book.
        events.append(BookEvent(
            exchange=self.name, symbol=self.canonical_symbol,
            event_time=now, receive_time=now,
            sequence=self._snapshot_sequence,
            event_type="reset",
            raw_payload={"reason": "pre-snapshot reset", "snapshot_seq": self._snapshot_sequence},
        ))

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
        Two phases:

        Phase 1 (queue drain):
          - Stop the background drainer so we have exclusive access to the
            queue.
          - Drop queued msgs with seq <= snapshot_sequence. Per Coinbase docs,
            these are pre-snapshot leftovers we MUST discard, not apply.
          - Sort the remainder by sequence (Coinbase normally sends in order,
            but we don't trust that across reconnects).
          - The first remaining seq MUST equal snapshot_sequence + 1. If not,
            the snapshot is stale and we raise SequenceGapError.
          - Apply queued msgs in order; any non-contiguous step raises.

        Phase 2 (live):
          - Read directly from the websocket.
          - For each frame: validate_sequence -> OK/SKIP/GAP.
              OK   -> normalize and yield
              SKIP -> drop silently (duplicate or pre-snapshot leftover)
              GAP  -> raise SequenceGapError; the recorder will reconnect.
        """
        assert self._ws is not None
        snap_seq = self._snapshot_sequence
        if snap_seq is None:
            raise RuntimeError("stream_events called before fetch_snapshot")

        # Stop the background drainer; we read the websocket directly from here.
        self._stop_drain.set()
        if self._drain_task is not None:
            try:
                await self._drain_task
            except (asyncio.CancelledError, Exception):
                # Surface drainer crashes as gap (forces resync).
                if self._drain_task.cancelled():
                    pass
                else:
                    exc = self._drain_task.exception()
                    if exc is not None:
                        raise SequenceGapError(f"drainer task failed: {exc}") from exc
            self._drain_task = None

        # ---- Phase 1: replay queued frames that postdate the snapshot ----
        queued_dicts = [
            m for m in self._queue
            if isinstance(m, dict) and isinstance(m.get("sequence"), int)
        ]
        # Drop pre-snapshot leftovers entirely. Per docs these are duplicates
        # of state already encoded in the snapshot.
        queued = [m for m in queued_dicts if m["sequence"] > snap_seq]
        queued.sort(key=lambda m: m["sequence"])
        # Reset last_sequence so we can validate the queue against snap_seq.
        self._last_sequence = snap_seq
        self._queue.clear()

        if queued:
            first_seq = queued[0]["sequence"]
            if first_seq != snap_seq + 1:
                raise SequenceGapError(
                    f"stale snapshot: snapshot_seq={snap_seq} "
                    f"first_queued_seq={first_seq} (expected {snap_seq+1}); "
                    f"resync required"
                )
            for msg in queued:
                status = self.validate_sequence(msg)
                if status is SequenceStatus.SKIP:
                    # Shouldn't happen here (we already filtered <= snap_seq)
                    # but skip silently if it does.
                    continue
                if status is SequenceStatus.GAP:
                    raise SequenceGapError(
                        f"gap during queued replay: expected "
                        f"{self._last_sequence} got {msg.get('sequence')}"
                    )
                for ev in self.normalize_message(msg):
                    yield ev

        # ---- Phase 2: live stream ----
        async for raw_text in self._ws:
            if self.on_raw_frame is not None:
                try:
                    self.on_raw_frame("ws", raw_text)
                except Exception:
                    pass
            msg = json.loads(raw_text)
            if not isinstance(msg, dict):
                continue
            status = self.validate_sequence(msg)
            if status is SequenceStatus.SKIP:
                # Duplicate or pre-snapshot leftover. Do not normalize, do
                # not apply, do not yield. Just drop it.
                continue
            if status is SequenceStatus.GAP:
                raise SequenceGapError(
                    f"sequence gap: expected {self._last_sequence} "
                    f"got {msg.get('sequence')}; reconnecting"
                )
            # status == OK (or message had no sequence, e.g. heartbeat).
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
            #  1. A 'trade' event (aggressor_side = taker's side).
            #  2. A 'match_fill' event so the OrderBook reduces the maker
            #     order's remaining size by `size`.
            match_size = _safe_float(raw_message.get("size")) or 0.0
            match_price = _safe_float(raw_message.get("price"))
            maker_order_id = raw_message.get("maker_order_id")
            # Coinbase: `side` in a match is the MAKER side. The taker is on
            # the OPPOSITE side. (Per docs: "side indicates the maker order side.")
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
                    size=match_size,  # AMOUNT TO SUBTRACT from the maker order
                    raw_payload={"match_fill": True, "maker_order_id": maker_order_id,
                                 "size": match_size, "price": match_price},
                ))
            return out

        # Unknown frame type — keep the raw_payload for forensics but do not
        # treat it as a book corruption signal (the order book's 'unknown'
        # handler would mark the book corrupt). Coinbase occasionally adds
        # new message types and we don't want every new one to force a resync.
        return []

    # ---- sequence validation ---------------------------------------------

    def validate_sequence(self, message: object) -> SequenceStatus:
        """
        Three-valued sequence check.

        See the SequenceStatus enum docstring.
        """
        if not isinstance(message, dict):
            return SequenceStatus.OK
        seq = message.get("sequence")
        if seq is None:
            # Messages without a sequence (e.g. some heartbeat shapes) are
            # treated as OK because they don't claim to be in the per-order
            # sequence stream.
            return SequenceStatus.OK
        if self._last_sequence is None:
            self._last_sequence = seq
            return SequenceStatus.OK
        if seq <= self._last_sequence:
            # Duplicate or pre-snapshot leftover. Caller must DROP this msg
            # entirely — do not normalize, do not yield. last_sequence is NOT
            # advanced (we already saw a same-or-newer message).
            return SequenceStatus.SKIP
        if seq != self._last_sequence + 1:
            # Missing one or more messages. Caller must raise so the recorder
            # treats the session as broken and reconnects.
            return SequenceStatus.GAP
        self._last_sequence = seq
        return SequenceStatus.OK

    async def close(self) -> None:
        # Stop the drainer if it's still running.
        self._stop_drain.set()
        if self._drain_task is not None and not self._drain_task.done():
            self._drain_task.cancel()
            try:
                await self._drain_task
            except (asyncio.CancelledError, Exception):
                pass
        self._drain_task = None
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
