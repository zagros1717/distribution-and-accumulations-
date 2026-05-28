"""
Bitfinex BTC/USD raw-book (R0) adapter.

Wire protocol summary (public docs):

  Subscribe:
    {"event":"subscribe","channel":"book","symbol":"tBTCUSD",
     "prec":"R0","len":"100"}

  Snapshot (first message after subscribed):
    [CHAN_ID, [[ORDER_ID, PRICE, AMOUNT], ...]]

  Update:
    [CHAN_ID, [ORDER_ID, PRICE, AMOUNT]]

  Heartbeat:
    [CHAN_ID, "hb"]

Semantics for R0:
  - PRICE == 0  -> delete order ORDER_ID. The wire message does NOT include
                   the side or the cancelled size, so the adapter must keep a
                   local map of order_id -> (side, price, size) and reconstruct
                   them at cancel time. Downstream features (cancel_*_size,
                   cancel_*_count, large_order_cancelled_*) depend on this.
  - AMOUNT > 0  -> bid side, size = AMOUNT
  - AMOUNT < 0  -> ask side, size = -AMOUNT
  - If we have never seen ORDER_ID before -> "add"
  - Otherwise                              -> "modify" (price/size may both change)

Bitfinex does not provide a per-message sequence number on the public book
channel; we rely on heartbeats (every ~15s) for connection liveness and on
order_id continuity for gap detection. validate_sequence() simply returns None.

Trades: the `te`/`tu` "trades" channel is a separate subscription. The first
message after subscribe is a SNAPSHOT — a list-of-lists of recent trades. We
sort them by trade timestamp (MTS, milliseconds) and emit them in chronological
order. Subsequent messages are individual `te` (execute) or `tu` (update);
we use `te` only to avoid double-counting.

References used:
  - https://docs.bitfinex.com/reference/ws-public-books
  - https://docs.bitfinex.com/reference/ws-public-trades
"""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Dict, Iterable, List, Optional, Tuple

import websockets

from src.adapters.base import MarketDataAdapter
from src.schema import BookEvent
from src.utils.logging import logger
from src.utils.time import from_unix_millis, utcnow


class BitfinexRawBookAdapter(MarketDataAdapter):
    """Bitfinex raw L3-style book + trades."""

    name = "bitfinex"

    def __init__(
        self,
        symbol: str = "tBTCUSD",
        canonical_symbol: str = "BTCUSD",
        ws_url: str = "wss://api-pub.bitfinex.com/ws/2",
        book_precision: str = "R0",
        book_length: str = "100",
    ) -> None:
        self.symbol = symbol
        self.canonical_symbol = canonical_symbol
        self.ws_url = ws_url
        self.book_precision = book_precision
        self.book_length = book_length

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._book_chan_id: Optional[int] = None
        self._trade_chan_id: Optional[int] = None
        # order_id -> (side, price, size). Used to enrich cancels with the
        # side and size of the order being removed.
        self._orders: Dict[int, Tuple[str, float, float]] = {}
        self._snapshot_received = False
        self._last_heartbeat_ts: Optional[float] = None
        self._trade_buffer: List[list] = []
        # Public hook for raw-frame capture (set by recorder).
        self.on_raw_frame = None

    # ----- connect / subscribe ----------------------------------------------

    async def connect(self) -> None:
        logger.info(f"[bitfinex] connecting to {self.ws_url}")
        self._ws = await websockets.connect(self.ws_url, ping_interval=20)
        # Subscribe to raw book.
        await self._ws.send(json.dumps({
            "event": "subscribe",
            "channel": "book",
            "symbol": self.symbol,
            "prec": self.book_precision,
            "len": self.book_length,
        }))
        # Subscribe to trades.
        await self._ws.send(json.dumps({
            "event": "subscribe",
            "channel": "trades",
            "symbol": self.symbol,
        }))
        # Bitfinex replies with "subscribed" events containing chanId. Wait for both.
        subscribed: Dict[str, int] = {}
        while len(subscribed) < 2:
            raw_text = await asyncio.wait_for(self._ws.recv(), timeout=30)
            if self.on_raw_frame is not None:
                try:
                    self.on_raw_frame("ws", raw_text)
                except Exception:
                    pass
            msg = json.loads(raw_text)
            if isinstance(msg, dict) and msg.get("event") == "subscribed":
                subscribed[msg["channel"]] = msg["chanId"]
            elif isinstance(msg, dict) and msg.get("event") == "error":
                raise RuntimeError(f"Bitfinex subscribe error: {msg}")
        self._book_chan_id = subscribed["book"]
        self._trade_chan_id = subscribed["trades"]
        logger.info(
            f"[bitfinex] subscribed: book chan={self._book_chan_id} "
            f"trades chan={self._trade_chan_id}"
        )

    # ----- snapshot ---------------------------------------------------------

    async def fetch_snapshot(self) -> Iterable[BookEvent]:
        """
        For Bitfinex the book 'snapshot' is the first message on the book
        channel AFTER subscribe (a list-of-lists). The first trades-channel
        message after subscribe is also a snapshot — a list of recent trades.

        We pump until we have the BOOK snapshot, while buffering trade frames
        so stream_events() can emit them in chronological order.

        Returns a 'reset' event followed by one BookEvent(event_type='snapshot')
        per resting order. The reset is what tells the reconstructor to wipe
        the book before reseeding from this snapshot — without it, a reconnect
        would leave phantom orders from the previous session.
        """
        assert self._ws is not None
        self._trade_buffer = []
        # Forget any orders we tracked in a previous session; the new
        # snapshot is authoritative.
        self._orders.clear()
        while True:
            raw_text = await asyncio.wait_for(self._ws.recv(), timeout=30)
            if self.on_raw_frame is not None:
                try:
                    self.on_raw_frame("ws", raw_text)
                except Exception:
                    pass
            msg = json.loads(raw_text)
            if isinstance(msg, dict):
                continue  # info/event messages, ignore here
            if not isinstance(msg, list) or len(msg) < 2:
                continue
            chan_id, payload = msg[0], msg[1]
            if chan_id == self._book_chan_id:
                if isinstance(payload, list) and payload and isinstance(payload[0], list):
                    self._snapshot_received = True
                    now = utcnow()
                    reset_ev = BookEvent(
                        exchange=self.name, symbol=self.canonical_symbol,
                        event_time=now, receive_time=now,
                        sequence=None, event_type="reset",
                        raw_payload={"reason": "pre-snapshot reset",
                                     "chan_id": chan_id},
                    )
                    snap_events = list(self._normalize_book_message(payload, is_snapshot=True))
                    logger.info(f"[bitfinex] snapshot received: {len(snap_events)} levels")
                    return [reset_ev] + snap_events
            elif chan_id == self._trade_chan_id:
                self._trade_buffer.append(msg)

    # ----- stream -----------------------------------------------------------

    async def stream_events(self) -> AsyncIterator[BookEvent]:
        assert self._ws is not None
        # Replay buffered trade frames in chronological order.
        for ev in self._drain_trade_buffer():
            yield ev

        async for raw_text in self._ws:
            if self.on_raw_frame is not None:
                try:
                    self.on_raw_frame("ws", raw_text)
                except Exception:
                    pass
            msg = json.loads(raw_text)
            if isinstance(msg, dict):
                continue
            for ev in self.normalize_message(msg):
                yield ev

    def _drain_trade_buffer(self) -> Iterable[BookEvent]:
        """
        Emit trades from the snapshot/early-buffer phase, sorted by trade time.

        The trades snapshot is `[CHAN, [[ID,MTS,AMOUNT,PRICE], ...]]` — a list of
        recent trades, NEWEST FIRST per docs. We must sort ascending by MTS so
        downstream sees them in arrival order.
        """
        flattened: List[Tuple[int, BookEvent]] = []  # (mts, event)
        for raw_msg in self._trade_buffer:
            if len(raw_msg) < 2:
                continue
            payload = raw_msg[1]
            # The trade-channel snapshot is a list-of-lists.
            if isinstance(payload, list) and payload and isinstance(payload[0], list):
                for row in payload:
                    if not isinstance(row, list) or len(row) < 4:
                        continue
                    ev = self._build_trade_event(row)
                    if ev is not None:
                        flattened.append((int(row[1]), ev))
            else:
                # Individual update [CHAN, "te"|"tu", [ID,MTS,AMOUNT,PRICE]] or
                # [CHAN, [ID,MTS,AMOUNT,PRICE]] — _normalize_trade_message handles tagging.
                for ev in self._normalize_trade_message(raw_msg):
                    mts = ev.raw_payload.get("mts", 0)
                    flattened.append((int(mts) if mts is not None else 0, ev))
        flattened.sort(key=lambda pair: pair[0])
        self._trade_buffer = []
        for _, ev in flattened:
            yield ev

    # ----- normalize --------------------------------------------------------

    def normalize_message(self, raw_message: object) -> Iterable[BookEvent]:
        """Convert one Bitfinex wire frame into 0..N BookEvents."""
        if not isinstance(raw_message, list) or len(raw_message) < 2:
            return []

        chan_id, payload = raw_message[0], raw_message[1]

        # Heartbeat: [CHAN, "hb"]
        if payload == "hb":
            self._last_heartbeat_ts = utcnow().timestamp()
            return [
                BookEvent(
                    exchange=self.name,
                    symbol=self.canonical_symbol,
                    event_time=utcnow(),
                    receive_time=utcnow(),
                    sequence=None,
                    event_type="heartbeat",
                    raw_payload={"chan_id": chan_id, "type": "hb"},
                )
            ]

        if chan_id == self._book_chan_id:
            # First snapshot is a list-of-lists; updates are flat [id, price, amount].
            if isinstance(payload, list) and payload and isinstance(payload[0], list):
                return list(self._normalize_book_message(payload, is_snapshot=True))
            return list(self._normalize_book_message(payload, is_snapshot=False))

        if chan_id == self._trade_chan_id:
            # Trade SNAPSHOT inside the stream phase (e.g. after resubscribe).
            # If payload is a list-of-lists, emit in mts order.
            if isinstance(payload, list) and payload and isinstance(payload[0], list):
                rows = sorted(
                    [r for r in payload if isinstance(r, list) and len(r) >= 4],
                    key=lambda r: int(r[1]),
                )
                out: List[BookEvent] = []
                for r in rows:
                    ev = self._build_trade_event(r)
                    if ev is not None:
                        out.append(ev)
                return out
            return list(self._normalize_trade_message(raw_message))

        return []

    def _normalize_book_message(self, payload, is_snapshot: bool) -> Iterable[BookEvent]:
        now = utcnow()
        rows = payload if is_snapshot else [payload]
        for row in rows:
            if not isinstance(row, list) or len(row) < 3:
                continue
            try:
                order_id = int(row[0])
                price = float(row[1])
                amount = float(row[2])
            except (TypeError, ValueError):
                continue

            if price == 0.0:
                # Cancel. Look up the order in our local map so we can attach
                # its side and the size being removed. If we don't know the
                # order, emit the cancel without side/size; the book engine
                # will be a no-op and downstream features will still tolerate
                # missing side via fillna.
                cached = self._orders.pop(order_id, None)
                if cached is not None:
                    side, cached_price, cached_size = cached
                    yield BookEvent(
                        exchange=self.name,
                        symbol=self.canonical_symbol,
                        event_time=now,
                        receive_time=now,
                        sequence=None,
                        event_type="cancel",
                        order_id=str(order_id),
                        side=side,
                        price=cached_price,
                        size=cached_size,  # size REMOVED from the book
                        raw_payload={"order_id": order_id, "price": price, "amount": amount},
                    )
                else:
                    yield BookEvent(
                        exchange=self.name,
                        symbol=self.canonical_symbol,
                        event_time=now,
                        receive_time=now,
                        sequence=None,
                        event_type="cancel",
                        order_id=str(order_id),
                        side=None,
                        price=None,
                        size=None,
                        raw_payload={"order_id": order_id, "price": price, "amount": amount,
                                     "note": "cancel for unknown order_id"},
                    )
                continue

            # Add or modify.
            side = "bid" if amount > 0 else "ask"
            size = abs(amount)
            event_type = "modify" if order_id in self._orders else ("snapshot" if is_snapshot else "add")
            self._orders[order_id] = (side, price, size)

            yield BookEvent(
                exchange=self.name,
                symbol=self.canonical_symbol,
                event_time=now,  # Bitfinex doesn't stamp book updates server-side
                receive_time=now,
                sequence=None,
                event_type=event_type,
                order_id=str(order_id),
                side=side,
                price=price,
                size=size,
                raw_payload={"order_id": order_id, "price": price, "amount": amount},
            )

    def _build_trade_event(self, row: list) -> Optional[BookEvent]:
        """Convert a [ID, MTS, AMOUNT, PRICE] row into a trade BookEvent."""
        if not isinstance(row, list) or len(row) < 4:
            return None
        try:
            trade_id = row[0]
            mts = int(row[1])
            amount = float(row[2])
            price = float(row[3])
        except (TypeError, ValueError):
            return None
        side = "bid" if amount > 0 else "ask"  # buyer-aggressor if amount>0
        return BookEvent(
            exchange=self.name,
            symbol=self.canonical_symbol,
            event_time=from_unix_millis(mts),
            receive_time=utcnow(),
            sequence=None,
            event_type="trade",
            trade_id=str(trade_id),
            trade_price=price,
            trade_size=abs(amount),
            aggressor_side=side,
            raw_payload={"trade_id": trade_id, "mts": mts, "amount": amount, "price": price},
        )

    def _normalize_trade_message(self, raw_message) -> Iterable[BookEvent]:
        """
        Bitfinex trades: [CHAN, [ID, MTS, AMOUNT, PRICE]]            (te/tu) or
                         [CHAN, "te", [ID, MTS, AMOUNT, PRICE]]
                         [CHAN, "tu", [ID, MTS, AMOUNT, PRICE]]
        Use "te" only (live execution) to avoid double-counting "tu" updates.
        """
        if len(raw_message) < 2:
            return
        tag = raw_message[1] if isinstance(raw_message[1], str) else None
        if tag == "tu":
            return  # tu is the post-trade update; te already counted it
        payload = raw_message[2] if tag else raw_message[1]
        ev = self._build_trade_event(payload) if isinstance(payload, list) else None
        if ev is not None:
            yield ev

    # ----- sequence validation ---------------------------------------------

    def validate_sequence(self, message: object) -> Optional[str]:
        """
        Bitfinex public book has no server-side sequence number. Liveness is
        signalled by heartbeats every ~15s (handled by the watchdog in the
        recorder). Per-message validation is therefore a no-op.
        """
        return None

    async def close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            finally:
                self._ws = None
