"""
L3 order book.

Data structures:

  _orders : dict[order_id -> (side, price, size)]
  _bids   : SortedDict[price (desc) -> {order_id -> size}]
  _asks   : SortedDict[price (asc)  -> {order_id -> size}]

We use sortedcontainers.SortedDict for the price-indexed sides so that
best_bid() / best_ask() are O(log n) without the cost of a balanced BST in
pure Python. The per-level inner dict gives us per-order resolution AND
trivial level totals (sum of values).

Event types this engine knows about (must stay in sync with schema.EVENT_TYPES):

  snapshot, add        -> insert the order
  modify               -> change size (and possibly price) of a known order
  cancel               -> remove a known order (size already cached)
  match_fill           -> reduce the size of a known MAKER order by `size`
                          (Coinbase 'match' messages produce these alongside
                          the public 'trade' event)
  trade, heartbeat     -> no-op for the book; pass-through for downstream
  reset                -> adapter is telling us the book is unreliable;
                          wipe state. Caller must then feed a fresh snapshot.
  unknown              -> something we couldn't parse; mark corrupt so the
                          downstream supervisor will trigger a resync.

Health invariants the engine enforces:

  - best_bid < best_ask (no crossed book)
  - All sizes > 0 (zero/negative => evict)
  - Order side never flips

Corruption handling: when the engine becomes corrupt, it ignores every
subsequent mutator until reset() is called or a 'snapshot' event is
applied (which is the resync path: caller flushes a fresh snapshot in).
We track corruption_count so the reconstructor can log how often resync
was needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sortedcontainers import SortedDict

from src.schema import BookEvent
from src.utils.logging import logger


class BookCorruptionError(RuntimeError):
    """Raised when the book is in an unrecoverable state."""


@dataclass
class BookSnapshot:
    ts_unix_ms: int
    best_bid: Optional[float]
    best_ask: Optional[float]
    mid_price: Optional[float]
    spread: Optional[float]
    bid_levels: List[Tuple[float, float, int]]   # (price, total_size, n_orders)
    ask_levels: List[Tuple[float, float, int]]

    @property
    def is_valid(self) -> bool:
        return (
            self.best_bid is not None
            and self.best_ask is not None
            and self.best_bid < self.best_ask
        )


class OrderBook:
    """L3 order book with per-order tracking."""

    def __init__(self, max_depth_tracked: int = 50) -> None:
        self.max_depth_tracked = max_depth_tracked
        self._orders: Dict[str, Tuple[str, float, float]] = {}
        # SortedDict keys ascend. For bids we *negate* prices so iter() gives best-first.
        self._bids: SortedDict[float, Dict[str, float]] = SortedDict()
        self._asks: SortedDict[float, Dict[str, float]] = SortedDict()
        self._corrupt = False
        # Count of times the book has been declared corrupt during its life.
        # Useful for the daily report's data-quality section.
        self.corruption_count = 0

    # ---- mutators --------------------------------------------------------

    def apply_event(self, ev: BookEvent) -> None:
        # 'reset' is the one event that's accepted while corrupt — it wipes
        # state so the next 'snapshot' can resync the book cleanly.
        if ev.event_type == "reset":
            self.reset()
            return

        # When corrupt, only 'snapshot' (resync) is accepted; everything else
        # is discarded so we don't compound the damage.
        if self._corrupt and ev.event_type != "snapshot":
            return

        try:
            if ev.event_type in ("trade", "heartbeat"):
                return
            if ev.event_type == "snapshot":
                # A snapshot following a corrupt state is the resync. Clear
                # any stale state first so we don't carry phantom orders.
                if self._corrupt:
                    # Wipe but keep the corruption_count history.
                    self._orders.clear()
                    self._bids.clear()
                    self._asks.clear()
                    self._corrupt = False
                self._apply_add(ev, replace=True)
            elif ev.event_type == "add":
                self._apply_add(ev, replace=False)
            elif ev.event_type == "modify":
                self._apply_modify(ev)
            elif ev.event_type == "cancel":
                self._apply_cancel(ev)
            elif ev.event_type == "match_fill":
                self._apply_match_fill(ev)
            elif ev.event_type == "unknown":
                # Treat as a corruption signal for the immediate window.
                self._mark_corrupt(f"'unknown' event: {ev.raw_payload}")
        except BookCorruptionError:
            raise
        except Exception as e:
            logger.exception(f"book: failed to apply event: {e} (event={ev})")
            self._mark_corrupt(f"exception applying event: {e}")

        if not self._corrupt:
            self._check_health()

    def _mark_corrupt(self, reason: str) -> None:
        if not self._corrupt:
            self.corruption_count += 1
            logger.warning(f"book: marked corrupt: {reason}")
        self._corrupt = True

    def _apply_add(self, ev: BookEvent, replace: bool) -> None:
        if ev.order_id is None or ev.price is None or ev.side is None:
            return
        if ev.size is None or ev.size <= 0:
            return
        side, price, size = ev.side, float(ev.price), float(ev.size)
        # If we already track this order_id, treat as modify
        if not replace and ev.order_id in self._orders:
            self._apply_modify(ev)
            return
        self._orders[ev.order_id] = (side, price, size)
        levels = self._bids if side == "bid" else self._asks
        # store as negative key on the bid side so the first element is best
        key = -price if side == "bid" else price
        bucket = levels.get(key)
        if bucket is None:
            bucket = {}
            levels[key] = bucket
        bucket[ev.order_id] = size

    def _apply_modify(self, ev: BookEvent) -> None:
        if ev.order_id is None:
            return
        cur = self._orders.get(ev.order_id)
        if cur is None:
            # Modify for unknown order; treat as add if we have enough info.
            if ev.price is not None and ev.side is not None and ev.size and ev.size > 0:
                self._apply_add(ev, replace=False)
            return
        side, old_price, _ = cur
        new_size = float(ev.size) if ev.size is not None else 0.0
        new_price = float(ev.price) if ev.price is not None else old_price
        if new_size <= 0:
            self._apply_cancel(ev)
            return
        if new_price != old_price:
            # Treat as cancel + add at new price.
            self._remove_from_level(ev.order_id, side, old_price)
            self._orders[ev.order_id] = (side, new_price, new_size)
            levels = self._bids if side == "bid" else self._asks
            key = -new_price if side == "bid" else new_price
            bucket = levels.setdefault(key, {})
            bucket[ev.order_id] = new_size
        else:
            self._orders[ev.order_id] = (side, old_price, new_size)
            levels = self._bids if side == "bid" else self._asks
            key = -old_price if side == "bid" else old_price
            if key in levels:
                levels[key][ev.order_id] = new_size

    def _apply_cancel(self, ev: BookEvent) -> None:
        if ev.order_id is None:
            return
        cur = self._orders.pop(ev.order_id, None)
        if cur is None:
            return
        side, price, _ = cur
        self._remove_from_level(ev.order_id, side, price)

    def _apply_match_fill(self, ev: BookEvent) -> None:
        """
        Coinbase match: reduce the MAKER order's remaining size by ev.size.
        If the resulting size <= 0, the maker is gone (the 'done' message
        from the same wire stream will normally arrive too, but we don't
        rely on its arrival).
        """
        if ev.order_id is None or ev.size is None:
            return
        cur = self._orders.get(ev.order_id)
        if cur is None:
            # Maker order not in our book — could be a fill against an order
            # that was placed before our snapshot. Safe no-op.
            return
        side, price, old_size = cur
        new_size = old_size - float(ev.size)
        if new_size <= 0:
            # Maker fully consumed by this match.
            self._orders.pop(ev.order_id, None)
            self._remove_from_level(ev.order_id, side, price)
            return
        self._orders[ev.order_id] = (side, price, new_size)
        levels = self._bids if side == "bid" else self._asks
        key = -price if side == "bid" else price
        bucket = levels.get(key)
        if bucket is not None:
            bucket[ev.order_id] = new_size

    def _remove_from_level(self, order_id: str, side: str, price: float) -> None:
        levels = self._bids if side == "bid" else self._asks
        key = -price if side == "bid" else price
        bucket = levels.get(key)
        if bucket is None:
            return
        bucket.pop(order_id, None)
        if not bucket:
            del levels[key]

    # ---- queries ---------------------------------------------------------

    def best_bid(self) -> Optional[float]:
        if not self._bids:
            return None
        # bids are stored with negative key, so first key is best
        return -self._bids.keys()[0]

    def best_ask(self) -> Optional[float]:
        if not self._asks:
            return None
        return self._asks.keys()[0]

    def mid_price(self) -> Optional[float]:
        b, a = self.best_bid(), self.best_ask()
        if b is None or a is None:
            return None
        return (a + b) / 2.0

    def spread(self) -> Optional[float]:
        b, a = self.best_bid(), self.best_ask()
        if b is None or a is None:
            return None
        return a - b

    def depth(self, side: str, levels: int = 10) -> List[Tuple[float, float, int]]:
        """Return up to `levels` levels best-first: (price, total_size, n_orders)."""
        src = self._bids if side == "bid" else self._asks
        out: List[Tuple[float, float, int]] = []
        for k in src.keys():
            if len(out) >= levels:
                break
            price = -k if side == "bid" else k
            bucket = src[k]
            out.append((price, sum(bucket.values()), len(bucket)))
        return out

    def order_count(self) -> int:
        return len(self._orders)

    @property
    def is_corrupt(self) -> bool:
        return self._corrupt

    def reset(self) -> None:
        """Wipe all state. The caller must feed a fresh snapshot to resume."""
        if self._corrupt:
            # We're being told to reset while corrupt; that already counted.
            pass
        else:
            # Voluntary reset (e.g. adapter resync, recorder restart) — count it.
            self.corruption_count += 1
        self._orders.clear()
        self._bids.clear()
        self._asks.clear()
        self._corrupt = False

    # ---- health ----------------------------------------------------------

    def _check_health(self) -> None:
        b = self.best_bid()
        a = self.best_ask()
        if b is not None and a is not None and b >= a:
            self._mark_corrupt(f"crossed book bid={b} ask={a}")

    # ---- snapshots -------------------------------------------------------

    def snapshot(self, ts_unix_ms: int, levels: int = 10) -> BookSnapshot:
        return BookSnapshot(
            ts_unix_ms=ts_unix_ms,
            best_bid=self.best_bid(),
            best_ask=self.best_ask(),
            mid_price=self.mid_price(),
            spread=self.spread(),
            bid_levels=self.depth("bid", levels),
            ask_levels=self.depth("ask", levels),
        )
