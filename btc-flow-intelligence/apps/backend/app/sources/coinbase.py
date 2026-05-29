"""Coinbase Exchange public BTC-USD market-structure signals.

Uses unauthenticated public order-book and recent-trades endpoints to measure
short-term bid/ask pressure and aggressive buy/sell flow. These are exchange
microstructure signals, not on-chain wallet-flow metrics.
"""

from __future__ import annotations

from app.config import get_settings
from app.schemas import SignalReading
from app.sources.base import SourceAdapter

settings = get_settings()


class CoinbaseAdapter(SourceAdapter):
    name = "coinbase"
    category = "market_structure"
    required_keys = ()
    BASE = "https://api.exchange.coinbase.com/products/BTC-USD"

    @property
    def can_go_live(self) -> bool:
        return not settings.mock_mode

    async def _fetch_live(self) -> list[SignalReading]:
        book = (await self._request("GET", f"{self.BASE}/book?level=2")).json()
        trades = (await self._request("GET", f"{self.BASE}/trades?limit=100")).json()

        bids = [(float(row[0]), float(row[1])) for row in book.get("bids", [])[:50]]
        asks = [(float(row[0]), float(row[1])) for row in book.get("asks", [])[:50]]
        if not bids or not asks:
            raise ValueError("Coinbase order book returned no bids or asks")

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask) / 2
        lower = mid * 0.995
        upper = mid * 1.005
        bid_notional = sum(price * size for price, size in bids if price >= lower)
        ask_notional = sum(price * size for price, size in asks if price <= upper)
        depth_total = bid_notional + ask_notional
        book_imbalance = ((bid_notional - ask_notional) / depth_total * 100) if depth_total else 0.0
        spread_bps = ((best_ask - best_bid) / mid * 10000) if mid else 0.0

        aggressive_buy_notional = 0.0
        aggressive_sell_notional = 0.0
        for trade in trades:
            notional = float(trade["price"]) * float(trade["size"])
            # Coinbase trade side is maker side: sell maker removed = aggressive buy.
            if trade.get("side") == "sell":
                aggressive_buy_notional += notional
            elif trade.get("side") == "buy":
                aggressive_sell_notional += notional
        trade_total = aggressive_buy_notional + aggressive_sell_notional
        trade_imbalance = (
            (aggressive_buy_notional - aggressive_sell_notional) / trade_total * 100
            if trade_total else 0.0
        )

        return [
            SignalReading(
                category="market_structure",
                source=self.name,
                metric="orderbook_imbalance",
                value=round(book_imbalance, 2),
                change_24h=None,
                score=self._score_from_change(book_imbalance, 5, 15),
                raw={
                    "product": "BTC-USD",
                    "depth_band_pct": 0.5,
                    "bid_depth_usd": round(bid_notional, 2),
                    "ask_depth_usd": round(ask_notional, 2),
                    "spread_bps": round(spread_bps, 3),
                },
            ),
            SignalReading(
                category="market_structure",
                source=self.name,
                metric="recent_trade_imbalance",
                value=round(trade_imbalance, 2),
                change_24h=None,
                score=self._score_from_change(trade_imbalance, 8, 20),
                raw={
                    "product": "BTC-USD",
                    "trades_sampled": len(trades),
                    "aggressive_buy_usd": round(aggressive_buy_notional, 2),
                    "aggressive_sell_usd": round(aggressive_sell_notional, 2),
                },
            ),
        ]

    def _mock(self) -> list[SignalReading]:
        return []
