"""CryptoQuant market-data adapter.

The configured CryptoQuant plan exposes BTC daily price OHLCV data, not the
premium exchange-flow/on-chain endpoints.  This adapter therefore uses the
available BTC market-data endpoint and never represents synthetic on-chain
flow metrics as live CryptoQuant readings.
"""

from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.schemas import SignalReading
from app.sources.base import SourceAdapter

settings = get_settings()


class CryptoQuantAdapter(SourceAdapter):
    name = "cryptoquant"
    category = "market_structure"
    required_keys = ("cryptoquant_api_key",)
    BASE = "https://api.cryptoquant.com/v1"

    @staticmethod
    def _close(row: dict[str, Any]) -> float:
        """Handle likely CryptoQuant OHLCV close-field variants defensively."""
        for key in ("close", "price_close", "close_price", "price"):
            if row.get(key) is not None:
                return float(row[key])
        raise ValueError("CryptoQuant price-ohlcv response contains no close field")

    async def _fetch_live(self) -> list[SignalReading]:
        headers = {"Authorization": f"Bearer {settings.cryptoquant_api_key}"}
        payload = (await self._request(
            "GET",
            f"{self.BASE}/btc/market-data/price-ohlcv?window=day&limit=2",
            headers=headers,
        )).json()
        rows = payload["result"]["data"]
        if len(rows) < 2:
            raise ValueError("CryptoQuant price-ohlcv requires two daily rows")
        latest = self._close(rows[0])
        previous = self._close(rows[1])
        change_pct = ((latest - previous) / previous * 100) if previous else 0.0
        return [
            SignalReading(
                category="market_structure",
                source=self.name,
                metric="btc_daily_close",
                value=latest,
                change_24h=change_pct,
                score=0,  # Context/confirmation only; avoid double-counting spot price.
                raw={"window": "day", "latest": rows[0], "previous": rows[1]},
            )
        ]

    def _mock(self) -> list[SignalReading]:
        close = self._r(58_000, 76_000, 0)
        change = self._r(-4.5, 4.5)
        return [
            SignalReading(
                category="market_structure",
                source=self.name,
                metric="btc_daily_close",
                value=close,
                change_24h=change,
                score=0,
                raw={"mock": True, "window": "day"},
            )
        ]
