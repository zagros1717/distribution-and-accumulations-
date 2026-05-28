"""
Kaiko — market microstructure: spot vs futures volume, order-book imbalance,
liquidity depth and spread. Licensed API (Bearer key); mock fallback otherwise.
"""

from __future__ import annotations

from app.config import get_settings
from app.schemas import SignalReading
from app.sources.base import SourceAdapter

settings = get_settings()


class KaikoAdapter(SourceAdapter):
    name = "kaiko"
    category = "market_structure"
    required_keys = ("kaiko_api_key",)
    BASE = "https://us.market-api.kaiko.io/v2/data"

    async def _fetch_live(self) -> list[SignalReading]:
        headers = {"X-Api-Key": settings.kaiko_api_key or "", "Accept": "application/json"}
        resp = await self._request(
            "GET",
            f"{self.BASE}/trades.v1/spot_exchange_rate/btc/usd?interval=1d&page_size=2",
            headers=headers,
        )
        rows = resp.json().get("data", [])
        vol = float(rows[0].get("volume", 0)) if rows else 0.0
        return [
            SignalReading(
                category="market_structure", source=self.name, metric="spot_volume",
                value=vol, change_24h=None, score=0, raw=rows[0] if rows else {},
            )
        ]

    def _mock(self) -> list[SignalReading]:
        spot_vol = self._r(8.0, 26.0, 1)        # $bn 24h spot
        fut_vol = self._r(30.0, 95.0, 1)        # $bn 24h futures
        spot_fut_ratio = round(spot_vol / fut_vol, 3)
        ob_imbalance = self._r(-18.0, 18.0)     # % bid-ask depth imbalance (+ = bid heavy)
        depth_2pct = self._r(180, 640, 0)       # $m within ±2%
        spread_bps = self._r(0.4, 3.5)
        return [
            SignalReading(category="market_structure", source=self.name, metric="spot_volume",
                          value=spot_vol, change_24h=None, score=0, raw={"unit": "$bn"}),
            SignalReading(category="market_structure", source=self.name, metric="futures_volume",
                          value=fut_vol, change_24h=None, score=0, raw={"unit": "$bn"}),
            SignalReading(category="market_structure", source=self.name, metric="spot_futures_ratio",
                          value=spot_fut_ratio, change_24h=None,
                          # higher spot share = healthier, spot-led demand
                          score=self._score_from_change((spot_fut_ratio - 0.25) * 100, 4, 10), raw={}),
            SignalReading(category="market_structure", source=self.name, metric="orderbook_imbalance",
                          value=round(ob_imbalance, 1), change_24h=None,
                          score=self._score_from_change(ob_imbalance, 6, 14),
                          raw={"depth_2pct_m": depth_2pct, "spread_bps": spread_bps}),
        ]
