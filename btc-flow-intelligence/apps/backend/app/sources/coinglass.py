"""
CoinGlass — derivatives positioning + aggregate liquidations.

Live endpoints require a CoinGlass API key (header `coinglassSecret`). Mock mode
returns a realistic derivatives snapshot.
"""

from __future__ import annotations

from app.config import get_settings
from app.schemas import SignalReading
from app.sources.base import SourceAdapter

settings = get_settings()


class CoinGlassAdapter(SourceAdapter):
    name = "coinglass"
    category = "derivatives"
    required_keys = ("coinglass_api_key",)
    BASE = "https://open-api-v3.coinglass.com/api"

    async def _fetch_live(self) -> list[SignalReading]:
        headers = {"coinglassSecret": settings.coinglass_api_key or "", "accept": "application/json"}
        oi = (await self._request("GET", f"{self.BASE}/futures/openInterest/ohlc-aggregated-history?symbol=BTC&interval=1d", headers=headers)).json()
        funding = (await self._request("GET", f"{self.BASE}/futures/fundingRate/oi-weight-ohlc-history?symbol=BTC&interval=1d", headers=headers)).json()

        oi_latest = oi["data"][-1]
        oi_prev = oi["data"][-2]
        oi_change = (float(oi_latest["close"]) / float(oi_prev["close"]) - 1) * 100
        fr = float(funding["data"][-1]["close"])

        return [
            SignalReading(
                category="derivatives", source=self.name, metric="open_interest",
                value=float(oi_latest["close"]), change_24h=round(oi_change, 2),
                # Rising OI + positive price usually = leverage building (mild distribution risk);
                # falling OI into strength = healthier spot-led move (accumulation).
                score=self._score_from_change(-oi_change, 3, 8), raw=oi_latest,
            ),
            SignalReading(
                category="derivatives", source=self.name, metric="funding_rate",
                value=fr, change_24h=None,
                # Very positive funding = crowded longs (distribution risk); negative = squeeze fuel.
                score=self._score_from_change(-fr * 10000, 1, 3), raw=funding["data"][-1],
            ),
        ]

    def _mock(self) -> list[SignalReading]:
        oi = self._r(18.0, 26.0, 2)          # $bn aggregated OI
        oi_change = self._r(-9.0, 9.0)
        funding = self._r(-0.015, 0.04, 4)   # 8h funding %
        ls = self._r(0.85, 1.25)             # long/short ratio
        liq_long = self._r(20, 180, 0)       # $m long liquidations 24h
        liq_short = self._r(20, 180, 0)
        liq_skew = (liq_short - liq_long) / max(liq_short + liq_long, 1) * 100

        return [
            SignalReading(category="derivatives", source=self.name, metric="open_interest",
                          value=oi, change_24h=oi_change,
                          score=self._score_from_change(-oi_change, 3, 8), raw={"unit": "$bn"}),
            SignalReading(category="derivatives", source=self.name, metric="funding_rate",
                          value=funding, change_24h=None,
                          score=self._score_from_change(-funding * 1000, 8, 25), raw={"interval": "8h"}),
            SignalReading(category="derivatives", source=self.name, metric="long_short_ratio",
                          value=ls, change_24h=None,
                          score=self._score_from_change((1 - ls) * 100, 8, 18), raw={}),
            SignalReading(category="derivatives", source=self.name, metric="liquidation_skew",
                          value=round(liq_skew, 1), change_24h=None,
                          # more shorts liquidated → upward pressure (accumulation-leaning)
                          score=self._score_from_change(liq_skew, 15, 40),
                          raw={"long_liq_m": liq_long, "short_liq_m": liq_short}),
        ]
