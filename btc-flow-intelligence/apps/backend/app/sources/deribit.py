"""
Deribit — BTC options open interest, put/call ratio and a max-pain estimate.

Deribit's public market-data API is keyless, so live mode works without
credentials when mock_mode is off.
"""

from __future__ import annotations

from collections import defaultdict

from app.schemas import SignalReading
from app.sources.base import SourceAdapter


class DeribitAdapter(SourceAdapter):
    name = "deribit"
    category = "derivatives"
    required_keys = ()
    BASE = "https://www.deribit.com/api/v2"

    @property
    def can_go_live(self) -> bool:
        from app.config import get_settings

        return not get_settings().mock_mode

    async def _fetch_live(self) -> list[SignalReading]:
        resp = await self._request(
            "GET",
            f"{self.BASE}/public/get_book_summary_by_currency?currency=BTC&kind=option",
        )
        instruments = resp.json()["result"]
        call_oi = sum(i.get("open_interest", 0) for i in instruments if i["instrument_name"].endswith("-C"))
        put_oi = sum(i.get("open_interest", 0) for i in instruments if i["instrument_name"].endswith("-P"))
        pcr = (put_oi / call_oi) if call_oi else 0.0

        # crude max-pain: strike with min total intrinsic OI weight
        strike_oi: dict[float, float] = defaultdict(float)
        for i in instruments:
            parts = i["instrument_name"].split("-")
            if len(parts) >= 3:
                try:
                    strike_oi[float(parts[2])] += i.get("open_interest", 0)
                except ValueError:
                    pass
        max_pain = max(strike_oi, key=strike_oi.get) if strike_oi else None

        return [
            SignalReading(category="derivatives", source=self.name, metric="options_put_call_ratio",
                          value=round(pcr, 3), change_24h=None,
                          score=self._score_from_change((0.7 - pcr) * 100, 5, 15),
                          raw={"call_oi": call_oi, "put_oi": put_oi, "max_pain": max_pain}),
        ]

    def _mock(self) -> list[SignalReading]:
        call_oi = self._r(180_000, 320_000, 0)
        put_oi = self._r(120_000, 280_000, 0)
        pcr = round(put_oi / call_oi, 3)
        max_pain = self._r(58_000, 70_000, 0)
        return [
            SignalReading(category="derivatives", source=self.name, metric="options_open_interest",
                          value=call_oi + put_oi, change_24h=None, score=0,
                          raw={"call_oi": call_oi, "put_oi": put_oi, "unit": "contracts"}),
            SignalReading(category="derivatives", source=self.name, metric="options_put_call_ratio",
                          value=pcr, change_24h=None,
                          # low PCR (call-heavy) = bullish positioning
                          score=self._score_from_change((0.7 - pcr) * 100, 5, 15),
                          raw={"max_pain": max_pain}),
        ]
