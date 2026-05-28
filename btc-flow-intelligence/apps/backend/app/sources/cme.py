"""
CME — institutional BTC futures/options open interest and basis.

CME's CVOL/market-data feeds are licensed; without a data subscription this
adapter runs on mock data. The live stub documents the expected shape.
"""

from __future__ import annotations

from app.schemas import SignalReading
from app.sources.base import SourceAdapter


class CMEAdapter(SourceAdapter):
    name = "cme"
    category = "derivatives"
    required_keys = ("kaiko_api_key",)  # CME data typically reached via a vendor (e.g. Kaiko)

    async def _fetch_live(self) -> list[SignalReading]:
        # Placeholder for a licensed CME data vendor call. Kept minimal on purpose:
        # without a verified vendor contract we do not ship a fabricated endpoint.
        raise NotImplementedError("CME live feed requires a licensed market-data vendor")

    def _mock(self) -> list[SignalReading]:
        oi = self._r(28_000, 38_000, 0)      # contracts
        oi_change = self._r(-6.0, 6.0)
        basis = self._r(4.0, 14.0)           # annualised front-month basis %
        return [
            SignalReading(category="derivatives", source=self.name, metric="cme_open_interest",
                          value=oi, change_24h=oi_change,
                          score=self._score_from_change(oi_change, 2, 5),
                          raw={"unit": "contracts"}),
            SignalReading(category="derivatives", source=self.name, metric="cme_annualised_basis",
                          value=basis, change_24h=None,
                          # healthy positive basis = institutional carry demand (mild accumulation)
                          score=self._score_from_change(basis - 8, 2, 4), raw={}),
        ]
