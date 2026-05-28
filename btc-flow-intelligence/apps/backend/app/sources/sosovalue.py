"""SoSoValue — spot BTC ETF holdings, AUM and daily volume."""

from __future__ import annotations

from app.config import get_settings
from app.schemas import SignalReading
from app.sources.base import SourceAdapter

settings = get_settings()


class SoSoValueAdapter(SourceAdapter):
    name = "sosovalue"
    category = "etf_flows"
    required_keys = ("sosovalue_api_key",)
    BASE = "https://api.sosovalue.xyz/openapi/v2/etf"

    async def _fetch_live(self) -> list[SignalReading]:
        headers = {"x-soso-api-key": settings.sosovalue_api_key or "", "Content-Type": "application/json"}
        resp = await self._request(
            "POST", f"{self.BASE}/historicalInflowChart",
            headers=headers, json={"type": "us-btc-spot"},
        )
        data = resp.json()["data"]
        latest = data[-1]
        flow = float(latest.get("totalNetInflow", 0)) / 1e6
        return [
            SignalReading(
                category="etf_flows", source=self.name, metric="etf_aum_net_inflow_usd_m",
                value=round(flow, 1), change_24h=None,
                score=self._score_from_change(flow, 50, 250), raw=latest,
            )
        ]

    def _mock(self) -> list[SignalReading]:
        aum = self._r(95, 135, 1)            # $bn total spot ETF AUM
        holdings = self._r(1_050_000, 1_280_000, 0)  # BTC held
        holdings_chg = self._r(-0.6, 0.9)    # % change in BTC held
        volume = self._r(1.2, 4.8, 1)        # $bn daily volume
        return [
            SignalReading(category="etf_flows", source=self.name, metric="etf_holdings_change_pct",
                          value=holdings_chg, change_24h=None,
                          score=self._score_from_change(holdings_chg, 0.15, 0.45),
                          raw={"aum_bn": aum, "holdings_btc": holdings, "volume_bn": volume}),
        ]
