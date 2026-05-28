"""
CryptoQuant — on-chain exchange flows, reserves, whale ratio, miner flows,
stablecoin liquidity, and valuation metrics (MVRV / SOPR / NUPL).

This adapter intentionally emits into THREE scoring categories
(onchain_flows, stablecoin, sentiment) because CryptoQuant covers all of them.
Live mode requires an API key (Bearer token).
"""

from __future__ import annotations

from app.config import get_settings
from app.schemas import SignalReading
from app.sources.base import SourceAdapter

settings = get_settings()


class CryptoQuantAdapter(SourceAdapter):
    name = "cryptoquant"
    category = "onchain_flows"
    required_keys = ("cryptoquant_api_key",)
    BASE = "https://api.cryptoquant.com/v1"

    async def _fetch_live(self) -> list[SignalReading]:
        headers = {"Authorization": f"Bearer {settings.cryptoquant_api_key}"}
        netflow = (await self._request(
            "GET",
            f"{self.BASE}/btc/exchange-flows/netflow?window=day&exchange=all_exchange&limit=2",
            headers=headers,
        )).json()
        rows = netflow["result"]["data"]
        latest = float(rows[0]["netflow_total"])
        prev = float(rows[1]["netflow_total"])
        change = (latest - prev)
        return [
            SignalReading(
                category="onchain_flows", source=self.name, metric="exchange_netflow",
                value=latest, change_24h=change,
                # Negative netflow (coins leaving exchanges) = accumulation.
                score=self._score_from_change(-latest / 1000, 1, 4), raw=rows[0],
            )
        ]

    def _mock(self) -> list[SignalReading]:
        netflow = self._r(-9000, 9000, 0)        # BTC, negative = outflow (bullish)
        reserve_change = self._r(-2.5, 2.5)       # % change in exchange reserve
        whale_ratio = self._r(0.35, 0.65)         # top-10 inflow share; high = sell pressure
        miner_netflow = self._r(-1200, 1200, 0)   # BTC miner→exchange; positive = sell
        stable_reserve_chg = self._r(-3.0, 4.0)   # % change stablecoin exchange reserve
        mvrv = self._r(1.6, 2.8)                  # >3.7 euphoric, <1 capitulation
        sopr = self._r(0.97, 1.05)
        nupl = self._r(0.35, 0.62)

        return [
            # --- on-chain flows ---
            SignalReading(category="onchain_flows", source=self.name, metric="exchange_netflow",
                          value=netflow, change_24h=None,
                          score=self._score_from_change(-netflow / 1000, 1.0, 4.0),
                          raw={"unit": "BTC", "note": "negative = outflow"}),
            SignalReading(category="onchain_flows", source=self.name, metric="exchange_reserve_change",
                          value=reserve_change, change_24h=None,
                          score=self._score_from_change(-reserve_change, 0.5, 1.5), raw={}),
            SignalReading(category="onchain_flows", source=self.name, metric="whale_ratio",
                          value=whale_ratio, change_24h=None,
                          # high whale inflow ratio → distribution
                          score=self._score_from_change((0.5 - whale_ratio) * 100, 5, 12), raw={}),
            SignalReading(category="onchain_flows", source=self.name, metric="miner_netflow_to_exchange",
                          value=miner_netflow, change_24h=None,
                          score=self._score_from_change(-miner_netflow / 100, 4, 9),
                          raw={"unit": "BTC", "note": "positive = miners selling"}),
            # --- stablecoin liquidity ---
            SignalReading(category="stablecoin", source=self.name, metric="stablecoin_exchange_reserve_change",
                          value=stable_reserve_chg, change_24h=None,
                          # rising stablecoins on exchanges = dry powder = accumulation-leaning
                          score=self._score_from_change(stable_reserve_chg, 1.0, 2.5), raw={}),
            # --- valuation / sentiment ---
            SignalReading(category="sentiment", source=self.name, metric="mvrv",
                          value=mvrv, change_24h=None,
                          # mid-range MVRV is constructive; extreme highs = distribution
                          score=self._score_from_change((2.4 - mvrv) * 2, 0.4, 1.2), raw={}),
            SignalReading(category="sentiment", source=self.name, metric="sopr",
                          value=sopr, change_24h=None,
                          score=self._score_from_change((sopr - 1) * 100, 1, 3), raw={}),
            SignalReading(category="sentiment", source=self.name, metric="nupl",
                          value=nupl, change_24h=None,
                          score=self._score_from_change((0.5 - nupl) * 100, 5, 12), raw={}),
        ]
