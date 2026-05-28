"""BTC spot price + 24h change. Uses CoinGecko's keyless public endpoint."""

from __future__ import annotations

from app.schemas import SignalReading
from app.sources.base import SourceAdapter


class PriceAdapter(SourceAdapter):
    name = "coingecko"
    category = "market_structure"
    required_keys = ()  # public endpoint

    @property
    def can_go_live(self) -> bool:
        from app.config import get_settings

        return not get_settings().mock_mode

    async def _fetch_live(self) -> list[SignalReading]:
        url = (
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=bitcoin&vs_currencies=usd&include_24hr_change=true"
        )
        resp = await self._request("GET", url)
        d = resp.json()["bitcoin"]
        price = float(d["usd"])
        change = float(d.get("usd_24h_change", 0.0))
        return [
            SignalReading(
                category="market_structure",
                source=self.name,
                metric="btc_price",
                value=price,
                change_24h=change,
                score=0,  # price itself doesn't vote; it's context
                raw=d,
            )
        ]

    def _mock(self) -> list[SignalReading]:
        price = self._r(58_000, 72_000, 0)
        change = self._r(-4.5, 4.5)
        return [
            SignalReading(
                category="market_structure",
                source=self.name,
                metric="btc_price",
                value=price,
                change_24h=change,
                score=0,
                raw={"mock": True},
            )
        ]
